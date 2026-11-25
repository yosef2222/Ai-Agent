"""Forward HTTPS for the allowlist and the single local Ollama endpoint."""

import http.client
import json
import logging
import select
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from agent.ollama_config import CONNECT_TIMEOUT, OLLAMA_MODEL, RESPONSE_TIMEOUT

LOGGER = logging.getLogger(__name__)
ALLOWED = {
    "api.telegram.org:443",
    "github.com:443",
    "duckduckgo.com:443",
    "links.duckduckgo.com:443",
    "html.duckduckgo.com:443",
    "lite.duckduckgo.com:443",
    "api.mwmbl.org:443",
    "en.wiktionary.org:443",
}
OLLAMA = "http://localhost:11434/api/chat"
LIMIT = 4 * 1024 * 1024
TIMEOUT = 30
MODEL_LOAD_TIMEOUT = 120
_model_loader = ThreadPoolExecutor(max_workers=1)
_model_lock = Lock()
_model_load = None
_model_ready_until = 0


def _mark_model_ready(seconds=240):
    global _model_ready_until
    with _model_lock:
        # Keep a shorter lease than Ollama's five-minute keep-alive.
        _model_ready_until = time.monotonic() + seconds


def _load_model():
    payload = json.dumps({"model": OLLAMA_MODEL, "messages": [], "stream": False, "keep_alive": "5m"})
    with closing(http.client.HTTPConnection("host.docker.internal", 11434, timeout=CONNECT_TIMEOUT)) as upstream:
        try:
            upstream.request("POST", "/api/chat", payload, {"Content-Type": "application/json"})
            upstream.sock.settimeout(MODEL_LOAD_TIMEOUT)
            response = upstream.getresponse()
            response.read(LIMIT + 1)
            status = response.status
        except (OSError, http.client.HTTPException):
            return 502
    if status == 200:
        _mark_model_ready()
    return status


def _wait_for_model(timeout):
    global _model_load
    with _model_lock:
        if time.monotonic() < _model_ready_until:
            return 200
        if _model_load is None or _model_load.done():
            _model_load = _model_loader.submit(_load_model)
        load = _model_load
    return load.result(timeout=timeout)


class Gateway(BaseHTTPRequestHandler):
    timeout = TIMEOUT

    def log_message(self, format, *args):
        # Request targets are deliberately absent from gateway logs.
        return

    def do_CONNECT(self):
        destination = self.path.lower()
        if destination not in ALLOWED:
            self.send_error(403, "Destination is not allowed")
            return
        deadline = time.monotonic() + TIMEOUT
        try:
            upstream = socket.create_connection((destination.removesuffix(":443"), 443), TIMEOUT)
        except OSError:
            self.send_error(502, "Could not reach destination")
            return
        with upstream:
            try:
                self.send_response(200, "Connection established")
                self.end_headers()
                peers = {self.connection: upstream, upstream: self.connection}
                while (remaining := deadline - time.monotonic()) > 0:
                    readable, _, _ = select.select(peers, [], [], remaining)
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        peers[source].settimeout(max(0.001, deadline - time.monotonic()))
                        peers[source].sendall(data)
            except OSError:
                return

    def do_POST(self):
        try:
            self._forward_ollama()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _forward_ollama(self):
        if self.path != OLLAMA:
            self.send_error(403, "Destination is not allowed")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= LIMIT:
            self.send_error(400, "Invalid request size")
            return
        payload = self.rfile.read(size)
        deadline = time.monotonic() + RESPONSE_TIMEOUT * 0.98
        try:
            status = _wait_for_model(max(0, deadline - time.monotonic()))
        except TimeoutError:
            self.send_error(504, "Ollama is warming up")
            return
        if status != 200:
            self.send_error(404 if status == 404 else 502, "Could not load the Ollama model")
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self.send_error(504, "Ollama is warming up")
            return
        with closing(http.client.HTTPConnection("host.docker.internal", 11434,
                                               timeout=min(CONNECT_TIMEOUT, remaining))) as upstream:
            try:
                upstream.request("POST", "/api/chat", payload, {"Content-Type": "application/json"})
                upstream.sock.settimeout(max(0.001, deadline - time.monotonic()))
                response = upstream.getresponse()
                data = response.read(LIMIT + 1)
            except TimeoutError:
                _mark_model_ready(0)
                LOGGER.warning("Ollama exceeded the %s-second response timeout.", RESPONSE_TIMEOUT)
                self.send_error(504, "Ollama response timed out")
                return
            except (OSError, http.client.HTTPException):
                _mark_model_ready(0)
                self.send_error(502, "Could not read the Ollama response")
                return
        if len(data) > LIMIT:
            self.send_error(502, "Ollama response exceeded the size limit")
            return
        if response.status == 200:
            _mark_model_ready()
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Gateway).serve_forever()
