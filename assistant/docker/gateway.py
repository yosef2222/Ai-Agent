"""Forward HTTPS for the allowlist and the single local Ollama endpoint."""

import http.client
import logging
import select
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.ollama_config import CONNECT_TIMEOUT, RESPONSE_TIMEOUT

LOGGER = logging.getLogger(__name__)
ALLOWED = {
    "api.telegram.org:443",
    "github.com:443",
    "duckduckgo.com:443",
    "links.duckduckgo.com:443",
    "html.duckduckgo.com:443",
    "lite.duckduckgo.com:443",
}
OLLAMA = "http://localhost:11434/api/chat"
LIMIT = 4 * 1024 * 1024
TIMEOUT = 30


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
        upstream = http.client.HTTPConnection("host.docker.internal", 11434, timeout=CONNECT_TIMEOUT)
        try:
            upstream.request("POST", "/api/chat", self.rfile.read(size), {"Content-Type": "application/json"})
        except (OSError, http.client.HTTPException):
            upstream.close()
            self.send_error(502, "Could not reach Ollama")
            return
        try:
            upstream.sock.settimeout(RESPONSE_TIMEOUT)
            response = upstream.getresponse()
            data = response.read(LIMIT + 1)
        except TimeoutError:
            LOGGER.warning("Ollama exceeded the %s-second response timeout.", RESPONSE_TIMEOUT)
            self.send_error(504, "Ollama response timed out")
            return
        except (OSError, http.client.HTTPException):
            self.send_error(502, "Could not read the Ollama response")
            return
        finally:
            upstream.close()
        if len(data) > LIMIT:
            self.send_error(502, "Ollama response exceeded the size limit")
            return
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Gateway).serve_forever()
