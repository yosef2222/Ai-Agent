"""Run Python snippets in a resource-limited subprocess."""

import json
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent.results import failure, success
from agent.skills.code_policy import validate

MAX_REQUEST = 512 * 1024
MAX_OUTPUT = 64 * 1024


def execute(code, timeout):
    validate(code, timeout)
    with tempfile.TemporaryDirectory(dir="/workspace") as directory:
        with tempfile.TemporaryFile(dir=directory) as stdout, tempfile.TemporaryFile(dir=directory) as stderr:
            try:
                process = subprocess.run(
                    [sys.executable, "-I", "-B", "/app/snippet.py"], input=code.encode("utf-8"),
                    stdout=stdout, stderr=stderr, timeout=timeout, cwd=directory,
                    env={"LANG": "C.UTF-8", "HOME": directory},
                )
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                exit_code = None
            stdout.seek(0)
            stderr.seek(0)
            output = {
                "stdout": stdout.read(MAX_OUTPUT).decode("utf-8", "replace"),
                "stderr": stderr.read(MAX_OUTPUT).decode("utf-8", "replace"),
                "exit_code": exit_code,
            }
    if exit_code is None:
        return failure("Code execution timed out.", output)
    if exit_code != 0:
        return failure("The snippet exited unsuccessfully.", output)
    return success(output)


class Runner(BaseHTTPRequestHandler):
    timeout = 30

    def log_message(self, format, *args):
        return

    def do_POST(self):
        if self.path != "/run":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST:
                raise ValueError("Invalid snippet request size.")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("Invalid snippet request.")
            result = execute(request.get("code"), request.get("timeout"))
        except (ValueError, OSError) as error:
            result = failure(str(error))
        except Exception:
            result = failure("The code runner could not execute the snippet.")
        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except OSError:
            pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8081), Runner).serve_forever()
