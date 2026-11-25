"""Timeout settings shared by the planner and its local gateway."""

import os

CONNECT_TIMEOUT = 5
RESPONSE_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))
if RESPONSE_TIMEOUT < 1:
    raise ValueError("OLLAMA_TIMEOUT_SECONDS must be a positive number of seconds.")
