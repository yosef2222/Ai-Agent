"""Timeout settings shared by the planner and its local gateway."""

import os

CONNECT_TIMEOUT = 5
OLLAMA_MODEL = "llama3.1"
MAX_RETRIES = 3
RESPONSE_TIMEOUT = min(30, int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "30")))
if RESPONSE_TIMEOUT < 1:
    raise ValueError("OLLAMA_TIMEOUT_SECONDS must be a positive number of seconds.")
