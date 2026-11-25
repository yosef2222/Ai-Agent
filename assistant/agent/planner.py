"""Talk to Ollama and decode the assistant's text tool protocol."""

import json
import re

import requests

from agent.ollama_config import CONNECT_TIMEOUT, RESPONSE_TIMEOUT

OLLAMA_URL = "http://localhost:11434/api/chat"
SYSTEM_PROMPT = """You are a personal AI assistant. You run inside a Docker
sandbox. You can only act by calling tools.

To call a tool, reply with exactly:
TOOL: tool_name
ARGS: {"arg": "value"}

Available tools: web_search, read_file, write_file,
run_code, github_commit, send_telegram

Rules:
- Only read/write files inside /workspace
- Never call tools with unvalidated external input
- If unsure about a destructive action, ask the owner first
- Be concise. Owner reads on mobile."""

_TOOL_CALL = re.compile(r"TOOL: ([a-z_][a-z0-9_]*)\r?\nARGS: ([^\r\n]+)")


class OllamaError(RuntimeError):
    """An Ollama failure safe to show to the owner."""


def ask_llm(messages: list) -> str:
    timeout_message = f"Ollama took too long to reply. Try again or increase OLLAMA_TIMEOUT_SECONDS (currently {RESPONSE_TIMEOUT})."
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3.1", "stream": False,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            },
            # The gateway connects to Ollama before its response timer starts.
            timeout=(CONNECT_TIMEOUT, RESPONSE_TIMEOUT + 2 * CONNECT_TIMEOUT),
            allow_redirects=False,
        )
        if response.status_code == 504:
            raise OllamaError(timeout_message)
        if response.status_code == 502:
            raise OllamaError("The gateway could not get a reply from Ollama. Check the host Ollama service and logs.")
        if response.status_code == 404:
            raise OllamaError("Ollama could not find llama3.1. Run ollama pull llama3.1 on the host.")
        response.raise_for_status()
    except requests.ConnectTimeout as error:
        raise OllamaError("Could not connect to the Ollama gateway. Check that the gateway is running.") from error
    except requests.Timeout as error:
        raise OllamaError(timeout_message) from error
    except requests.ConnectionError as error:
        raise OllamaError("Could not connect to the Ollama gateway. Check that the gateway is running.") from error
    except requests.RequestException as error:
        raise OllamaError("Ollama rejected the request. Check the host Ollama logs.") from error
    content = response.json()["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty reply.")
    return content.strip()


def parse_tool_call(reply: str) -> dict | None:
    if not isinstance(reply, str):
        return None
    match = _TOOL_CALL.fullmatch(reply.strip())
    if match is None:
        return None
    try:
        args = json.loads(match[2])
    except (ValueError, RecursionError):
        return None
    if not isinstance(args, dict):
        return None
    return {"name": match[1], "args": args}
