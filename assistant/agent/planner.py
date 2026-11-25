"""Talk to Ollama and decode the assistant's text tool protocol."""

import json
import re

import requests

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


def ask_llm(messages: list) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3.1", "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        },
        timeout=30,
        allow_redirects=False,
    )
    response.raise_for_status()
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
