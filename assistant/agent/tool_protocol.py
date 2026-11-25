"""Recognize tool-call syntax at the boundary to user-visible replies."""

import re

_TEXT_MARKER = re.compile(r"(?:^|[\s`])(?:TOOL|ARGS)\s*:", re.IGNORECASE)
_JSON_WRAPPER = re.compile(r"""["'](?:tool_calls?|function_call)["']\s*:""", re.IGNORECASE)
_JSON_NAME = re.compile(r"""["'](?:name|tool|tool_name)["']\s*:\s*["'][\w.-]+["']""", re.IGNORECASE)
_JSON_ARGS = re.compile(r"""["'](?:parameters|arguments|args)["']\s*:""", re.IGNORECASE)


def looks_like_tool_protocol(text):
    return isinstance(text, str) and bool(
        _TEXT_MARKER.search(text) or _JSON_WRAPPER.search(text) or
        (_JSON_NAME.search(text) and _JSON_ARGS.search(text))
    )
