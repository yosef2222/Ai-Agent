"""Validate and dispatch the assistant's named tools."""

from inspect import Parameter, signature

from agent.results import failure
from agent.skills.file_ops import list_files, read_file, write_file
from agent.skills.github_ops import github_commit
from agent.skills.run_code import run_code
from agent.skills.telegram_ops import send_telegram
from agent.skills.web_search import web_search

TOOLS = {tool.__name__: tool for tool in (web_search, list_files, read_file, write_file, run_code, github_commit, send_telegram)}

_STRING = {"type": "string"}
_SCHEMAS = {
    "web_search": ("Search the web and get source links. Include the full subject and context in the query.", {"query": _STRING}),
    "list_files": ("List immediate entries in a workspace directory. Start at /workspace if unsure.", {"path": _STRING}),
    "read_file": ("Read an existing UTF-8 file by its absolute /workspace path.", {"path": _STRING}),
    "write_file": ("Create a UTF-8 file. Provide the full absolute /workspace path and the complete file content.",
                   {"path": _STRING, "content": _STRING}),
    "run_code": ("Run isolated Python for calculations or small examples. The runner has no project files or network.",
                 {"code": _STRING, "timeout": {"type": "integer", "minimum": 1, "maximum": 30}}),
    "github_commit": ("Commit named files from /workspace/project and push to the configured GitHub main branch.",
                      {"message": _STRING, "files": {"type": "array", "items": _STRING, "minItems": 1, "maxItems": 100}}),
    "send_telegram": ("Send a separate Telegram message to the owner when explicitly requested.", {"message": _STRING}),
}


def tool_specs(excluded=()):
    return [
        {"type": "function", "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object", "properties": properties, "additionalProperties": False,
                "required": [parameter.name for parameter in signature(TOOLS[name]).parameters.values()
                             if parameter.default is Parameter.empty],
            },
        }}
        for name, (description, properties) in _SCHEMAS.items() if name not in excluded
    ]


def call_tool(name, args):
    if not isinstance(name, str) or name not in TOOLS:
        return failure("Unknown tool.")
    if not isinstance(args, dict) or not all(isinstance(key, str) for key in args):
        return failure("Tool arguments must be a JSON object with string keys.")
    tool = TOOLS[name]
    try:
        signature(tool).bind(**args)
    except TypeError as error:
        parameters = ", ".join(signature(tool).parameters)
        return failure(f"Invalid arguments for {name}. {error}. Use {name}({parameters}).")
    return tool(**args)


def list_tools():
    return list(TOOLS)
