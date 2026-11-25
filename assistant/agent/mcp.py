"""Validate and dispatch the assistant's named tools."""

from inspect import signature

from agent.results import failure
from agent.skills.file_ops import read_file, write_file
from agent.skills.github_ops import github_commit
from agent.skills.run_code import run_code
from agent.skills.telegram_ops import send_telegram
from agent.skills.web_search import web_search

TOOLS = {tool.__name__: tool for tool in (web_search, read_file, write_file, run_code, github_commit, send_telegram)}


def call_tool(name, args):
    if not isinstance(name, str) or name not in TOOLS:
        return failure("Unknown tool.")
    if not isinstance(args, dict) or not all(isinstance(key, str) for key in args):
        return failure("Tool arguments must be a JSON object with string keys.")
    tool = TOOLS[name]
    try:
        signature(tool).bind(**args)
    except TypeError as error:
        return failure(f"Invalid arguments for {name}. {error}")
    return tool(**args)


def list_tools():
    return list(TOOLS)
