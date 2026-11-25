"""Identify explicit file targets before a model can reinterpret them."""

import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

OUTSIDE_WORKSPACE = "I can only access files inside /workspace. The requested path is outside it."

_PATH = re.compile(
    r"""(?<![\w:/\\])(?:file://|[a-z]:[\\/]|\\\\|~[\w.-]*[\\/]|(?:\.\.[\\/])+|/)[^\s"'`“”‘’<>{}\[\](),;!?]*""",
    re.IGNORECASE,
)
_FILE_COMMAND = re.compile(
    r"^\s*(?:(?:please|(?:can|could|would|will)\s+you)\s+)*"
    r"(?:write|save|create|modify|edit|append|overwrite|update|replace|read|open|list|delete|remove|move|copy)\b",
    re.IGNORECASE,
)
_CONTENT_PREFIX = re.compile(
    r"\b(?:containing(?:\s+the\s+(?:string|text))?|with\s+(?:(?:the|exact)\s+)*(?:text|contents?))"
    r"\s*[:=]?\s*[\"'`“‘]?$", re.IGNORECASE,
)


def _in_workspace(path):
    if path.lower().startswith("file://"):
        try:
            url = urlsplit(path)
        except ValueError:
            return False
        if url.netloc.lower() not in {"", "localhost"}:
            return False
        path = unquote(url.path)
    return (path == "/workspace" or path.startswith("/workspace/")) and (
        "\\" not in path and ".." not in PurePosixPath(path).parts
    )


def mentions_outside_path(task):
    task = task.replace(r"\/", "/")
    workspace_target = None
    for match in _PATH.finditer(task):
        path = match[0]
        if path in {"/workspace.", "/workspace:"}:
            path = "/workspace"
        if _in_workspace(path):
            workspace_target = match.end()
        elif workspace_target is None or not _CONTENT_PREFIX.fullmatch(task[workspace_target:match.start()].strip()):
            return True
    return False


def is_file_command(task):
    return _FILE_COMMAND.match(task) is not None
