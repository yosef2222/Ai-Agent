"""Validate the snippet payload before the subprocess is started."""

import ast
import re

_FORBIDDEN = re.compile(
    r"\bimport\s+(?:os|sys|subprocess)(?:\b|\.)|\b__import__\b|\b(?:eval|exec)\s*\(|\bopen\s*\(\s*['\"]?/"
)
_MODULES = {"os", "sys", "subprocess"}


class ProhibitedCodeError(ValueError):
    pass


def validate(code, timeout):
    if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > 64 * 1024:
        raise ValueError("Provide a nonempty Python snippet of up to 64 KiB.")
    if type(timeout) is not int or not 1 <= timeout <= 30:
        raise ValueError("Timeout must be an integer between 1 and 30 seconds.")
    if _FORBIDDEN.search(code):
        raise ProhibitedCodeError("The snippet contains a prohibited operation.")
    try:
        syntax = ast.parse(code)
    except SyntaxError as error:
        raise ValueError(f"Invalid Python syntax on line {error.lineno}.") from None
    for node in ast.walk(syntax):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        if any(module.split(".", 1)[0] in _MODULES for module in modules):
            raise ProhibitedCodeError("The snippet imports a prohibited module.")
