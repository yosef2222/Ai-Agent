"""Run a bounded tool loop and deliver results to the owner."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from threading import Lock

from agent.mcp import call_tool
from agent.planner import OllamaError, ask_llm, parse_tool_call
from agent.request_paths import OUTSIDE_WORKSPACE, is_file_command, mentions_outside_path
from agent.results import failure
from agent.skills.code_policy import PROHIBITED_REPLY, contains_prohibited_code
from agent.skills.telegram_ops import send_telegram
from agent.tool_protocol import looks_like_tool_protocol

LOGGER = logging.getLogger(__name__)
WORK_LOCK = Lock()
MAX_ITERATIONS = 10
MAX_TASK_LENGTH = 500
MAX_CONTENT_LENGTH = 1000
_CODE_CONTEXT = re.compile(
    r"```|\b(?:python|snippet|script|code)\b|\b(?:Path|read_file|write_file)\s*\("
    r"|\b(?:cat|head|tail|less|Get-Content)\s+", re.IGNORECASE,
)
_REPEAT_NUMBER = re.compile(r"[0-9]{1,6}")
_REPEATED_FILE_REQUEST = re.compile(
    r"^\s*(?:(?:please|(?:can|could|would)\s+you)\s+)*write(?:\s+exactly)?\s+"
    r"(?P<count>[0-9]{1,6}|[0-9]{1,3}(?:,[0-9]{3}){1,2})\s+"
    r"(?:(?P<case>upper|lower)case\s+)?(?P<char>[a-z])(?:['’]s|\s+characters?)?\s+(?:to|into|in)\s+"
    r"(?P<path>/workspace/[^\s\"'`]+?)"
    r"[.,]?\s+(?:and(?:\s+then)?|then)\s+read\s+(?:it|(?:that|the)\s+file)(?:\s+back)?"
    r"(?:(?:[.!]\s*|\s+and\s+)(?:report\s+the\s+(?:character\s+)?count|tell\s+me\s+how\s+many\s+characters\s+it\s+contains"
    r"(?:\s+and\s+summarize\s+the\s+content\s+briefly)?))?\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_INVALID_REPLY = "I couldn't complete the request because the model returned an invalid tool call."
_WRITE_TOOLS = {"write_file", "save_code_output"}
_WORKSPACE_TOOLS = {"list_files", "read_file", *_WRITE_TOOLS}
_FILE_TOOLS = {*_WORKSPACE_TOOLS, "github_commit"}
_SCHEDULED_RESTRICTED = {"github_commit", "send_telegram"}
_AUTONOMOUS = """Work on the owner-approved backlog task below. Project files are in /workspace/project.
The scheduler handles backlog changes, reports, and Git publishing. Leave those to it.
If you fully completed the task, start your final answer with TASK_COMPLETE on its own line.
If you could not complete it or need approval, explain what remains."""


@dataclass
class Outcome:
    answer: str
    succeeded: bool = False
    errors: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)


def _blocked_code(text):
    return contains_prohibited_code(text) or (
        isinstance(text, str) and _CODE_CONTEXT.search(text) is not None and mentions_outside_path(text)
    )


def _call(tool, autonomous, last_stdout=None):
    name, args = tool["name"], tool["args"]
    if autonomous and name in _SCHEDULED_RESTRICTED:
        return failure("The scheduler handles publishing and notifications for backlog tasks.")
    if autonomous and name in _WORKSPACE_TOOLS:
        path = args.get("path")
        if not isinstance(path, str) or not (path.startswith("/workspace/project/") or (
            name == "list_files" and path == "/workspace/project"
        )):
            return failure("Backlog tasks can only access files inside /workspace/project.")
        relative = PurePosixPath(path).relative_to("/workspace/project")
        if name in _WRITE_TOOLS and relative.parts and relative.parts[0].casefold() in {"backlog.md", "reports"}:
            return failure("The scheduler manages the backlog and report files.")
    if name == "run_code" and _blocked_code(args.get("code")):
        return failure(PROHIBITED_REPLY, {"prohibited": True})
    if name == "send_telegram" and _blocked_code(args.get("message")):
        return failure(PROHIBITED_REPLY, {"prohibited": True})
    if name == "save_code_output":
        if set(args) != {"path"}:
            return failure("Use save_code_output(path) after a successful run_code call.")
        return call_tool(name, {"path": args["path"], "stdout": last_stdout})
    if name == "write_file":
        args = args.copy()
        # Llama sometimes quotes the repeat count in native tool arguments.
        repeat = args.get("repeat")
        if isinstance(repeat, str) and _REPEAT_NUMBER.fullmatch(repeat):
            args["repeat"] = int(repeat)
        if isinstance(args.get("content"), str) and len(args["content"]) > MAX_CONTENT_LENGTH:
            return failure("Inline content is limited to 1,000 characters. Use repeat or generate the content with run_code.")
    return call_tool(name, args)


def _preview(text):
    return {"characters": len(text), "preview": text[:MAX_CONTENT_LENGTH], "truncated": True}


def _tool_output(name, result):
    data = result.get("result")
    if name == "read_file" and result["ok"] and isinstance(data, str) and len(data) > MAX_CONTENT_LENGTH:
        result = {**result, "result": _preview(data)}
    elif name == "run_code" and isinstance(data, dict):
        data = {key: _preview(value) if isinstance(value, str) and len(value) > MAX_CONTENT_LENGTH else value
                for key, value in data.items()}
        result = {**result, "result": data}
    serialized = json.dumps(result, ensure_ascii=True)
    if len(serialized) > 12000:
        serialized = json.dumps({"preview": serialized[:12000], "truncated": True})
    return serialized


def _overwrite_reply(data):
    return f"Overwrite {data['path']}? To approve, send {data['command']} within 15 minutes."


def _write_and_read_repeated_text(request):
    path = request["path"]
    count = int(request["count"].replace(",", ""))
    char = request["char"]
    if request["case"]:
        char = char.upper() if request["case"].lower() == "upper" else char.lower()
    write = call_tool("write_file", {"path": path, "content": char, "repeat": count})
    if not write["ok"]:
        data = write.get("result")
        reply = _overwrite_reply(data) if isinstance(data, dict) and data.get("confirmation_required") else write["error"]
        return Outcome(reply, errors=[write["error"]])
    read = call_tool("read_file", {"path": path})
    if not read["ok"]:
        return Outcome(f"I wrote {path}, but couldn't read it back. {read['error']}", errors=[read["error"]])
    text = read["result"]
    if len(text) != count or any(value != char for value in text):
        message = f"The readback from {path} did not match the requested content."
        return Outcome(message, errors=[message])
    return Outcome(f"I wrote and read back {count:,} copies of {char!r} in {path}.", succeeded=True)


def _loop(task, autonomous):
    errors = []
    written = set()
    try:
        if not isinstance(task, str) or not task.strip():
            return Outcome("Task is empty.", errors=["Task is empty."])
        if len(task) > MAX_TASK_LENGTH:
            message = f"Task too long. Max {MAX_TASK_LENGTH} characters."
            return Outcome(message, errors=[message])
        if _blocked_code(task):
            return Outcome(PROHIBITED_REPLY, errors=[PROHIBITED_REPLY])
        outside_path = mentions_outside_path(task)
        if outside_path and is_file_command(task):
            return Outcome(OUTSIDE_WORKSPACE, errors=[OUTSIDE_WORKSPACE])
        if not autonomous and not outside_path:
            repeat_request = _REPEATED_FILE_REQUEST.fullmatch(task)
            if repeat_request:
                return _write_and_read_repeated_text(repeat_request)
        prompt = f"{_AUTONOMOUS}\n\nBacklog task:\n{task}" if autonomous else task
        messages = [{"role": "user", "content": prompt}]
        unresolved = set()
        last_failed_call = None
        search_error = None
        search_failures = 0
        search_sources = []
        invalid_replies = 0
        last_stdout = None
        for _ in range(MAX_ITERATIONS):
            reply = ask_llm(messages, excluded_tools=_SCHEDULED_RESTRICTED if autonomous else ())
            tool = parse_tool_call(reply)
            if tool is None:
                if _blocked_code(reply):
                    return Outcome(PROHIBITED_REPLY, errors=[*errors, PROHIBITED_REPLY], files_changed=sorted(written))
                if looks_like_tool_protocol(reply):
                    invalid_replies += 1
                    if invalid_replies >= 2:
                        return Outcome(_INVALID_REPLY, errors=[*errors, _INVALID_REPLY], files_changed=sorted(written))
                    messages.append({"role": "user", "content": "Use a native tool call or answer in plain language. Do not print tool-call syntax."})
                    continue
                marker, _, details = reply.partition("\n")
                if autonomous and marker.strip() == "TASK_COMPLETE":
                    answer = details.strip() or "Backlog task completed."
                    if unresolved:
                        return Outcome("A tool error remains unresolved. " + answer, False, errors, sorted(written))
                    return Outcome(answer, True, errors, sorted(written))
                if not autonomous and search_sources and not any(url in reply for url in search_sources):
                    reply += f"\n\nSource\n{search_sources[0]}"
                return Outcome(reply, not autonomous, errors, sorted(written))

            name = tool["name"]
            if outside_path and name in _FILE_TOOLS:
                return Outcome(OUTSIDE_WORKSPACE, errors=[*errors, OUTSIDE_WORKSPACE], files_changed=sorted(written))
            if name == "web_search" and search_error:
                return Outcome(search_error, False, errors, sorted(written))
            if last_failed_call and tool == last_failed_call[0]:
                return Outcome(last_failed_call[1], False, errors, sorted(written))

            result = _call(tool, autonomous, last_stdout)
            data = result.get("result")
            if name == "run_code":
                stdout = data.get("stdout") if result["ok"] and isinstance(data, dict) else None
                last_stdout = stdout if isinstance(stdout, str) else None
            if result["ok"]:
                unresolved.discard(name)
                last_failed_call = None
                if name == "web_search":
                    search_failures = 0
                    if isinstance(data, list):
                        search_sources.extend(hit["url"] for hit in data if isinstance(hit, dict)
                                              and isinstance(hit.get("url"), str))
            else:
                unresolved.add(name)
                error = f"{name}: {result['error']}"
                last_failed_call = (tool, result["error"])
                if name == "web_search":
                    search_failures += 1
                    if search_failures >= 2 or (isinstance(data, dict) and data.get("unavailable")):
                        search_error = result["error"]
                errors.append(error)
                LOGGER.warning("%s", error)

            if result["ok"] and autonomous and name in _WRITE_TOOLS:
                written.add(PurePosixPath(data["path"]).relative_to("/workspace/project").as_posix())
            if isinstance(data, dict) and data.get("confirmation_required"):
                return Outcome(_overwrite_reply(data), False, errors, sorted(written))
            if isinstance(data, dict) and data.get("prohibited"):
                return Outcome(PROHIBITED_REPLY, False, errors, sorted(written))

            recorded_args = {key: value[:MAX_CONTENT_LENGTH] if isinstance(value, str) else value
                             for key, value in tool["args"].items()}
            messages.extend([
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": recorded_args}},
                ]},
                {"role": "tool", "tool_name": name, "content": _tool_output(name, result)},
            ])
        message = f"Stopped after {MAX_ITERATIONS} planning iterations."
    except OllamaError as error:
        message = str(error)
    except Exception as error:
        LOGGER.error("Agent loop failed (%s).", type(error).__name__)
        message = "I could not complete the task. Check the Ollama service and assistant logs."
    LOGGER.error("%s", message)
    return Outcome(message, False, [*errors, message], sorted(written))


def _run(task, autonomous=False):
    outcome = _loop(task, autonomous)
    if _blocked_code(outcome.answer):
        outcome = Outcome(PROHIBITED_REPLY, errors=[*outcome.errors, PROHIBITED_REPLY], files_changed=outcome.files_changed)
    elif looks_like_tool_protocol(outcome.answer):
        outcome = Outcome(_INVALID_REPLY, errors=[*outcome.errors, _INVALID_REPLY], files_changed=outcome.files_changed)
    delivery = send_telegram(outcome.answer)
    if not delivery["ok"]:
        LOGGER.warning("Final answer delivery failed. %s", delivery["error"])
        outcome.errors.append(delivery["error"])
    return outcome


def run(task: str) -> str:
    return _run(task).answer


def run_autonomous(task: str) -> Outcome:
    return _run(task, autonomous=True)
