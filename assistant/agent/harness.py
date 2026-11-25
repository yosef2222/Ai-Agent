"""Run a bounded tool loop and deliver results to the owner."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from threading import Lock

from agent.mcp import call_tool
from agent.planner import OllamaError, ask_llm, parse_tool_call
from agent.request_paths import OUTSIDE_WORKSPACE, is_file_command, mentions_outside_path
from agent.results import failure
from agent.skills.telegram_ops import send_telegram
from agent.tool_protocol import looks_like_tool_protocol

LOGGER = logging.getLogger(__name__)
WORK_LOCK = Lock()
MAX_ITERATIONS = 10
_INVALID_REPLY = "I couldn't complete the request because the model returned an invalid tool call."
_FILE_TOOLS = {"list_files", "read_file", "write_file", "github_commit"}
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


def _call(tool, autonomous):
    name, args = tool["name"], tool["args"]
    if autonomous and name in _SCHEDULED_RESTRICTED:
        return failure("The scheduler handles publishing and notifications for backlog tasks.")
    if autonomous and name in {"list_files", "read_file", "write_file"}:
        path = args.get("path")
        if not isinstance(path, str) or not (path.startswith("/workspace/project/") or (
            name == "list_files" and path == "/workspace/project"
        )):
            return failure("Backlog tasks can only access files inside /workspace/project.")
        relative = PurePosixPath(path).relative_to("/workspace/project")
        if name == "write_file" and relative.parts and relative.parts[0].casefold() in {"backlog.md", "reports"}:
            return failure("The scheduler manages the backlog and report files.")
    return call_tool(name, args)


def _loop(task, autonomous):
    errors = []
    written = set()
    try:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task is empty.")
        outside_path = mentions_outside_path(task)
        if outside_path and is_file_command(task):
            return Outcome(OUTSIDE_WORKSPACE, errors=[OUTSIDE_WORKSPACE])
        prompt = f"{_AUTONOMOUS}\n\nBacklog task:\n{task}" if autonomous else task
        messages = [{"role": "user", "content": prompt}]
        unresolved = set()
        last_failed_call = None
        search_error = None
        search_failures = 0
        search_sources = []
        invalid_replies = 0
        for _ in range(MAX_ITERATIONS):
            reply = ask_llm(messages, excluded_tools=_SCHEDULED_RESTRICTED if autonomous else ())
            tool = parse_tool_call(reply)
            if tool is None:
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

            result = _call(tool, autonomous)
            data = result.get("result")
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

            if result["ok"] and autonomous and name == "write_file":
                written.add(PurePosixPath(data["path"]).relative_to("/workspace/project").as_posix())
            if isinstance(data, dict) and data.get("confirmation_required"):
                return Outcome(f"Overwrite {data['path']}? To approve, send {data['command']} within 15 minutes.",
                               False, errors, sorted(written))
            if name == "run_code" and isinstance(data, dict) and data.get("prohibited"):
                return Outcome(result["error"], False, errors, sorted(written))

            serialized = json.dumps(result, ensure_ascii=True)
            if len(serialized) > 12000:
                serialized = json.dumps({"preview": serialized[:12000], "truncated": True})
            messages.extend([
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": tool["args"]}},
                ]},
                {"role": "tool", "tool_name": name, "content": serialized},
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
    if looks_like_tool_protocol(outcome.answer):
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
