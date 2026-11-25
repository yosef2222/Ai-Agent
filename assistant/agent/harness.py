"""Run a bounded tool loop and deliver results to the owner."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from threading import Lock

from agent.mcp import call_tool
from agent.planner import OllamaError, ask_llm, parse_tool_call
from agent.results import failure
from agent.skills.telegram_ops import send_telegram

LOGGER = logging.getLogger(__name__)
WORK_LOCK = Lock()
MAX_ITERATIONS = 10
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
        prompt = f"{_AUTONOMOUS}\n\nBacklog task:\n{task}" if autonomous else task
        messages = [{"role": "user", "content": prompt}]
        unresolved = set()
        last_failed_call = None
        search_error = None
        for _ in range(MAX_ITERATIONS):
            reply = ask_llm(messages, excluded_tools=_SCHEDULED_RESTRICTED if autonomous else ())
            tool = parse_tool_call(reply)
            if tool is None:
                text = reply.lstrip()
                if text.startswith("TOOL:") or (text.startswith("```") and "TOOL:" in text):
                    messages.extend([
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": "Use the matching tool schema or provide a final answer."},
                    ])
                    continue
                marker, _, details = reply.partition("\n")
                if autonomous and marker.strip() == "TASK_COMPLETE":
                    answer = details.strip() or "Backlog task completed."
                    if unresolved:
                        return Outcome("A tool error remains unresolved. " + answer, False, errors, sorted(written))
                    return Outcome(answer, True, errors, sorted(written))
                return Outcome(reply, not autonomous, errors, sorted(written))

            name = tool["name"]
            if name == "web_search" and search_error:
                return Outcome(search_error, False, errors, sorted(written))
            if last_failed_call and tool == last_failed_call[0]:
                return Outcome(last_failed_call[1], False, errors, sorted(written))

            result = _call(tool, autonomous)
            data = result.get("result")
            if result["ok"]:
                unresolved.discard(name)
                last_failed_call = None
            else:
                unresolved.add(name)
                error = f"{name}: {result['error']}"
                last_failed_call = (tool, result["error"])
                if name == "web_search" and isinstance(data, dict) and data.get("unavailable"):
                    search_error = result["error"]
                errors.append(error)
                LOGGER.warning("%s", error)

            if result["ok"] and autonomous and name == "write_file":
                written.add(PurePosixPath(data["path"]).relative_to("/workspace/project").as_posix())
            if isinstance(data, dict) and data.get("confirmation_required"):
                return Outcome(f"Overwrite {data['path']}? To approve, send {data['command']} within 15 minutes.",
                               False, errors, sorted(written))

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
    delivery = send_telegram(outcome.answer)
    if not delivery["ok"]:
        LOGGER.warning("Final answer delivery failed. %s", delivery["error"])
        outcome.errors.append(delivery["error"])
    return outcome


def run(task: str) -> str:
    return _run(task).answer


def run_autonomous(task: str) -> Outcome:
    return _run(task, autonomous=True)
