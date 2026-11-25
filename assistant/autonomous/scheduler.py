"""Run the next approved backlog task when the assistant is available."""

import json
import logging
import os
import re
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler

from agent.harness import WORK_LOCK, run_autonomous
from agent.skills import file_ops, github_ops
from autonomous.reporter import write_report

LOGGER = logging.getLogger(__name__)
_OPEN_TASK = re.compile(r"^\s*(?:(?:[-*+]|\d+[.)])\s+)?\[ \]\s+(.+?)\s*$")


def _next_task(lines):
    for index, line in enumerate(lines):
        match = _OPEN_TASK.fullmatch(line)
        if match:
            return index, match[1]
    return None, None


def _timezone():
    return ZoneInfo(os.environ.get("ASSISTANT_TIMEZONE", "UTC"))


def _publish(journal, pending):
    record = pending["report"]
    subject = f"Assistant report {record['date']} ({record['id']})"
    head = github_ops.git("rev-parse", "HEAD")
    if head == pending["head"]:
        backlog = file_ops.checked_path("/workspace/project/backlog.md")
        current = backlog.read_bytes().decode("utf-8")
        if current not in {pending["before"], pending["after"]}:
            raise ValueError("The backlog changed before publication. Restore the pending version to retry.")
        report = write_report(record)
        published_path = f"reports/{report.name}"
        destination = file_ops.checked_path(f"/workspace/project/{published_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_ops.atomic_write(destination, report.read_bytes())
        if current != pending["after"]:
            file_ops.atomic_write(backlog, pending["after"].encode("utf-8"))
        result = github_ops.github_commit(subject, [*record["files"], "backlog.md", published_path])
    else:
        parent = github_ops.git("rev-parse", "HEAD^")
        actual_subject = github_ops.git("log", "-1", "--format=%s")
        if parent != pending["head"] or actual_subject != subject:
            raise ValueError("The checkout moved while a report was pending. Restore its branch to retry.")
        result = github_ops.retry_push()
    if not result["ok"]:
        raise ValueError(result["error"])
    journal.unlink()
    LOGGER.info("Published backlog completion and report %s.", record["date"])


def _work():
    state = file_ops.WORKSPACE / ".assistant"
    if state.is_symlink():
        raise ValueError("The assistant state directory cannot be a symbolic link.")
    journal = state / "pending-report.json"
    if journal.is_symlink():
        raise ValueError("The assistant state file cannot be a symbolic link.")
    if journal.is_file():
        pending = json.loads(journal.read_text(encoding="utf-8"))
        _publish(journal, pending)
        return

    backlog = file_ops.checked_path("/workspace/project/backlog.md")
    if not backlog.is_file():
        LOGGER.info("Autonomous mode is waiting for /workspace/project/backlog.md.")
        return
    before = backlog.read_bytes().decode("utf-8")
    lines = before.splitlines(keepends=True)
    index, task = _next_task(lines)
    if index is None:
        return
    github_ops.publish_target()
    head = github_ops.git("rev-parse", "HEAD")
    dirty = github_ops.changes().keys() - {"backlog.md"}
    if dirty:
        LOGGER.warning("Autonomous mode is waiting for a clean project checkout.")
        return

    now = datetime.now(_timezone())
    outcome = run_autonomous(task)
    work_changes = github_ops.changes()
    files = sorted(work_changes.keys() & set(outcome.files_changed))
    completed = outcome.succeeded
    if set(outcome.files_changed) - work_changes.keys():
        completed = False
        outcome.errors.append("A file created by the task is missing or ignored by Git.")
    if github_ops.git("rev-parse", "HEAD") != head or backlog.read_bytes().decode("utf-8") != before:
        completed = False
        outcome.errors.append("The project checkout changed during execution.")
    if completed:
        lines[index] = lines[index].replace("[ ]", "[x]", 1)
    _, next_task = _next_task(lines)
    record = {
        "id": uuid4().hex, "date": now.date().isoformat(), "time": now.strftime("%H:%M %Z"),
        "task": task, "completed": completed, "answer": outcome.answer, "next_task": next_task,
        "files": files, "errors": outcome.errors or ([] if completed else [outcome.answer]),
    }
    if not completed:
        write_report(record)
        return
    state.mkdir(mode=0o700, exist_ok=True)
    pending = {"head": head, "before": before, "after": "".join(lines), "report": record}
    file_ops.atomic_write(journal, json.dumps(pending).encode("utf-8"))
    _publish(journal, pending)


def autonomous_run():
    if not WORK_LOCK.acquire(blocking=False):
        return
    try:
        _work()
    except Exception as error:
        LOGGER.error("Autonomous run failed. %s", error)
    finally:
        WORK_LOCK.release()


def start_scheduler():
    scheduler = BackgroundScheduler(timezone=_timezone())
    scheduler.add_job(autonomous_run, "interval", hours=2, max_instances=1, coalesce=True)
    scheduler.start()
    return scheduler
