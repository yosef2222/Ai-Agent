"""Write an idempotent entry in the daily workspace Markdown report."""

from datetime import date

from agent.skills.file_ops import atomic_write, checked_path


def _line(value):
    return " ".join(str(value).splitlines()) or "None"


def write_report(record):
    day = date.fromisoformat(record["date"]).isoformat()
    target = checked_path(f"/workspace/reports/{day}.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    current = target.read_text(encoding="utf-8") if target.is_file() else f"# Assistant report for {day}\n"
    marker = f"<!-- run {record['id']} -->"
    if marker in current:
        return target
    completed = _line(record["task"]) if record["completed"] else "None"
    lines = [
        "", marker, f"## {'Completed' if record['completed'] else 'Incomplete'} at {record['time']}", "",
        f"- Task completed: {completed}",
        f"- Attempted task: {_line(record['task'])}",
        f"- Summary: {_line(record['answer'])}",
        f"- Next task: {_line(record['next_task'] or 'None')}",
        "", "### Files changed", "",
    ]
    lines.extend(f"- {_line(path)}" for path in (record["files"] or ["None"]))
    lines.extend(["", "### Errors", ""])
    lines.extend(f"- {_line(error)}" for error in (record["errors"] or ["None"]))
    atomic_write(target, (current.rstrip("\n") + "\n" + "\n".join(lines) + "\n").encode("utf-8"))
    return target
