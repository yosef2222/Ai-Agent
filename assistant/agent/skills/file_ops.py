"""Workspace file access and one-time owner approval for overwrites."""

import hashlib
import os
import secrets
import tempfile
import time
from pathlib import Path, PurePosixPath

from agent.results import failure, skill, success

WORKSPACE = Path("/workspace")
MAX_BYTES = 256 * 1024
_pending = {}


def checked_path(path, *, allow_root=False):
    if allow_root and path in ("/workspace", "/workspace/"):
        return WORKSPACE
    if not isinstance(path, str) or not path.startswith("/workspace/") or "\\" in path or "\0" in path:
        raise ValueError("Use an absolute file path inside /workspace.")
    parts = PurePosixPath(path).parts[2:]
    if not parts or any(part.casefold() in {"..", ".git", ".assistant"} for part in parts):
        raise ValueError("The path references a protected location.")
    target = WORKSPACE
    for part in parts:
        target /= part
        if target.is_symlink():
            raise ValueError("Symbolic links are not allowed in workspace paths.")
    return target


def _read(target):
    if not target.is_file():
        raise ValueError("The path must reference a regular file.")
    with target.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("The file exceeds the 256 KiB tool limit.")
    return data


def atomic_write(target, data):
    """Replace one file by rename, keeping temporary data on the same mount."""
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=".assistant-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            os.chmod(name, target.stat().st_mode & 0o777)
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)


@skill
def read_file(path: str) -> dict:
    target = checked_path(path, allow_root=True)
    if target.is_dir():
        return failure("This path is a directory. Use list_files to see its contents.")
    if not target.exists():
        return failure("File not found. Use list_files on its parent directory to check the path.")
    return success(_read(target).decode("utf-8"))


@skill
def list_files(path: str) -> dict:
    directory = checked_path(path, allow_root=True)
    if not directory.is_dir():
        return failure("Directory not found. Start with /workspace or /workspace/project.")
    entries = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if entry.name.casefold() in {".git", ".assistant"} or entry.is_symlink():
            continue
        if entry.is_file() or entry.is_dir():
            entries.append({"path": f"/workspace/{entry.relative_to(WORKSPACE).as_posix()}",
                            "type": "directory" if entry.is_dir() else "file"})
        if len(entries) > 100:
            break
    return success({"entries": entries[:100], "truncated": len(entries) > 100})


@skill
def write_file(path: str, content: str, repeat: int = 1) -> dict:
    target = checked_path(path)
    if not isinstance(content, str):
        return failure("Content must be a string.")
    if type(repeat) is not int or not 1 <= repeat <= MAX_BYTES:
        return failure("Repeat must be a positive integer within the file size limit.")
    data = content.encode("utf-8")
    if len(data) * repeat > MAX_BYTES:
        return failure("Content exceeds the 256 KiB tool limit.")
    data *= repeat
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        digest = hashlib.sha256(_read(target)).digest()
        for approval, (expires, *_) in list(_pending.items()):
            if expires <= time.monotonic():
                del _pending[approval]
        if len(_pending) >= 32:
            return failure("Too many pending overwrite requests.")
        approval = secrets.token_hex(8)
        _pending[approval] = (time.monotonic() + 900, path, digest, data)
        return failure("File exists. The owner must approve the overwrite.", {
            "path": path, "confirmation_required": True, "command": f"/approve {approval}",
        })
    return success({"path": path, "bytes_written": len(data), "characters_written": len(content) * repeat})


@skill
def save_code_output(path: str, *, stdout: str | None = None) -> dict:
    if stdout is None:
        return failure("Run code successfully before saving its output.")
    return write_file(path, stdout)


@skill
def approve_overwrite(approval: str) -> dict:
    pending = _pending.pop(approval, None)
    if pending is None or pending[0] <= time.monotonic():
        return failure("Overwrite approval is unknown or expired.")
    _, path, digest, data = pending
    target = checked_path(path)
    if hashlib.sha256(_read(target)).digest() != digest:
        return failure("The file changed after the approval request. Request the overwrite again.")
    atomic_write(target, data)
    return success({"path": path, "bytes_written": len(data)})
