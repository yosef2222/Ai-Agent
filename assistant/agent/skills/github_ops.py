"""Commit selected project paths and push the owner's configured repository."""

import base64
import os
import re
import subprocess
from pathlib import PurePosixPath

from agent.results import failure, skill, success
from agent.skills import file_ops


def project_root():
    return file_ops.checked_path("/workspace/project")


def git(*args, authorization=None):
    root = project_root()
    git_dir = root / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise ValueError("Clone the configured repository into /workspace/project first.")
    env = {key: value for key, value in os.environ.items() if key in {
        "PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "HOME", "USERPROFILE", "HTTPS_PROXY", "NO_PROXY",
    }}
    config = {
        "safe.directory": root.as_posix(),
        "core.hooksPath": os.devnull,
        "core.fsmonitor": "false",
        "core.attributesFile": os.devnull,
        "user.name": "Personal AI Assistant",
        "user.email": "assistant@users.noreply.github.com",
        "commit.gpgsign": "false",
        "push.gpgsign": "false",
        "push.followTags": "false",
        "http.followRedirects": "false",
        "credential.helper": "",
    }
    if authorization:
        config["http.https://github.com/.extraHeader"] = f"Authorization: Basic {authorization}"
    env.update({
        "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ALLOW_PROTOCOL": "https", "GIT_CONFIG_COUNT": str(len(config)), "LC_ALL": "C.UTF-8",
    })
    for index, (key, value) in enumerate(config.items()):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    try:
        process = subprocess.run(
            ["git", "--literal-pathspecs", *args], cwd=root, env=env, timeout=30, capture_output=True, check=False,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(f"Git {args[0]} timed out.") from None
    if process.returncode:
        raise ValueError(f"Git {args[0]} failed. Check the project checkout, branch and GitHub permissions.")
    return process.stdout.decode("utf-8", "surrogateescape").rstrip("\n")


def changes(*paths):
    records = git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames", "--", *paths)
    return {entry[3:]: entry[:2] for entry in records.split("\0") if entry}


def publish_target():
    repo = os.environ.get("GITHUB_REPO", "").strip().removeprefix("https://github.com/").removesuffix(".git")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", repo) or repo.split("/")[1] in {".", ".."}:
        raise ValueError("GITHUB_REPO must identify one GitHub repository as owner/name.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise ValueError("Set GITHUB_TOKEN before publishing.")
    remote = f"https://github.com/{repo}"
    origin = git("remote", "get-url", "--all", "origin").splitlines()
    if len(origin) != 1 or origin[0].removesuffix(".git").casefold() != remote.casefold():
        raise ValueError("The project's HTTPS origin must match GITHUB_REPO.")
    if git("symbolic-ref", "--short", "HEAD") != "main":
        raise ValueError("Check out the main branch before publishing.")
    authorization = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return remote, authorization


def _paths(files):
    paths = []
    for file in files:
        if not isinstance(file, str) or not file.strip() or "\n" in file or "\r" in file:
            raise ValueError("Each commit path must be a file in /workspace/project.")
        path = PurePosixPath(file.removeprefix("/workspace/project/"))
        if path.is_absolute() or not path.parts:
            raise ValueError("Each commit path must be a file in /workspace/project.")
        target = file_ops.checked_path(f"/workspace/project/{path}")
        if target.exists() and not target.is_file():
            raise ValueError("Commit paths must name individual regular files.")
        paths.append(path.as_posix())
    return list(dict.fromkeys(paths))


@skill
def github_commit(message: str, files: list) -> dict:
    if not isinstance(message, str) or not 1 <= len(message.strip()) <= 500 or "\0" in message:
        return failure("Provide a commit message between 1 and 500 characters.")
    if not isinstance(files, list) or not 1 <= len(files) <= 100:
        return failure("Provide a list of 1 to 100 project files.")
    paths = _paths(files)
    remote, authorization = publish_target()
    status = changes(*paths)
    for path in paths:
        if path not in status:
            git("ls-files", "--error-unmatch", "--", path)
    untracked = [path for path, code in status.items() if code == "??"]
    if untracked:
        git("add", "--intent-to-add", "--", *untracked)
    # A staged-only edit may equal HEAD on disk, so test the working tree first.
    if status and git("diff", "--name-only", "HEAD", "--", *paths):
        git("commit", "--only", "-m", message.strip(), "--", *paths)
    return _push(remote, authorization)


def _push(remote, authorization):
    commit = git("rev-parse", "HEAD")
    try:
        git("push", "--porcelain", remote, "HEAD:refs/heads/main", authorization=authorization)
    except ValueError as error:
        return failure(str(error), {"commit": commit, "pushed": False})
    return success({"commit": commit, "pushed": True})


@skill
def retry_push() -> dict:
    return _push(*publish_target())
