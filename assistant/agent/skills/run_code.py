import os

import requests

from agent.results import failure, skill
from agent.skills.code_policy import validate


@skill
def run_code(code: str, timeout: int = 30) -> dict:
    validate(code, timeout)
    url = os.environ.get("RUNNER_URL", "http://runner:8081/run")
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(url, json={"code": code, "timeout": timeout}, timeout=30, allow_redirects=False)
            response.raise_for_status()
            result = response.json()
    except requests.Timeout:
        return failure("Code execution timed out.")
    except (requests.RequestException, ValueError):
        return failure("The isolated code runner is unavailable.")
    if not isinstance(result, dict) or set(result) != {"ok", "result", "error"}:
        return failure("The isolated code runner returned an invalid result.")
    return result
