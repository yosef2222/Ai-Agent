"""Common result format for the tool registry and its skills."""

import logging
from functools import wraps


def success(result):
    return {"ok": True, "result": result, "error": None}


def failure(error, result=None):
    return {"ok": False, "result": result, "error": error}


def skill(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError) as error:
            return failure(str(error))
        except Exception as error:
            logging.getLogger(__name__).warning("%s failed (%s)", function.__name__, type(error).__name__)
            return failure(f"{function.__name__} failed ({type(error).__name__}).")

    return checked
