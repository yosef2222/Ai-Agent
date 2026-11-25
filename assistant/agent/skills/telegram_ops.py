import os
import re

import requests

from agent.results import failure, skill, success


def telegram_settings():
    token = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token) or not re.fullmatch(r"[1-9][0-9]*", chat_id):
        raise ValueError("Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID to your bot token and private chat ID.")
    return token, int(chat_id)


def telegram_request(token, method, payload):
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/{method}", json=payload, timeout=30, allow_redirects=False,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("ok") is True:
            return body["result"]
    except (requests.RequestException, ValueError, TypeError, AttributeError, KeyError):
        pass
    # HTTP exceptions can contain the bot token in the request URL.
    raise RuntimeError(f"Telegram {method} failed.") from None


@skill
def send_telegram(message: str) -> dict:
    if not isinstance(message, str) or not message.strip():
        return failure("Telegram message must be a nonempty string.")
    try:
        token, chat_id = telegram_settings()
        response = telegram_request(token, "sendMessage", {"chat_id": chat_id, "text": message[:4000]})
    except RuntimeError as error:
        return failure(str(error))
    return success({"message_id": response["message_id"]})
