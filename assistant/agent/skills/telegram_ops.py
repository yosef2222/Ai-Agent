import os
import re
import time

import requests

from agent.results import failure, skill, success
from agent.tool_protocol import looks_like_tool_protocol

TEXT_LIMIT = 4000


class TelegramError(RuntimeError):
    def __init__(self, method, reason, retry_after=None):
        super().__init__(f"Telegram {method} failed. {reason}")
        self.retry_after = retry_after


def telegram_settings():
    token = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token) or not re.fullmatch(r"[1-9][0-9]*", chat_id):
        raise ValueError("Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID to your bot token and private chat ID.")
    return token, int(chat_id)


def telegram_request(token, method, payload, *, files=None):
    body_args = {"json": payload} if files is None else {"data": payload, "files": files}
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/{method}", timeout=(5, 23), allow_redirects=False, **body_args,
        )
    except (requests.ConnectTimeout, requests.exceptions.ProxyError):
        raise TelegramError(method, "Could not connect through the gateway.", retry_after=2) from None
    except requests.Timeout:
        raise TelegramError(method, "Telegram did not respond before the timeout.") from None
    except requests.RequestException:
        # Request exceptions can include the token in the URL.
        raise TelegramError(method, "The connection ended before Telegram confirmed the request.") from None

    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise TelegramError(method, f"Unexpected response (HTTP {response.status_code}).")
    if response.ok and body.get("ok") is True and "result" in body:
        return body["result"]

    status = body.get("error_code", response.status_code)
    if type(status) is not int:
        status = response.status_code
    if status == 429:
        parameters = body.get("parameters")
        delay = parameters.get("retry_after") if isinstance(parameters, dict) else None
        delay = delay if type(delay) is int and delay > 0 else 5
        raise TelegramError(method, f"Rate limited. Retry after {delay} seconds.", retry_after=delay)
    if status == 409 and method == "getUpdates":
        raise TelegramError(method, "Another poller or webhook is using this bot. Use only one polling instance.")
    if status in {401, 403}:
        raise TelegramError(method, f"Access denied (HTTP {status}). Check the bot token and owner chat permissions.")
    if method in {"sendMessage", "sendDocument"} and status == 400:
        description = str(body.get("description", "")).lower()
        if "chat not found" in description:
            raise TelegramError(method, "Owner chat not found. Check TELEGRAM_CHAT_ID and start the bot in Telegram.")
    raise TelegramError(method, f"Telegram returned HTTP {status}.")


def _send_to_owner(method, fields, **kwargs):
    token, chat_id = telegram_settings()
    payload = {"chat_id": chat_id, **fields}
    for attempt in range(2):
        try:
            response = telegram_request(token, method, payload, **kwargs)
            break
        except TelegramError as error:
            # Retry only failures for which Telegram did not accept the message.
            if attempt or not error.retry_after or error.retry_after > 5:
                return failure(str(error))
            time.sleep(error.retry_after)
    return success({"message_id": response["message_id"]})


@skill
def send_telegram(message: str) -> dict:
    if not isinstance(message, str) or not message.strip():
        return failure("Telegram message must be a nonempty string.")
    if looks_like_tool_protocol(message):
        return failure("Tool-call syntax cannot be sent to Telegram.")
    text = message.strip().encode("utf-16-le", errors="replace")[:TEXT_LIMIT * 2].decode("utf-16-le", errors="ignore")
    return _send_to_owner("sendMessage", {"text": text})


@skill
def send_telegram_file(filename: str, content: bytes, caption: str) -> dict:
    if not isinstance(filename, str) or not filename or any(char in filename for char in "/\\\r\n\0"):
        return failure("Telegram requires a valid attachment filename.")
    if not isinstance(content, bytes) or not content:
        return failure("Telegram requires a nonempty attachment.")
    if not isinstance(caption, str) or len(caption.encode("utf-16-le")) > 2048 or looks_like_tool_protocol(caption):
        return failure("The Telegram attachment caption is invalid.")
    return _send_to_owner("sendDocument", {"caption": caption}, files={"document": (filename, content, "text/plain")})
