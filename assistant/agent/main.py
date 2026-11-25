"""Poll Telegram for the owner's messages and start the backlog scheduler."""

import logging
import re
import signal
from threading import Event

from agent.harness import WORK_LOCK, run
from agent.skills.file_ops import approve_overwrite
from agent.skills.telegram_ops import send_telegram, telegram_request, telegram_settings
from autonomous.scheduler import start_scheduler

LOGGER = logging.getLogger(__name__)


def handle_update(update, owner_id):
    message = update.get("message", {})
    chat = message.get("chat", {})
    sender = message.get("from", {})
    if chat.get("id") != owner_id or chat.get("type") != "private" or sender.get("id") != owner_id:
        return
    task = message.get("text")
    if not isinstance(task, str) or not task.strip():
        return
    task = task.strip()
    with WORK_LOCK:
        if task.split(maxsplit=1)[0] == "/approve":
            match = re.fullmatch(r"/approve\s+([0-9a-f]{16})", task)
            if match is None:
                send_telegram("Send the /approve command shown in the overwrite request.")
                return
            result = approve_overwrite(match[1])
            reply = f"Updated {result['result']['path']}." if result["ok"] else result["error"]
            send_telegram(reply)
        else:
            run(task)


def poll(token, owner_id, stop):
    offset = 0
    while not stop.is_set():
        try:
            updates = telegram_request(token, "getUpdates", {
                "offset": offset, "timeout": 20, "allowed_updates": ["message"],
            })
            if not isinstance(updates, list):
                raise RuntimeError("Telegram returned an invalid update list.")
        except RuntimeError as error:
            LOGGER.warning("%s", error)
            stop.wait(5)
            continue
        for update in updates:
            if stop.is_set():
                return
            if not isinstance(update, dict) or type(update.get("update_id")) is not int:
                continue
            offset = max(offset, update["update_id"] + 1)
            try:
                handle_update(update, owner_id)
            except Exception as error:
                LOGGER.error("Telegram update failed (%s).", type(error).__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("primp").setLevel(logging.WARNING)
    try:
        token, owner_id = telegram_settings()
        scheduler = start_scheduler()
    except Exception as error:
        LOGGER.error("Startup failed. %s", error)
        return 1
    stop = Event()

    def stop_requested(*_):
        stop.set()

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    LOGGER.info("Assistant started. Polling Telegram and scheduling backlog work every two hours.")
    try:
        poll(token, owner_id, stop)
    finally:
        scheduler.shutdown(wait=True)
        LOGGER.info("Assistant stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
