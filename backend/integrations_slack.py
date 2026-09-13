"""Send a CONFIRMED message to the person's caregiver circle on Slack.

Real Slack Web API shape (chat.postMessage / conversations.*). SLACK_BASE_URL
points at an Arga Labs Slack twin during testing and at https://slack.com in
production — same code either way. Only text the user explicitly confirmed and
explicitly asked to send ever reaches this function.
"""
import os

import httpx


def configured() -> bool:
    return bool(os.getenv("SLACK_BOT_TOKEN"))


def _base() -> str:
    return os.getenv("SLACK_BASE_URL", "https://slack.com").rstrip("/")


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"}


def channel_id(name: str | None = None) -> str | None:
    name = (name or os.getenv("SLACK_CHANNEL", "echoloop-messages")).lstrip("#")
    r = httpx.get(f"{_base()}/api/conversations.list", headers=_headers(), timeout=20)
    r.raise_for_status()
    for c in r.json().get("channels", []):
        if c.get("name") == name:
            return c["id"]
    return None


def send(text: str, channel: str | None = None) -> dict:
    """Post `text` to the caregiver channel. Returns {ok, channel, ts, permalink?}."""
    if not configured():
        return {"ok": False, "error": "slack_not_configured"}
    cid = channel_id(channel)
    if cid is None:
        return {"ok": False, "error": "channel_not_found"}
    r = httpx.post(f"{_base()}/api/chat.postMessage", headers=_headers(), timeout=20,
                   json={"channel": cid, "text": text})
    r.raise_for_status()
    body = r.json()
    return {"ok": bool(body.get("ok")), "channel": body.get("channel"), "ts": body.get("ts"),
            "error": body.get("error"), "base_url": _base()}


def recent(limit: int = 5, channel: str | None = None) -> list[dict]:
    if not configured():
        return []
    cid = channel_id(channel)
    if cid is None:
        return []
    r = httpx.get(f"{_base()}/api/conversations.history", headers=_headers(), timeout=20,
                  params={"channel": cid, "limit": limit})
    r.raise_for_status()
    return [{"text": m.get("text", ""), "user": m.get("user", ""), "ts": m.get("ts", "")}
            for m in r.json().get("messages", [])]
