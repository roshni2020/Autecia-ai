"""Provision an Arga Labs Slack twin (seeded caregiver workspace) and write its
URL + token into .env so EchoLoop's "Send to caregiver" can be tested safely.

    python -m eval.arga_twin            # needs ARGA_API_KEY in .env
    python -m tests.test_slack_twin     # then verify confirm -> send -> channel

Uses Arga's MCP endpoint (create_twin_run / get_twin_run). Twins on the
hackathon plan live 10 minutes; re-run to get a fresh one.
"""
import json
import os
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MCP = "https://api.argalabs.com/mcp"
SCENARIO = ("A small Slack workspace for a family/caregiver circle supporting one person who uses an "
            "AAC communication assistant called EchoLoop. Channels: #echoloop-messages (where EchoLoop "
            "posts the person's confirmed messages) and #general. Members: user_01 (the person), Priya "
            "(caregiver), Sam (support worker). A few earlier messages in #echoloop-messages such as "
            "'I need a break.' and 'It is too loud.'")


class Arga:
    def __init__(self, key: str):
        self.c = httpx.Client(follow_redirects=True, timeout=120, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"})
        r = self.c.post(MCP, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "echoloop", "version": "1"}}})
        self.url, self.n = str(r.url), 1
        if r.headers.get("mcp-session-id"):
            self.c.headers["Mcp-Session-Id"] = r.headers["mcp-session-id"]
        self.c.post(self.url, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, name: str, **args):
        self.n += 1
        r = self.c.post(self.url, json={"jsonrpc": "2.0", "id": self.n, "method": "tools/call",
                                        "params": {"name": name, "arguments": args}})
        txt = r.text
        if "data:" in txt:
            txt = "".join(ln[5:] for ln in txt.splitlines() if ln.startswith("data:"))
        d = json.loads(txt)
        if "error" in d:
            raise RuntimeError(d["error"])
        parts = [p.get("text", "") for p in d["result"].get("content", []) if p.get("type") == "text"]
        try:
            return json.loads(parts[0])
        except Exception:
            return parts


def set_env(values: dict) -> None:
    p = ROOT / ".env"
    s = p.read_text(encoding="utf-8") if p.exists() else ""
    for k, v in values.items():
        if re.search(rf"^{k}=.*$", s, re.M):
            s = re.sub(rf"^{k}=.*$", f"{k}={v}", s, flags=re.M)
        else:
            s += f"\n{k}={v}"
    p.write_text(s, encoding="utf-8")


def main() -> None:
    from backend.integrations import load_env
    load_env()
    key = os.getenv("ARGA_API_KEY")
    if not key:
        raise SystemExit("ARGA_API_KEY missing in .env")
    a = Arga(key)
    r = a.call("create_twin_run", twins="slack", ttl_minutes=int(os.getenv("ARGA_TTL_MIN", "10")),
               scenario_prompt=SCENARIO)
    if not isinstance(r, dict):
        raise SystemExit(f"create_twin_run: {r}")
    rid = r.get("run_id") or r.get("id")
    print("run", rid, "…")
    for _ in range(40):
        time.sleep(8)
        s = a.call("get_twin_run", run_id=rid)
        st = s.get("status") if isinstance(s, dict) else None
        print("  status:", st)
        if st in ("ready", "running", "completed", "succeeded"):
            tw = s["twins"]["slack"] if isinstance(s["twins"], dict) else s["twins"][0]
            set_env({"SLACK_BASE_URL": tw["base_url"], "SLACK_BOT_TOKEN": tw["env_vars"]["SLACK_BOT_TOKEN"],
                     "SLACK_CHANNEL": "echoloop-messages", "ARGA_RUN_ID": rid})
            print("twin ready:", tw["base_url"], "\nadmin:", tw.get("admin_url"),
                  "\ndashboard:", s.get("dashboard_url"), "\nexpires:", s.get("expires_at"))
            return
        if st in ("failed", "error"):
            raise SystemExit(f"twin failed: {s.get('error')}")
    raise SystemExit("timed out waiting for twin")


if __name__ == "__main__":
    main()
