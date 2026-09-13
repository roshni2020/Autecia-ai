"""SQLite store: profiles, confirmed memories (vectors), interactions, policy.

ponytail: numpy cosine over one user's confirmed rows instead of a vector DB.
Users have tens-to-hundreds of memories; a service is not worth it.
Move to Chroma/pgvector when a single user exceeds ~10k memories.
"""
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from .embed import DIM, cosine, embed
from .schemas import MemoryHit, SupportProfile

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "echoloop.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
  user_id TEXT PRIMARY KEY, profile TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT, fragment TEXT, visual_summary TEXT, confirmed_text TEXT,
  reward INTEGER, embedding BLOB, success_count INT DEFAULT 0,
  failure_count INT DEFAULT 0, timestamp REAL);
CREATE TABLE IF NOT EXISTS interactions (
  interaction_id TEXT PRIMARY KEY, user_id TEXT, session_id TEXT,
  observation TEXT, feedback TEXT, reflection TEXT,
  policy_version INT, created REAL);
CREATE TABLE IF NOT EXISTS policy (
  user_id TEXT PRIMARY KEY, weights TEXT, version INT, updates INT);
CREATE INDEX IF NOT EXISTS mem_user ON memories(user_id);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


# ---- profile ---------------------------------------------------------------

def get_profile(con, user_id: str) -> SupportProfile:
    row = con.execute("SELECT profile FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    return SupportProfile(**json.loads(row["profile"])) if row else SupportProfile()


def set_profile(con, user_id: str, profile: SupportProfile) -> None:
    con.execute("INSERT INTO profiles(user_id,profile) VALUES(?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET profile=excluded.profile",
                (user_id, profile.model_dump_json()))
    con.commit()


# ---- confirmed memories ----------------------------------------------------

def add_memory(con, user_id: str, fragment: str, visual_summary: str,
               confirmed_text: str, reward: int) -> None:
    """Only confirmed communications are stored (spec §13, §25)."""
    key = f"{fragment} {visual_summary}"
    row = con.execute(
        "SELECT id,success_count,failure_count FROM memories "
        "WHERE user_id=? AND confirmed_text=?", (user_id, confirmed_text)).fetchone()
    if row:
        col = "success_count" if reward >= 0 else "failure_count"
        con.execute(f"UPDATE memories SET {col}={col}+1, timestamp=?, fragment=? "
                    "WHERE id=?", (time.time(), fragment, row["id"]))
    else:
        con.execute(
            "INSERT INTO memories(user_id,fragment,visual_summary,confirmed_text,reward,"
            "embedding,success_count,failure_count,timestamp) VALUES(?,?,?,?,?,?,?,?,?)",
            (user_id, fragment, visual_summary, confirmed_text, reward,
             embed(key).tobytes(), 1 if reward >= 0 else 0, 0 if reward >= 0 else 1,
             time.time()))
    con.commit()


def search(con, user_id: str, query: str, k: int = 3) -> list[MemoryHit]:
    rows = con.execute("SELECT * FROM memories WHERE user_id=?", (user_id,)).fetchall()
    if not rows:
        return []
    q = embed(query)
    scored = []
    for r in rows:
        e = np.frombuffer(r["embedding"], dtype=np.float32)
        if e.shape[0] != DIM:
            continue
        scored.append(MemoryHit(fragment=r["fragment"], confirmed_text=r["confirmed_text"],
                                similarity=cosine(q, e), success_count=r["success_count"],
                                failure_count=r["failure_count"]))
    scored.sort(key=lambda m: -m.similarity)
    return scored[:k]


def forget(con, user_id: str, confirmed_text: str) -> int:
    cur = con.execute("DELETE FROM memories WHERE user_id=? AND confirmed_text=?",
                      (user_id, confirmed_text))
    con.commit()
    return cur.rowcount


# ---- interaction records (observation kept separate from feedback, §14) -----

def save_observation(con, interaction_id, user_id, session_id, observation, version):
    con.execute("INSERT OR REPLACE INTO interactions(interaction_id,user_id,session_id,"
                "observation,feedback,reflection,policy_version,created) "
                "VALUES(?,?,?,?,NULL,NULL,?,?)",
                (interaction_id, user_id, session_id, json.dumps(observation),
                 version, time.time()))
    con.commit()


def get_interaction(con, interaction_id: str):
    return con.execute("SELECT * FROM interactions WHERE interaction_id=?",
                       (interaction_id,)).fetchone()


def save_feedback(con, interaction_id: str, feedback: dict, reflection: dict) -> None:
    con.execute("UPDATE interactions SET feedback=?, reflection=? WHERE interaction_id=?",
                (json.dumps(feedback), json.dumps(reflection), interaction_id))
    con.commit()


def history(con, user_id: str, limit: int = 50) -> list[dict]:
    rows = con.execute("SELECT * FROM interactions WHERE user_id=? ORDER BY created DESC "
                       "LIMIT ?", (user_id, limit)).fetchall()
    out = []
    for r in rows:
        out.append({"interaction_id": r["interaction_id"],
                    "observation": json.loads(r["observation"]),
                    "feedback": json.loads(r["feedback"]) if r["feedback"] else None,
                    "reflection": json.loads(r["reflection"]) if r["reflection"] else None,
                    "policy_version": r["policy_version"], "created": r["created"]})
    return out
