from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, channel TEXT NOT NULL, created_at REAL NOT NULL,
    updated_at REAL NOT NULL, title TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5, created_at REAL NOT NULL, accessed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_scope_idx ON facts(scope, accessed_at DESC);
CREATE TABLE IF NOT EXISTS tool_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, tool TEXT NOT NULL,
    arguments TEXT NOT NULL, outcome TEXT NOT NULL, remote INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
    when_to_use TEXT NOT NULL, steps TEXT NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0, successes INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, accessed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS skills_accessed_idx ON skills(accessed_at DESC);
"""


class MemoryStore:
    def __init__(self, path: Path, message_limit: int = 10_000, fact_limit: int = 2_000, skill_limit: int = 200):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.message_limit = message_limit
        self.fact_limit = fact_limit
        self.skill_limit = skill_limit
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def new_session(self, channel: str = "terminal", title: str = "") -> str:
        session_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO sessions(id, channel, created_at, updated_at, title) VALUES(?,?,?,?,?)",
                (session_id, channel, now, now, title[:200]),
            )
        return session_id

    def ensure_session(self, session_id: str, channel: str = "terminal") -> None:
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO sessions(id, channel, created_at, updated_at) VALUES(?,?,?,?)",
                (session_id, channel, now, now),
            )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.ensure_session(session_id)
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
                (session_id, role, content, now),
            )
            self._db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
        self.prune()

    def list_sessions(self, channel: str = "terminal", limit: int = 50) -> list[dict[str, Any]]:
        """Recent sessions for browsing past chats, newest first, with a message count and
        the first user line as a title so a session is recognisable."""
        with self._lock:
            rows = self._db.execute(
                "SELECT s.id, s.title, s.updated_at, "
                "(SELECT count(*) FROM messages m WHERE m.session_id = s.id) AS messages "
                "FROM sessions s WHERE s.channel=? AND "
                "(SELECT count(*) FROM messages m WHERE m.session_id = s.id) > 0 "
                "ORDER BY s.updated_at DESC LIMIT ?",
                (channel, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_messages(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Find past conversation lines across all sessions, so Kilo can recall something
        said in an earlier chat, not just the current one."""
        terms = [term for term in query.lower().split() if len(term) > 2][:6]
        if not terms:
            return []
        clauses = " OR ".join("lower(content) LIKE ?" for _ in terms)
        params: list[Any] = [*[f"%{term}%" for term in terms], limit]
        with self._lock:
            rows = self._db.execute(
                f"SELECT session_id, role, content, created_at FROM messages "
                f"WHERE ({clauses}) ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def history(self, session_id: str, limit: int = 40) -> list[dict[str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def remember(self, content: str, scope: str = "global", importance: float = 0.5) -> int:
        now = time.time()
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO facts(scope, content, importance, created_at, accessed_at) VALUES(?,?,?,?,?)",
                (scope, content[:4000], max(0.0, min(1.0, importance)), now, now),
            )
        self.prune()
        return int(cursor.lastrowid)

    def recall(self, query: str, scope: str = "global", limit: int = 8) -> list[str]:
        terms = [term for term in query.lower().split() if len(term) > 2][:6]
        if not terms:
            return []
        clauses = " OR ".join("lower(content) LIKE ?" for _ in terms)
        params: list[Any] = [scope, *[f"%{term}%" for term in terms], limit]
        with self._lock, self._db:
            rows = self._db.execute(
                f"SELECT id, content FROM facts WHERE scope=? AND ({clauses}) "
                "ORDER BY importance DESC, accessed_at DESC LIMIT ?",
                params,
            ).fetchall()
            if rows:
                self._db.executemany("UPDATE facts SET accessed_at=? WHERE id=?", [(time.time(), r["id"]) for r in rows])
        return [str(row["content"]) for row in rows]

    def save_skill(self, name: str, when_to_use: str, steps: str) -> int:
        """Record a reusable procedure. Re-saving a name refines it in place, so a skill
        improves rather than accumulating near-duplicates."""
        now = time.time()
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO skills(name, when_to_use, steps, created_at, accessed_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET when_to_use=excluded.when_to_use, "
                "steps=excluded.steps, accessed_at=excluded.accessed_at",
                (name[:120], when_to_use[:600], steps[:4000], now, now),
            )
        self.prune()
        return int(cursor.lastrowid)

    def recall_skills(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Find skills whose name or trigger matches the request.

        Matching is deliberately narrow: an irrelevant skill is not free, it is prompt
        tokens the model has to read on hardware where that costs real time.
        """
        terms = [term for term in query.lower().split() if len(term) > 3][:6]
        if not terms:
            return []
        clauses = " OR ".join("(lower(name) LIKE ? OR lower(when_to_use) LIKE ?)" for _ in terms)
        params: list[Any] = []
        for term in terms:
            params.extend([f"%{term}%", f"%{term}%"])
        params.append(limit)
        with self._lock, self._db:
            rows = self._db.execute(
                f"SELECT id, name, when_to_use, steps, uses, successes FROM skills WHERE {clauses} "
                "ORDER BY (successes + 1.0) / (uses + 1.0) DESC, accessed_at DESC LIMIT ?",
                params,
            ).fetchall()
            if rows:
                self._db.executemany("UPDATE skills SET accessed_at=? WHERE id=?", [(time.time(), r["id"]) for r in rows])
        return [dict(row) for row in rows]

    def list_skills(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT name, when_to_use, uses, successes FROM skills ORDER BY accessed_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def record_skill_outcome(self, name: str, succeeded: bool) -> None:
        """Track whether a skill actually worked, so reliable ones are preferred."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE skills SET uses = uses + 1, successes = successes + ? WHERE name = ?",
                (1 if succeeded else 0, name),
            )

    def audit(self, session_id: str, tool: str, arguments: dict[str, Any], outcome: str, remote: bool) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO tool_audit(session_id, tool, arguments, outcome, remote, created_at) VALUES(?,?,?,?,?,?)",
                (session_id, tool, json.dumps(arguments, sort_keys=True)[:16000], outcome[:4000], int(remote), time.time()),
            )

    def prune(self) -> None:
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM messages WHERE id IN (SELECT id FROM messages ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (self.message_limit,),
            )
            self._db.execute(
                "DELETE FROM facts WHERE id IN (SELECT id FROM facts ORDER BY importance DESC, accessed_at DESC LIMIT -1 OFFSET ?)",
                (self.fact_limit,),
            )
            # Least reliable and least recently used skills go first, so the registry
            # keeps what actually works rather than whatever was written last.
            self._db.execute(
                "DELETE FROM skills WHERE id IN (SELECT id FROM skills "
                "ORDER BY (successes + 1.0) / (uses + 1.0) DESC, accessed_at DESC LIMIT -1 OFFSET ?)",
                (self.skill_limit,),
            )

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                name: int(self._db.execute(f"SELECT count(*) FROM {name}").fetchone()[0])
                for name in ("sessions", "messages", "facts", "tool_audit", "skills")
            }

