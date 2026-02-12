from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EpisodeStore:
    def __init__(self, db_path: Path, long_term_path: Path) -> None:
        self.db_path = db_path
        self.long_term_path = long_term_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.long_term_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    draft TEXT NOT NULL,
                    verifier_pass INTEGER NOT NULL,
                    risk_score REAL NOT NULL,
                    trust_score REAL NOT NULL,
                    violations TEXT NOT NULL,
                    promoted INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def record_episode(
        self,
        *,
        session_id: str,
        user_input: str,
        draft: str,
        verifier_pass: bool,
        risk_score: float,
        trust_score: float,
        violations: list[str],
        source: str,
    ) -> int:
        created_at = datetime.now(UTC).isoformat()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO episodes (
                    session_id, user_input, draft, verifier_pass, risk_score,
                    trust_score, violations, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    session_id,
                    user_input,
                    draft,
                    1 if verifier_pass else 0,
                    risk_score,
                    trust_score,
                    json.dumps(violations),
                    source,
                    created_at,
                ),
            )
            return int(cursor.lastrowid)

    def pending_episodes(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_id, user_input, draft, verifier_pass,
                       risk_score, trust_score, violations, source, created_at
                FROM episodes
                WHERE promoted = 0
                ORDER BY id ASC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def pass_rate(self, window: int = 200) -> float:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT verifier_pass
                FROM episodes
                ORDER BY id DESC
                LIMIT ?;
                """,
                (window,),
            ).fetchall()
        if not rows:
            return 1.0
        passed = sum(int(row["verifier_pass"]) for row in rows)
        return passed / len(rows)

    def regression_fail_count(self, window: int = 200) -> int:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM (
                    SELECT verifier_pass, risk_score
                    FROM episodes
                    ORDER BY id DESC
                    LIMIT ?
                )
                WHERE verifier_pass = 0 OR risk_score > 0.85;
                """,
                (window,),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def promote_episode(self, episode_id: int) -> None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, session_id, user_input, draft, verifier_pass,
                       risk_score, trust_score, violations, source, created_at
                FROM episodes
                WHERE id = ?;
                """,
                (episode_id,),
            ).fetchone()
            if row is None:
                return
            with self._conn:
                self._conn.execute(
                    "UPDATE episodes SET promoted = 1 WHERE id = ?;", (episode_id,)
                )
            payload = dict(row)

        with self.long_term_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) AS promoted
                FROM episodes;
                """
            ).fetchone()
        total = int(row["total"]) if row else 0
        promoted = int(row["promoted"]) if row and row["promoted"] is not None else 0
        return {
            "total_episodes": total,
            "promoted_episodes": promoted,
            "pass_rate": self.pass_rate(),
        }

