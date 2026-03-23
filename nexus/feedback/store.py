"""SQLite 交互记录持久化"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class FeedbackStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                score TEXT DEFAULT 'neutral',
                feedback_text TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.commit()
        conn.close()

    def save(
        self,
        model: str,
        user_message: str,
        assistant_message: str,
        score: str = "neutral",
        feedback_text: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.execute(
            """INSERT INTO interactions
               (timestamp, model, user_message, assistant_message, score, feedback_text, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                model,
                user_message,
                assistant_message,
                score,
                feedback_text,
                json.dumps(metadata or {}),
            ),
        )
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id  # type: ignore

    def update_score(self, interaction_id: int, score: str, feedback_text: str = "") -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            "UPDATE interactions SET score = ?, feedback_text = ? WHERE id = ?",
            (score, feedback_text, interaction_id),
        )
        conn.commit()
        conn.close()

    def get_positive_samples(self, limit: int = 100) -> list[dict]:
        """获取正样本（用于训练导出）"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM interactions WHERE score = 'positive' ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        conn = sqlite3.connect(str(self._db_path))
        total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        positive = conn.execute("SELECT COUNT(*) FROM interactions WHERE score = 'positive'").fetchone()[0]
        negative = conn.execute("SELECT COUNT(*) FROM interactions WHERE score = 'negative'").fetchone()[0]
        conn.close()
        return {"total": total, "positive": positive, "negative": negative, "neutral": total - positive - negative}
