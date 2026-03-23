"""混合检索 — SQLite FTS5 关键词搜索（MVP 先不做向量）"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from nexus.core.types import MemorySnippet
from nexus.memory.store import MemoryStore


class MemorySearch:
    def __init__(self, memory_store: MemoryStore, db_path: Path | None = None):
        self._store = memory_store
        self._db_path = db_path or (memory_store._memory_dir / "search.db")
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(source, content)"
        )
        conn.commit()
        conn.close()

    def index(self) -> int:
        """重建索引：扫描所有记忆文件，写入 FTS5"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("DELETE FROM memory_fts")

        count = 0
        for file_path in self._store.all_memory_files():
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            # 按段落分块
            chunks = self._split_chunks(content)
            for chunk in chunks:
                conn.execute(
                    "INSERT INTO memory_fts(source, content) VALUES (?, ?)",
                    (str(file_path), chunk),
                )
                count += 1

        conn.commit()
        conn.close()
        return count

    def search(self, query: str, limit: int = 5) -> list[MemorySnippet]:
        """FTS5 关键词搜索"""
        # 先重建索引（MVP 简单做法，后续改为增量）
        self.index()

        conn = sqlite3.connect(str(self._db_path))
        # FTS5 match 查询
        try:
            rows = conn.execute(
                """
                SELECT source, content, rank
                FROM memory_fts
                WHERE memory_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # 查询语法不兼容时 fallback 到 LIKE
            rows = conn.execute(
                """
                SELECT source, content, 0 as rank
                FROM memory_fts
                WHERE content LIKE ?
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()
        conn.close()

        return [
            MemorySnippet(source=row[0], content=row[1], score=abs(row[2]))
            for row in rows
        ]

    @staticmethod
    def _split_chunks(text: str, max_len: int = 500) -> list[str]:
        """按段落分块，超长段落截断"""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(current) + len(p) > max_len and current:
                chunks.append(current)
                current = p
            else:
                current = f"{current}\n\n{p}" if current else p
        if current:
            chunks.append(current)
        return chunks
