"""文件优先记忆存储 — Markdown 文件作为持久记忆"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


class MemoryStore:
    def __init__(self, memory_dir: Path):
        self._memory_dir = memory_dir
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        (self._memory_dir / "daily").mkdir(exist_ok=True)

    @property
    def long_term_path(self) -> Path:
        return self._memory_dir / "MEMORY.md"

    def daily_path(self, d: date | None = None) -> Path:
        d = d or date.today()
        return self._memory_dir / "daily" / f"{d.isoformat()}.md"

    # --- 长期记忆 ---

    def read_long_term(self) -> str:
        if self.long_term_path.exists():
            return self.long_term_path.read_text(encoding="utf-8")
        return ""

    def save_long_term(self, content: str) -> None:
        self.long_term_path.write_text(content, encoding="utf-8")

    def append_long_term(self, entry: str) -> None:
        current = self.read_long_term()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_entry = f"\n\n## {timestamp}\n\n{entry}"
        self.save_long_term(current + new_entry)

    # --- 每日日志 ---

    def read_daily(self, d: date | None = None) -> str:
        path = self.daily_path(d)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def append_daily(self, entry: str, d: date | None = None) -> None:
        path = self.daily_path(d)
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_entry = f"\n\n### {timestamp}\n\n{entry}"
        path.write_text(current + new_entry, encoding="utf-8")

    # --- 所有记忆文件 ---

    def all_memory_files(self) -> list[Path]:
        """返回所有记忆文件路径"""
        files = []
        if self.long_term_path.exists():
            files.append(self.long_term_path)
        daily_dir = self._memory_dir / "daily"
        if daily_dir.exists():
            files.extend(sorted(daily_dir.glob("*.md"), reverse=True))
        return files
