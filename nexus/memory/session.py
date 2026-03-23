"""会话管理 — 自动记录对话到每日日志"""

from __future__ import annotations

from nexus.core.types import Message
from nexus.memory.store import MemoryStore


class SessionManager:
    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store

    def log_interaction(self, user_msg: str, assistant_msg: str) -> None:
        """记录一次交互到每日日志"""
        entry = f"**User:** {user_msg}\n\n**Assistant:** {assistant_msg}"
        self._store.append_daily(entry)

    def extract_key_info(self, messages: list[Message]) -> str | None:
        """从对话中提取值得长期记忆的信息（MVP: 简单启发式）"""
        # 检测用户是否明确要求记住某些内容
        trigger_phrases = ["记住", "记一下", "以后记得", "remember", "注意以后"]
        for msg in messages:
            if msg.role == "user":
                for phrase in trigger_phrases:
                    if phrase in msg.content:
                        return msg.content
        return None

    def save_if_important(self, messages: list[Message]) -> bool:
        """如果对话包含重要信息，保存到长期记忆"""
        key_info = self.extract_key_info(messages)
        if key_info:
            self._store.append_long_term(key_info)
            return True
        return False
