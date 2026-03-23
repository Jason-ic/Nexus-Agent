"""动态上下文 — 对话历史滑动窗口"""

from __future__ import annotations

from nexus.core.types import Message


class DynamicContext:
    def __init__(self, max_turns: int = 50):
        self._max_turns = max_turns

    def trim(self, messages: list[Message]) -> list[Message]:
        """保留最近的 N 轮对话，始终保留 system message"""
        if len(messages) <= self._max_turns:
            return messages

        system_msgs = [m for m in messages if m.role == "system"]
        chat_msgs = [m for m in messages if m.role != "system"]

        # 保留最近的对话
        trimmed = chat_msgs[-self._max_turns:]
        return system_msgs + trimmed

    @staticmethod
    def estimate_tokens(messages: list[Message]) -> int:
        """粗略估算 token 数（1 中文字 ≈ 2 tokens，1 英文词 ≈ 1.3 tokens）"""
        total_chars = sum(len(m.content) for m in messages)
        return int(total_chars * 1.5)
