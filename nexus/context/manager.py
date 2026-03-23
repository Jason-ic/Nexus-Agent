"""上下文管理器 — 在 token 预算内组装静态规则 + 记忆 + 对话历史"""

from __future__ import annotations

from pathlib import Path

from nexus.context.dynamic import DynamicContext
from nexus.context.static import StaticContext
from nexus.core.types import Message, MemorySnippet
from nexus.memory.search import MemorySearch


class ContextManager:
    def __init__(
        self,
        static: StaticContext,
        dynamic: DynamicContext,
        memory_search: MemorySearch | None = None,
        token_budget: int = 8000,
    ):
        self._static = static
        self._dynamic = dynamic
        self._memory_search = memory_search
        self._token_budget = token_budget

    def assemble(self, messages: list[Message]) -> list[Message]:
        """组装最终发送给模型的消息列表。

        优先级: 静态规则 > 记忆检索 > 对话历史（从旧裁剪）
        """
        result: list[Message] = []
        used_tokens = 0

        # 1. 静态上下文 → system message
        static_content = self._static.assemble()
        if static_content:
            system_msg = Message(role="system", content=static_content)
            est = self._dynamic.estimate_tokens([system_msg])
            if est < self._token_budget * 0.3:  # 最多占 30% 预算
                result.append(system_msg)
                used_tokens += est

        # 2. 记忆检索 → 追加到 system message
        if self._memory_search and messages:
            # 用最后一条用户消息作为查询
            last_user = next(
                (m for m in reversed(messages) if m.role == "user"), None
            )
            if last_user:
                snippets = self._memory_search.search(last_user.content, limit=3)
                if snippets:
                    memory_text = self._format_memories(snippets)
                    mem_tokens = int(len(memory_text) * 1.5)
                    if used_tokens + mem_tokens < self._token_budget * 0.5:
                        if result and result[0].role == "system":
                            result[0] = Message(
                                role="system",
                                content=result[0].content + "\n\n" + memory_text,
                            )
                        else:
                            result.insert(0, Message(role="system", content=memory_text))
                        used_tokens += mem_tokens

        # 3. 对话历史（裁剪以适应剩余预算）
        remaining_budget = self._token_budget - used_tokens
        chat_msgs = [m for m in messages if m.role != "system"]

        # 从旧到新裁剪
        while (
            chat_msgs
            and self._dynamic.estimate_tokens(chat_msgs) > remaining_budget
        ):
            chat_msgs.pop(0)

        result.extend(chat_msgs)
        return result

    @staticmethod
    def _format_memories(snippets: list[MemorySnippet]) -> str:
        parts = ["# Relevant Memories\n"]
        for s in snippets:
            parts.append(f"- {s.content[:200]}")
        return "\n".join(parts)
