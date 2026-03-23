"""ModelProvider Protocol — 所有 provider 的统一接口"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from nexus.core.types import CompletionResponse, Message


@runtime_checkable
class ModelProvider(Protocol):
    """模型提供者协议。所有 provider 实现此接口。"""

    name: str

    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse: ...

    async def stream(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]: ...
