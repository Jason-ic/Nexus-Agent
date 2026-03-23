"""OpenAI Provider — 支持 OpenAI API 及兼容接口"""

from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from nexus.core.types import CompletionResponse, Message


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return CompletionResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.name,
            usage=usage,
        )

    async def stream(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
