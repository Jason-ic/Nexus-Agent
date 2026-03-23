"""Anthropic Provider — Claude 系列模型"""

from __future__ import annotations

from typing import AsyncIterator

from anthropic import AsyncAnthropic

from nexus.core.types import CompletionResponse, Message


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        # 分离 system message
        system_msg = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        params: dict = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
        }
        if system_msg:
            params["system"] = system_msg
        if temperature is not None:
            params["temperature"] = temperature

        response = await self._client.messages.create(**params)
        content = response.content[0].text if response.content else ""
        return CompletionResponse(
            content=content,
            model=response.model,
            provider=self.name,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
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
        system_msg = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        params: dict = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
        }
        if system_msg:
            params["system"] = system_msg
        if temperature is not None:
            params["temperature"] = temperature

        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text
