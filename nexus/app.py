"""FastAPI API 服务"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from nexus.config import load_settings
from nexus.core.pipeline import Pipeline
from nexus.core.types import CompletionResponse, Message, NexusRequest
from nexus.providers.anthropic import AnthropicProvider
from nexus.providers.ollama import OllamaProvider
from nexus.providers.openai import OpenAIProvider
from nexus.providers.registry import ProviderRegistry


def create_app() -> FastAPI:
    app = FastAPI(title="Nexus", version="0.1.0")
    settings = load_settings(Path.cwd())
    reg = ProviderRegistry()

    for name, cfg in settings.providers.items():
        if name == "anthropic":
            reg.register(AnthropicProvider(api_key=cfg.api_key))
        else:
            provider = OpenAIProvider(api_key=cfg.api_key, base_url=cfg.base_url)
            provider.name = name
            reg.register(provider)
    reg.register(OllamaProvider(base_url=settings.ollama_base_url))

    pipeline = Pipeline(settings=settings, registry=reg)

    class CompleteRequest(BaseModel):
        messages: list[Message]
        model: str | None = None
        temperature: float | None = None
        max_tokens: int | None = None

    @app.post("/v1/complete")
    async def complete(req: CompleteRequest) -> CompletionResponse:
        nexus_req = NexusRequest(
            messages=req.messages,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return await pipeline.run(nexus_req)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "providers": reg.available_providers,
        }

    return app


app = create_app()
