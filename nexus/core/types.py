"""共享数据模型"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NexusRequest(BaseModel):
    messages: list[Message]
    model: str | None = None  # e.g. "openai/gpt-4o", "ollama/llama3"
    context_sources: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None


class ValidationResult(BaseModel):
    passed: bool
    rule_name: str
    message: str = ""


class FeedbackScore(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class MemorySnippet(BaseModel):
    source: str  # file path
    content: str
    score: float = 0.0
