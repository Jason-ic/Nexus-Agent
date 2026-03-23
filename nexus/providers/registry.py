"""Provider 注册表 — 根据 'provider/model' 字符串路由到具体实现"""

from __future__ import annotations

from nexus.core.errors import ProviderNotFoundError
from nexus.providers.base import ModelProvider


class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, ModelProvider] = {}

    def register(self, provider: ModelProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, provider_name: str) -> ModelProvider:
        if provider_name not in self._providers:
            available = ", ".join(self._providers.keys()) or "(none)"
            raise ProviderNotFoundError(
                f"Provider '{provider_name}' not found. Available: {available}"
            )
        return self._providers[provider_name]

    def parse_model_string(self, model_string: str) -> tuple[str, str]:
        """解析 'provider/model' 格式，返回 (provider_name, model_name)"""
        if "/" not in model_string:
            raise ValueError(
                f"Invalid model string '{model_string}'. Expected format: 'provider/model'"
            )
        provider_name, model_name = model_string.split("/", 1)
        return provider_name, model_name

    @property
    def available_providers(self) -> list[str]:
        return list(self._providers.keys())


# 全局注册表
registry = ProviderRegistry()
