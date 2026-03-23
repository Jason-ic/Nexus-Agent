"""全局配置"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ProviderConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None


class NexusSettings(BaseSettings):
    default_model: str = "ollama/qwen3"

    # Provider 配置（从 yaml 加载）
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    # Ollama 默认配置
    ollama_base_url: str = "http://localhost:11434"

    # 服务
    host: str = "0.0.0.0"
    port: int = 8000

    # 路径
    nexus_dir: str = ".nexus"

    model_config = {"env_prefix": "NEXUS_"}


def load_settings(project_dir: Path | None = None) -> NexusSettings:
    """加载配置：环境变量 + .nexus/config.yaml"""
    kwargs: dict[str, Any] = {}
    if project_dir:
        config_file = project_dir / ".nexus" / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                file_config = yaml.safe_load(f) or {}

            # 提取 providers 配置
            if "providers" in file_config:
                providers_raw = file_config.pop("providers")
                kwargs["providers"] = {
                    name: ProviderConfig(**cfg)
                    for name, cfg in providers_raw.items()
                }

            kwargs.update(file_config)

    return NexusSettings(**kwargs)
