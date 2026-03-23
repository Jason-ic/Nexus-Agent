"""微调模型注册 — 自动注册到 Ollama 或作为本地 HF 模型使用"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def register_to_ollama(
    model_path: str,
    model_name: str,
    base_model: str = "",
    quantize: str = "",
) -> str:
    """将微调后的模型注册到 Ollama

    Args:
        model_path: 合并后的 HF 模型路径
        model_name: 注册到 Ollama 的模型名称
        base_model: 基础模型名（用于 Modelfile FROM 指令）
        quantize: 量化方式，如 "q4_0", "q8_0"（留空不量化）

    Returns:
        注册后的模型名称
    """
    model_path = Path(model_path)

    # 生成 Modelfile
    modelfile_content = f'FROM {model_path}\n'
    if base_model:
        # 读取训练元数据
        meta_path = model_path / "nexus_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            modelfile_content = f'FROM {model_path}\n'
            modelfile_content += f'# Base: {meta.get("base_model", base_model)}\n'
            modelfile_content += f'# LoRA rank: {meta.get("lora_rank", "?")}\n'

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False,
    ) as f:
        f.write(modelfile_content)
        modelfile_path = f.name

    # ollama create
    cmd = ["ollama", "create", model_name, "-f", modelfile_path]
    if quantize:
        cmd.extend(["--quantize", quantize])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ollama 注册失败: {result.stderr}")

    Path(modelfile_path).unlink(missing_ok=True)
    return model_name


def list_nexus_models() -> list[dict]:
    """列出所有 Nexus 训练过的本地模型"""
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []

    models = []
    for line in result.stdout.strip().split("\n")[1:]:  # skip header
        parts = line.split()
        if parts and parts[0].startswith("nexus-"):
            models.append({
                "name": parts[0],
                "size": parts[2] if len(parts) > 2 else "?",
                "modified": parts[-1] if len(parts) > 3 else "?",
            })
    return models
