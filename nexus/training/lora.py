"""LoRA 微调 — 基于 transformers + peft"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoRAConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    data_path: str = ""
    output_dir: str = ".nexus/training/checkpoints"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
    )
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4


def train_lora(config: LoRAConfig) -> Path:
    """执行 LoRA 微调，返回 checkpoint 路径"""
    try:
        import torch
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
    except ImportError as e:
        raise RuntimeError(
            "训练依赖未安装。请运行: pip install 'nexus-agent[training]'"
        ) from e

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # LoRA 配置
    peft_config = PeftLoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 加载训练数据
    dataset = _load_sft_dataset(config.data_path, tokenizer, config.max_seq_length)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # 保存最终 adapter
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # 保存训练元数据
    meta = {
        "base_model": config.base_model,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "num_epochs": config.num_epochs,
        "data_path": config.data_path,
    }
    (final_path / "nexus_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )

    return final_path


def merge_adapter(base_model: str, adapter_path: str, output_path: str) -> Path:
    """将 LoRA adapter 合并回基础模型"""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "训练依赖未安装。请运行: pip install 'nexus-agent[training]'"
        ) from e

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    return out


def _load_sft_dataset(
    data_path: str,
    tokenizer: Any,
    max_seq_length: int,
) -> Any:
    """从 JSONL 加载 SFT 数据并 tokenize"""
    import torch
    from torch.utils.data import Dataset as TorchDataset

    records: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    class SFTDataset(TorchDataset):
        def __init__(self, records: list[dict], tokenizer: Any, max_len: int):
            self.examples: list[dict] = []
            for rec in records:
                messages = rec.get("messages", [])
                if not messages:
                    continue
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)
                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone(),
                })

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> dict:
            return self.examples[idx]

    return SFTDataset(records, tokenizer, max_seq_length)
