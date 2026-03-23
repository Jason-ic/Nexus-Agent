"""从 feedback store 导出训练数据（SFT / DPO 格式）"""

from __future__ import annotations

import json
from pathlib import Path

from nexus.feedback.store import FeedbackStore


class TrainingExporter:
    def __init__(self, feedback_store: FeedbackStore):
        self._store = feedback_store

    def export_sft(self, output_path: Path, limit: int = 1000) -> int:
        """导出正样本为 SFT 格式 (JSONL)，每行一个对话"""
        samples = self._store.get_positive_samples(limit=limit)
        if not samples:
            return 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                record = {
                    "messages": [
                        {"role": "user", "content": s["user_message"]},
                        {"role": "assistant", "content": s["assistant_message"]},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(samples)

    def export_dpo(self, output_path: Path, limit: int = 500) -> int:
        """导出 DPO 格式：正样本 vs 负样本配对

        DPO 需要同一 prompt 下的 chosen/rejected 对。
        这里用正面评分的回复作为 chosen，负面评分的作为 rejected。
        只有当同一用户消息同时存在正/负样本时才导出。
        """
        import sqlite3

        conn = sqlite3.connect(str(self._store._db_path))
        conn.row_factory = sqlite3.Row

        # 找到既有正面又有负面评分的 user_message
        rows = conn.execute("""
            SELECT p.user_message,
                   p.assistant_message AS chosen,
                   n.assistant_message AS rejected
            FROM interactions p
            JOIN interactions n ON p.user_message = n.user_message
            WHERE p.score = 'positive' AND n.score = 'negative'
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        if not rows:
            return 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in rows:
                record = {
                    "prompt": r["user_message"],
                    "chosen": r["chosen"],
                    "rejected": r["rejected"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(rows)
