"""自动评估 — 基于约束检查结果生成隐式反馈"""

from __future__ import annotations

from nexus.core.types import ValidationResult


class AutoEvaluator:
    @staticmethod
    def score_from_constraints(failures: list[ValidationResult]) -> str:
        """约束全部通过 → neutral（等待用户评分），有失败 → negative"""
        if failures:
            return "negative"
        return "neutral"
