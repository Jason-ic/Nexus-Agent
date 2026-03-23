"""约束验证引擎 — 校验模型输出，失败可重试"""

from __future__ import annotations

from nexus.constraints.base import ConstraintRule
from nexus.constraints.rules import MaxLengthRule, NotEmptyRule, NoSensitiveInfoRule
from nexus.core.types import CompletionResponse, NexusRequest, ValidationResult


class ConstraintEngine:
    def __init__(self):
        self._rules: list[ConstraintRule] = []
        # 注册默认规则
        self.add_rule(NotEmptyRule())
        self.add_rule(MaxLengthRule())
        self.add_rule(NoSensitiveInfoRule())

    def add_rule(self, rule: ConstraintRule) -> None:
        self._rules.append(rule)

    def validate(
        self, response: CompletionResponse, request: NexusRequest
    ) -> list[ValidationResult]:
        """运行所有规则，返回失败的结果列表"""
        failures = []
        for rule in self._rules:
            result = rule.validate(response, request)
            if not result.passed:
                failures.append(result)
        return failures

    def format_failures(self, failures: list[ValidationResult]) -> str:
        """格式化失败信息，用于重试时注入上下文"""
        lines = ["The previous response failed the following constraints:"]
        for f in failures:
            lines.append(f"- [{f.rule_name}] {f.message}")
        lines.append("Please regenerate a response that satisfies all constraints.")
        return "\n".join(lines)
