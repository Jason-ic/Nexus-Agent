"""内置约束规则"""

from __future__ import annotations

import json
import re

from nexus.core.types import CompletionResponse, NexusRequest, ValidationResult


class MaxLengthRule:
    """限制回复最大长度"""

    def __init__(self, max_chars: int = 10000):
        self.name = "max_length"
        self._max_chars = max_chars

    def validate(self, response: CompletionResponse, request: NexusRequest) -> ValidationResult:
        if len(response.content) > self._max_chars:
            return ValidationResult(
                passed=False,
                rule_name=self.name,
                message=f"Response too long: {len(response.content)} > {self._max_chars} chars",
            )
        return ValidationResult(passed=True, rule_name=self.name)


class NotEmptyRule:
    """回复不能为空"""

    name = "not_empty"

    def validate(self, response: CompletionResponse, request: NexusRequest) -> ValidationResult:
        if not response.content.strip():
            return ValidationResult(
                passed=False,
                rule_name=self.name,
                message="Response is empty",
            )
        return ValidationResult(passed=True, rule_name=self.name)


class JsonFormatRule:
    """要求回复是合法 JSON"""

    name = "json_format"

    def validate(self, response: CompletionResponse, request: NexusRequest) -> ValidationResult:
        try:
            json.loads(response.content)
            return ValidationResult(passed=True, rule_name=self.name)
        except json.JSONDecodeError as e:
            return ValidationResult(
                passed=False,
                rule_name=self.name,
                message=f"Invalid JSON: {e}",
            )


class RegexMatchRule:
    """回复必须匹配指定正则"""

    def __init__(self, pattern: str, description: str = ""):
        self.name = "regex_match"
        self._pattern = re.compile(pattern)
        self._description = description or f"Must match: {pattern}"

    def validate(self, response: CompletionResponse, request: NexusRequest) -> ValidationResult:
        if self._pattern.search(response.content):
            return ValidationResult(passed=True, rule_name=self.name)
        return ValidationResult(
            passed=False,
            rule_name=self.name,
            message=self._description,
        )


class NoSensitiveInfoRule:
    """检测回复中是否包含敏感信息"""

    name = "no_sensitive_info"
    _patterns = [
        r"sk-[a-zA-Z0-9]{20,}",         # API keys
        r"-----BEGIN.*PRIVATE KEY-----",  # Private keys
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card numbers
    ]

    def validate(self, response: CompletionResponse, request: NexusRequest) -> ValidationResult:
        for pattern in self._patterns:
            if re.search(pattern, response.content):
                return ValidationResult(
                    passed=False,
                    rule_name=self.name,
                    message="Response contains potentially sensitive information",
                )
        return ValidationResult(passed=True, rule_name=self.name)
