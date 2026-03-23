"""约束规则 Protocol"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nexus.core.types import CompletionResponse, NexusRequest, ValidationResult


@runtime_checkable
class ConstraintRule(Protocol):
    name: str

    def validate(
        self, response: CompletionResponse, request: NexusRequest
    ) -> ValidationResult: ...
