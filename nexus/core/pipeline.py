"""请求管线编排器"""

from __future__ import annotations

from nexus.config import NexusSettings
from nexus.constraints.engine import ConstraintEngine
from nexus.context.manager import ContextManager
from nexus.core.types import CompletionResponse, Message, NexusRequest
from nexus.feedback.collector import FeedbackCollector
from nexus.feedback.evaluator import AutoEvaluator
from nexus.memory.session import SessionManager
from nexus.providers.registry import ProviderRegistry


class Pipeline:
    def __init__(
        self,
        settings: NexusSettings,
        registry: ProviderRegistry,
        context_manager: ContextManager | None = None,
        session_manager: SessionManager | None = None,
        constraint_engine: ConstraintEngine | None = None,
        feedback_collector: FeedbackCollector | None = None,
        max_retries: int = 2,
    ):
        self._settings = settings
        self._registry = registry
        self._context_manager = context_manager
        self._session_manager = session_manager
        self._constraint_engine = constraint_engine
        self._feedback_collector = feedback_collector
        self._max_retries = max_retries

    async def run(self, request: NexusRequest) -> CompletionResponse:
        model_string = request.model or self._settings.default_model
        provider_name, model_name = self._registry.parse_model_string(model_string)
        provider = self._registry.get(provider_name)

        # 上下文组装
        if self._context_manager:
            messages = self._context_manager.assemble(request.messages)
        else:
            messages = list(request.messages)

        # 模型调用 + 约束重试循环
        response = None
        for attempt in range(1 + self._max_retries):
            response = await provider.complete(
                messages=messages,
                model=model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # 约束检查
            if self._constraint_engine:
                failures = self._constraint_engine.validate(response, request)
                if failures and attempt < self._max_retries:
                    # 注入失败信息重试
                    error_msg = self._constraint_engine.format_failures(failures)
                    messages.append(Message(role="assistant", content=response.content))
                    messages.append(Message(role="user", content=error_msg))
                    continue
            break

        assert response is not None

        # 记录交互 + 检测重要信息
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            None,
        )
        if last_user:
            if self._session_manager:
                self._session_manager.log_interaction(last_user, response.content)
                self._session_manager.save_if_important(request.messages)

            if self._feedback_collector:
                # 自动评分
                auto_score = "neutral"
                if self._constraint_engine:
                    failures = self._constraint_engine.validate(response, request)
                    auto_score = AutoEvaluator.score_from_constraints(failures)
                self._feedback_collector.record(
                    model=model_string,
                    user_message=last_user,
                    assistant_message=response.content,
                )
                if auto_score == "negative":
                    self._feedback_collector.rate(auto_score)

        return response
