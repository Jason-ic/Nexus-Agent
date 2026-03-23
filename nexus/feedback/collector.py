"""用户反馈收集"""

from __future__ import annotations

from nexus.feedback.store import FeedbackStore


class FeedbackCollector:
    def __init__(self, store: FeedbackStore):
        self._store = store
        self._last_interaction_id: int | None = None

    def record(self, model: str, user_message: str, assistant_message: str) -> int:
        """记录一次交互，返回交互 ID"""
        self._last_interaction_id = self._store.save(
            model=model,
            user_message=user_message,
            assistant_message=assistant_message,
        )
        return self._last_interaction_id

    def rate(self, score: str, feedback_text: str = "") -> bool:
        """对最近的交互评分"""
        if self._last_interaction_id is None:
            return False
        self._store.update_score(self._last_interaction_id, score, feedback_text)
        return True

    def rate_by_id(self, interaction_id: int, score: str, feedback_text: str = "") -> None:
        self._store.update_score(interaction_id, score, feedback_text)
