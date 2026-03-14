from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class TaskStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"


class TaskEvent(BaseModel):
    task_id: str = Field(default_factory=_new_id)
    manager_id: str
    task_type: str
    prompt: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class ResultEvent(BaseModel):
    task_id: str
    worker_id: str
    prompt: str = ""
    result: str
    status: TaskStatus
    steps: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    model_version: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class FeedbackEvent(BaseModel):
    task_id: str
    manager_id: str
    worker_id: str
    score: float = Field(ge=0.0, le=1.0)
    textual_feedback: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class TrainingRolloutEvent(BaseModel):
    task_id: str
    worker_id: str
    prompt: str
    response: str
    steps: list[str] = Field(default_factory=list)
    step_scores: list[float] = Field(default_factory=list)
    outcome_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)


class ModelUpdateEvent(BaseModel):
    model_version: str
    checkpoint_path: str
    target_worker_id: str | None = None
    metrics: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class SessionEvent(BaseModel):
    """Logged by the intercept proxy for every inference call."""

    session_id: str = Field(default_factory=_new_id)
    worker_id: str = ""
    model: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)
    response: str = ""
    token_logprobs: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class OPDHintEvent(BaseModel):
    """On-Policy Distillation hint derived from manager textual feedback."""

    task_id: str
    worker_id: str
    hint_text: str
    teacher_logprobs: list[float] = Field(default_factory=list)
    original_response: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class CombinedRolloutEvent(BaseModel):
    """Joined RL rollout + OPD hint for combined training."""

    task_id: str
    worker_id: str
    prompt: str
    response: str
    steps: list[str] = Field(default_factory=list)
    step_scores: list[float] = Field(default_factory=list)
    outcome_score: float = Field(ge=0.0, le=1.0)
    # OPD fields (empty if pure RL rollout)
    hint_text: str = ""
    teacher_logprobs: list[float] = Field(default_factory=list)
    has_opd: bool = False
    created_at: datetime = Field(default_factory=_utcnow)


class ManagerMetaRollout(BaseModel):
    """Meta-RL rollout for training the manager's feedback quality."""

    manager_id: str
    feedback_task_ids: list[str] = Field(default_factory=list)
    improvement_delta: float = 0.0
    feedback_texts: list[str] = Field(default_factory=list)
    mean_score_before: float = 0.0
    mean_score_after: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)
