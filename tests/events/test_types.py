"""Tests for event type serialization — no NATS required."""

from src.events.types import (
    CombinedRolloutEvent,
    FeedbackEvent,
    ManagerMetaRollout,
    ModelUpdateEvent,
    OPDHintEvent,
    ResultEvent,
    SessionEvent,
    TaskEvent,
    TaskStatus,
    TrainingRolloutEvent,
)


def test_task_event_roundtrip():
    event = TaskEvent(manager_id="mgr-1", task_type="coding", prompt="Write hello world")
    data = event.model_dump_json()
    restored = TaskEvent.model_validate_json(data)
    assert restored.task_id == event.task_id
    assert restored.prompt == "Write hello world"
    assert restored.created_at == event.created_at


def test_result_event_roundtrip():
    event = ResultEvent(
        task_id="abc-123",
        worker_id="w-1",
        prompt="Write hello world",
        result="done",
        status=TaskStatus.SUCCESS,
        steps=["step1", "step2"],
        elapsed_seconds=1.5,
    )
    data = event.model_dump_json()
    restored = ResultEvent.model_validate_json(data)
    assert restored.status == TaskStatus.SUCCESS
    assert restored.prompt == "Write hello world"
    assert restored.steps == ["step1", "step2"]
    assert restored.elapsed_seconds == 1.5


def test_result_event_model_version_roundtrip():
    event = ResultEvent(
        task_id="abc-123",
        worker_id="w-1",
        result="done",
        status=TaskStatus.SUCCESS,
        model_version="v0005",
    )
    data = event.model_dump_json()
    restored = ResultEvent.model_validate_json(data)
    assert restored.model_version == "v0005"


def test_result_event_model_version_defaults_none():
    event = ResultEvent(
        task_id="abc-123",
        worker_id="w-1",
        result="done",
        status=TaskStatus.SUCCESS,
    )
    assert event.model_version is None


def test_result_event_prompt_defaults_empty():
    event = ResultEvent(
        task_id="abc-123",
        worker_id="w-1",
        result="done",
        status=TaskStatus.SUCCESS,
    )
    assert event.prompt == ""


def test_feedback_event_roundtrip():
    event = FeedbackEvent(
        task_id="abc-123",
        manager_id="mgr-1",
        worker_id="w-1",
        score=0.85,
        textual_feedback="Good work",
    )
    data = event.model_dump_json()
    restored = FeedbackEvent.model_validate_json(data)
    assert restored.score == 0.85
    assert restored.textual_feedback == "Good work"


def test_training_rollout_event_roundtrip():
    event = TrainingRolloutEvent(
        task_id="abc-123",
        worker_id="w-1",
        prompt="Write code",
        response="print('hello')",
        steps=["parse", "generate"],
        step_scores=[0.9, 0.8],
        outcome_score=0.85,
    )
    data = event.model_dump_json()
    restored = TrainingRolloutEvent.model_validate_json(data)
    assert restored.step_scores == [0.9, 0.8]
    assert restored.outcome_score == 0.85


def test_model_update_event_roundtrip():
    event = ModelUpdateEvent(
        model_version="v0.1",
        checkpoint_path="/tmp/ckpt",
        metrics={"loss": 0.5},
    )
    data = event.model_dump_json()
    restored = ModelUpdateEvent.model_validate_json(data)
    assert restored.model_version == "v0.1"
    assert restored.metrics == {"loss": 0.5}


def test_task_event_auto_generates_id():
    e1 = TaskEvent(manager_id="m", task_type="t", prompt="p")
    e2 = TaskEvent(manager_id="m", task_type="t", prompt="p")
    assert e1.task_id != e2.task_id


def test_feedback_score_bounds():
    import pytest

    with pytest.raises(Exception):
        FeedbackEvent(task_id="x", manager_id="m", worker_id="w", score=1.5)
    with pytest.raises(Exception):
        FeedbackEvent(task_id="x", manager_id="m", worker_id="w", score=-0.1)


def test_model_update_target_worker_id():
    event = ModelUpdateEvent(
        model_version="v1",
        checkpoint_path="/ckpt",
        target_worker_id="w1",
    )
    data = event.model_dump_json()
    restored = ModelUpdateEvent.model_validate_json(data)
    assert restored.target_worker_id == "w1"


def test_model_update_target_worker_id_defaults_none():
    event = ModelUpdateEvent(model_version="v1", checkpoint_path="/ckpt")
    assert event.target_worker_id is None


def test_session_event_roundtrip():
    event = SessionEvent(
        worker_id="w1", model="qwen2.5:1.5b",
        messages=[{"role": "user", "content": "hi"}],
        response="hello", token_logprobs=[-0.5, -1.0],
    )
    data = event.model_dump_json()
    restored = SessionEvent.model_validate_json(data)
    assert restored.worker_id == "w1"
    assert restored.response == "hello"
    assert restored.token_logprobs == [-0.5, -1.0]


def test_opd_hint_event_roundtrip():
    event = OPDHintEvent(
        task_id="t1", worker_id="w1",
        hint_text="use DP", teacher_logprobs=[-0.3],
        original_response="bad answer",
    )
    data = event.model_dump_json()
    restored = OPDHintEvent.model_validate_json(data)
    assert restored.hint_text == "use DP"
    assert restored.teacher_logprobs == [-0.3]


def test_combined_rollout_event_roundtrip():
    event = CombinedRolloutEvent(
        task_id="t1", worker_id="w1", prompt="p", response="r",
        outcome_score=0.8, has_opd=True, hint_text="hint",
    )
    data = event.model_dump_json()
    restored = CombinedRolloutEvent.model_validate_json(data)
    assert restored.has_opd is True
    assert restored.hint_text == "hint"


def test_manager_meta_rollout_roundtrip():
    event = ManagerMetaRollout(
        manager_id="m1",
        feedback_task_ids=["t1", "t2"],
        improvement_delta=0.15,
        feedback_texts=["do better", "nice"],
        mean_score_before=0.4,
        mean_score_after=0.55,
    )
    data = event.model_dump_json()
    restored = ManagerMetaRollout.model_validate_json(data)
    assert restored.improvement_delta == 0.15
    assert len(restored.feedback_task_ids) == 2
