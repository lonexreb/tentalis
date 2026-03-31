"""Tests for TrajectoryStore — SQLite-backed trajectory storage."""

import pytest

from src.events.types import TrainingRolloutEvent
from src.training.trajectory_store import StoredTrajectory, TrajectoryStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_trajectories.db"
    s = TrajectoryStore(db_path=db_path)
    yield s
    s.close()


def _make_trajectory(
    trajectory_id: str = "t1",
    task_id: str = "task-1",
    worker_id: str = "worker-01",
    outcome_score: float = 0.8,
    prompt: str = "Solve 2+2",
) -> StoredTrajectory:
    return StoredTrajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        worker_id=worker_id,
        prompt=prompt,
        response="The answer is 4",
        steps=["Step 1: Add 2+2", "Step 2: Result is 4"],
        step_scores=[0.7, 0.9],
        outcome_score=outcome_score,
        created_at="2026-03-31T12:00:00+00:00",
    )


def test_add_and_get(store):
    traj = _make_trajectory()
    store.add(traj)
    retrieved = store.get("t1")
    assert retrieved is not None
    assert retrieved.task_id == "task-1"
    assert retrieved.steps == ["Step 1: Add 2+2", "Step 2: Result is 4"]
    assert retrieved.step_scores == [0.7, 0.9]
    assert retrieved.outcome_score == 0.8


def test_get_nonexistent(store):
    assert store.get("nonexistent") is None


def test_add_from_rollout(store):
    rollout = TrainingRolloutEvent(
        task_id="task-2",
        worker_id="worker-02",
        prompt="What is 3*4?",
        response="12",
        steps=["Multiply"],
        step_scores=[0.95],
        outcome_score=0.95,
    )
    stored = store.add_from_rollout(rollout)
    assert stored.task_id == "task-2"
    assert stored.trajectory_id  # auto-generated UUID
    retrieved = store.get(stored.trajectory_id)
    assert retrieved is not None
    assert retrieved.prompt == "What is 3*4?"


def test_query_by_worker_id(store):
    store.add(_make_trajectory("t1", worker_id="w-a"))
    store.add(_make_trajectory("t2", worker_id="w-b"))
    store.add(_make_trajectory("t3", worker_id="w-a"))
    results = store.query(worker_id="w-a")
    assert len(results) == 2
    assert all(r.worker_id == "w-a" for r in results)


def test_query_by_score_range(store):
    store.add(_make_trajectory("t1", outcome_score=0.3))
    store.add(_make_trajectory("t2", outcome_score=0.6))
    store.add(_make_trajectory("t3", outcome_score=0.9))
    results = store.query(min_score=0.5, max_score=0.7)
    assert len(results) == 1
    assert results[0].outcome_score == 0.6


def test_query_by_prompt_contains(store):
    store.add(_make_trajectory("t1", prompt="Solve the math problem"))
    store.add(_make_trajectory("t2", prompt="Write a poem"))
    results = store.query(prompt_contains="math")
    assert len(results) == 1
    assert "math" in results[0].prompt


def test_query_limit(store):
    for i in range(10):
        store.add(_make_trajectory(f"t{i}"))
    results = store.query(limit=3)
    assert len(results) == 3


def test_query_by_date_range(store):
    store.add(StoredTrajectory(
        trajectory_id="t1", task_id="t", worker_id="w", prompt="p", response="r",
        steps=[], step_scores=[], outcome_score=0.5, created_at="2026-01-01T00:00:00",
    ))
    store.add(StoredTrajectory(
        trajectory_id="t2", task_id="t", worker_id="w", prompt="p", response="r",
        steps=[], step_scores=[], outcome_score=0.5, created_at="2026-06-01T00:00:00",
    ))
    results = store.query(after="2026-03-01T00:00:00")
    assert len(results) == 1
    assert results[0].trajectory_id == "t2"


def test_count(store):
    assert store.count() == 0
    store.add(_make_trajectory("t1"))
    store.add(_make_trajectory("t2"))
    assert store.count() == 2


def test_sample(store):
    for i in range(20):
        store.add(_make_trajectory(f"t{i}", outcome_score=i * 0.05))
    sampled = store.sample(5)
    assert len(sampled) == 5


def test_sample_with_min_score(store):
    store.add(_make_trajectory("t1", outcome_score=0.2))
    store.add(_make_trajectory("t2", outcome_score=0.8))
    store.add(_make_trajectory("t3", outcome_score=0.9))
    sampled = store.sample(10, min_score=0.7)
    assert len(sampled) == 2
    assert all(s.outcome_score >= 0.7 for s in sampled)


def test_upsert(store):
    store.add(_make_trajectory("t1", outcome_score=0.5))
    store.add(_make_trajectory("t1", outcome_score=0.9))
    assert store.count() == 1
    assert store.get("t1").outcome_score == 0.9


def test_json_roundtrip(store):
    traj = _make_trajectory()
    traj.steps = ["Step with 'quotes'", 'Step with "double"']
    traj.step_scores = [0.123456789, 0.987654321]
    store.add(traj)
    retrieved = store.get("t1")
    assert retrieved.steps == traj.steps
    assert retrieved.step_scores == traj.step_scores
