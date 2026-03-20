"""Tests for SkillEvolver — skill extraction from low-scoring feedback."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.topics import FEEDBACK_SCORED, SKILLS_CREATED
from src.events.types import FeedbackEvent, SkillCreatedEvent
from src.skills.evolver import SkillEvolver
from src.skills.retriever import SkillRetriever
from src.skills.store import SkillStore


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def store(tmp_path):
    s = SkillStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def retriever(store):
    r = SkillRetriever(store, model_name="nonexistent")
    # Mock encode to return a fixed embedding
    r.encode = lambda text: [0.1, 0.2, 0.3]
    return r


@pytest.fixture
def evolver(mock_bus, store, retriever, mock_client):
    return SkillEvolver(
        bus=mock_bus,
        store=store,
        retriever=retriever,
        client=mock_client,
        threshold=0.4,
    )


async def test_start_subscribes(evolver, mock_bus):
    await evolver.start()
    mock_bus.subscribe.assert_called_once()
    args = mock_bus.subscribe.call_args[0]
    assert args[0] == FEEDBACK_SCORED
    assert args[1] == FeedbackEvent


async def test_high_score_ignored(evolver, mock_client):
    feedback = FeedbackEvent(
        task_id="t1", manager_id="m1", worker_id="w1",
        score=0.8, textual_feedback="Great work!",
    )
    await evolver._handle_feedback(feedback)
    mock_client.chat.assert_not_called()


async def test_no_feedback_text_ignored(evolver, mock_client):
    feedback = FeedbackEvent(
        task_id="t1", manager_id="m1", worker_id="w1",
        score=0.2, textual_feedback="",
    )
    await evolver._handle_feedback(feedback)
    mock_client.chat.assert_not_called()


async def test_skill_extraction_success(evolver, mock_client, mock_bus, store):
    mock_client.chat.return_value = json.dumps({
        "skill_name": "check edge cases",
        "skill_text": "Always verify boundary conditions before returning",
        "category": "reasoning",
    })

    feedback = FeedbackEvent(
        task_id="t1", manager_id="m1", worker_id="w1",
        score=0.2, textual_feedback="Missed boundary conditions",
    )
    await evolver._handle_feedback(feedback)

    # Skill should be stored
    assert store.count() == 1
    skill = store.list_all()[0]
    assert skill.name == "check edge cases"

    # SkillCreatedEvent should be published
    mock_bus.publish.assert_called_once()
    call_args = mock_bus.publish.call_args[0]
    assert call_args[0] == SKILLS_CREATED
    event = call_args[1]
    assert isinstance(event, SkillCreatedEvent)
    assert event.skill_name == "check edge cases"
    assert event.task_id == "t1"


async def test_skill_extraction_bad_json(evolver, mock_client, mock_bus, store):
    mock_client.chat.return_value = "not valid json"

    feedback = FeedbackEvent(
        task_id="t1", manager_id="m1", worker_id="w1",
        score=0.1, textual_feedback="Bad output",
    )
    await evolver._handle_feedback(feedback)

    # No skill stored, no event published
    assert store.count() == 0
    mock_bus.publish.assert_not_called()


async def test_skill_extraction_empty_text(evolver, mock_client, mock_bus, store):
    mock_client.chat.return_value = json.dumps({
        "skill_name": "empty",
        "skill_text": "",
        "category": "general",
    })

    feedback = FeedbackEvent(
        task_id="t1", manager_id="m1", worker_id="w1",
        score=0.1, textual_feedback="Bad output",
    )
    await evolver._handle_feedback(feedback)

    assert store.count() == 0
    mock_bus.publish.assert_not_called()
