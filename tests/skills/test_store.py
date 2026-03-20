"""Tests for SkillStore — SQLite-backed skill storage."""

import tempfile
from pathlib import Path

import pytest

from src.skills.store import Skill, SkillStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_skills.db"
    s = SkillStore(db_path=db_path)
    yield s
    s.close()


def _make_skill(skill_id: str = "s1", name: str = "test skill") -> Skill:
    return Skill(
        skill_id=skill_id,
        name=name,
        text="Always check boundary conditions",
        category="reasoning",
        embedding=[0.1, 0.2, 0.3],
        source_task_id="task-1",
        source_score=0.3,
    )


def test_add_and_get(store):
    skill = _make_skill()
    store.add(skill)
    retrieved = store.get("s1")
    assert retrieved is not None
    assert retrieved.name == "test skill"
    assert retrieved.embedding == [0.1, 0.2, 0.3]
    assert retrieved.category == "reasoning"


def test_get_nonexistent(store):
    assert store.get("nonexistent") is None


def test_list_all(store):
    store.add(_make_skill("s1", "skill one"))
    store.add(_make_skill("s2", "skill two"))
    skills = store.list_all()
    assert len(skills) == 2


def test_list_by_category(store):
    store.add(_make_skill("s1", "reasoning skill"))
    s2 = _make_skill("s2", "formatting skill")
    s2.category = "formatting"
    store.add(s2)
    reasoning_skills = store.list_all(category="reasoning")
    assert len(reasoning_skills) == 1
    assert reasoning_skills[0].name == "reasoning skill"


def test_delete(store):
    store.add(_make_skill())
    assert store.count() == 1
    deleted = store.delete("s1")
    assert deleted is True
    assert store.count() == 0


def test_delete_nonexistent(store):
    assert store.delete("nope") is False


def test_count(store):
    assert store.count() == 0
    store.add(_make_skill("s1"))
    store.add(_make_skill("s2"))
    assert store.count() == 2


def test_upsert(store):
    store.add(_make_skill("s1", "original"))
    store.add(_make_skill("s1", "updated"))
    assert store.count() == 1
    assert store.get("s1").name == "updated"
