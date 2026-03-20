"""Tests for SkillRetriever — embedding-based semantic search."""

import pytest

from src.skills.retriever import SkillRetriever, _cosine_similarity
from src.skills.store import Skill, SkillStore


@pytest.fixture
def store(tmp_path):
    s = SkillStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def test_cosine_similarity_identical():
    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_empty():
    assert _cosine_similarity([], []) == 0.0


def test_cosine_similarity_mismatched_length():
    assert _cosine_similarity([1, 2], [1, 2, 3]) == 0.0


def test_retrieve_without_encoder(store):
    """Without sentence-transformers, retrieve returns empty."""
    retriever = SkillRetriever(store, model_name="nonexistent-model")
    # This will fail to load the encoder, so retrieve returns []
    results = retriever.retrieve("test query")
    assert results == []


def test_retrieve_with_manual_embeddings(store):
    """Test retrieval logic with pre-computed embeddings (no model needed)."""
    retriever = SkillRetriever(store, top_k=2, similarity_threshold=0.0)

    # Add skills with known embeddings
    store.add(Skill(
        skill_id="s1", name="boundary check",
        text="Check edge cases", category="reasoning",
        embedding=[1.0, 0.0, 0.0],
    ))
    store.add(Skill(
        skill_id="s2", name="format output",
        text="Use JSON format", category="formatting",
        embedding=[0.0, 1.0, 0.0],
    ))
    store.add(Skill(
        skill_id="s3", name="validate input",
        text="Validate all inputs", category="reasoning",
        embedding=[0.9, 0.1, 0.0],
    ))

    # Mock encode to return a known vector
    retriever.encode = lambda text: [1.0, 0.0, 0.0]

    results = retriever.retrieve("test")
    assert len(results) == 2
    # s1 should be first (exact match), s3 second (high similarity)
    assert results[0].skill_id == "s1"
    assert results[1].skill_id == "s3"


def test_format_skills_prompt():
    retriever = SkillRetriever.__new__(SkillRetriever)
    skills = [
        Skill(skill_id="s1", name="check edge cases",
              text="Always verify boundary conditions",
              category="reasoning", embedding=[]),
    ]
    prompt = retriever.format_skills_prompt(skills)
    assert "[Relevant Skills]" in prompt
    assert "check edge cases" in prompt
    assert "Always verify boundary conditions" in prompt


def test_format_empty_skills():
    retriever = SkillRetriever.__new__(SkillRetriever)
    assert retriever.format_skills_prompt([]) == ""
