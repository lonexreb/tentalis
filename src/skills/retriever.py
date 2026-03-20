"""SkillRetriever — embedding-based semantic search for relevant skills."""

from __future__ import annotations

import logging
import math

from src.skills.store import Skill, SkillStore

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SkillRetriever:
    """Finds semantically relevant skills for a given task prompt.

    Uses sentence-transformers for encoding and cosine similarity for matching.
    Falls back to returning no skills if sentence-transformers is unavailable.
    """

    def __init__(
        self,
        store: SkillStore,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        similarity_threshold: float = 0.3,
    ) -> None:
        self._store = store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._encoder = None
        self._model_name = model_name

    def _ensure_encoder(self) -> bool:
        if self._encoder is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._model_name)
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                'Install with: pip install -e ".[skills]"'
            )
            return False

    def encode(self, text: str) -> list[float]:
        """Encode text into a dense embedding vector."""
        if not self._ensure_encoder():
            return []
        embedding = self._encoder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def retrieve(self, prompt: str, category: str | None = None) -> list[Skill]:
        """Find top-K skills semantically relevant to the given prompt."""
        query_embedding = self.encode(prompt)
        if not query_embedding:
            return []

        all_skills = self._store.list_all(category=category)
        if not all_skills:
            return []

        scored: list[tuple[float, Skill]] = []
        for skill in all_skills:
            if not skill.embedding:
                continue
            sim = _cosine_similarity(query_embedding, skill.embedding)
            if sim >= self._similarity_threshold:
                scored.append((sim, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [skill for _, skill in scored[: self._top_k]]
        logger.info(
            "Retrieved %d skills for prompt (top similarity: %.3f)",
            len(results),
            scored[0][0] if scored else 0.0,
        )
        return results

    def format_skills_prompt(self, skills: list[Skill]) -> str:
        """Format retrieved skills into a system prompt section."""
        if not skills:
            return ""
        lines = ["[Relevant Skills]"]
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.text}")
        lines.append("[End Skills]\n")
        return "\n".join(lines)
