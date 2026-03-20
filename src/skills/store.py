"""SkillStore — SQLite-backed storage for reusable skills with embedding vectors."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    skill_id: str
    name: str
    text: str
    category: str
    embedding: list[float]
    source_task_id: str = ""
    source_score: float = 0.0


class SkillStore:
    """CRUD operations for skills with embedding persistence in SQLite."""

    def __init__(self, db_path: str | Path = "skills_data/skills.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                text TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                embedding TEXT DEFAULT '[]',
                source_task_id TEXT DEFAULT '',
                source_score REAL DEFAULT 0.0
            )
        """)
        self._conn.commit()

    def add(self, skill: Skill) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO skills
               (skill_id, name, text, category, embedding, source_task_id, source_score)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                skill.skill_id,
                skill.name,
                skill.text,
                skill.category,
                json.dumps(skill.embedding),
                skill.source_task_id,
                skill.source_score,
            ),
        )
        self._conn.commit()
        logger.info("Stored skill %s: %s", skill.skill_id, skill.name)

    def get(self, skill_id: str) -> Skill | None:
        row = self._conn.execute(
            "SELECT * FROM skills WHERE skill_id = ?", (skill_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_skill(row)

    def list_all(self, category: str | None = None) -> list[Skill]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM skills WHERE category = ?", (category,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM skills").fetchall()
        return [self._row_to_skill(r) for r in rows]

    def delete(self, skill_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM skills WHERE skill_id = ?", (skill_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> Skill:
        return Skill(
            skill_id=row["skill_id"],
            name=row["name"],
            text=row["text"],
            category=row["category"],
            embedding=json.loads(row["embedding"]),
            source_task_id=row["source_task_id"],
            source_score=row["source_score"],
        )
