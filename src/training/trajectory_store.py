"""TrajectoryStore — SQLite-backed storage for scored training trajectories."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.events.types import TrainingRolloutEvent

logger = logging.getLogger(__name__)


@dataclass
class StoredTrajectory:
    trajectory_id: str
    task_id: str
    worker_id: str
    prompt: str
    response: str
    steps: list[str]
    step_scores: list[float]
    outcome_score: float
    created_at: str  # ISO8601 string


class TrajectoryStore:
    """CRUD + query operations for scored training trajectories in SQLite."""

    def __init__(self, db_path: str | Path = "trajectory_data/trajectories.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                steps TEXT DEFAULT '[]',
                step_scores TEXT DEFAULT '[]',
                outcome_score REAL DEFAULT 0.0,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traj_worker ON trajectories(worker_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traj_score ON trajectories(outcome_score)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traj_created ON trajectories(created_at)"
        )
        self._conn.commit()

    def add(self, trajectory: StoredTrajectory) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO trajectories
               (trajectory_id, task_id, worker_id, prompt, response,
                steps, step_scores, outcome_score, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trajectory.trajectory_id,
                trajectory.task_id,
                trajectory.worker_id,
                trajectory.prompt,
                trajectory.response,
                json.dumps(trajectory.steps),
                json.dumps(trajectory.step_scores),
                trajectory.outcome_score,
                trajectory.created_at,
            ),
        )
        self._conn.commit()
        logger.info("Stored trajectory %s (score=%.2f)", trajectory.trajectory_id, trajectory.outcome_score)

    def add_from_rollout(self, rollout: TrainingRolloutEvent) -> StoredTrajectory:
        trajectory = StoredTrajectory(
            trajectory_id=str(uuid.uuid4()),
            task_id=rollout.task_id,
            worker_id=rollout.worker_id,
            prompt=rollout.prompt,
            response=rollout.response,
            steps=rollout.steps,
            step_scores=rollout.step_scores,
            outcome_score=rollout.outcome_score,
            created_at=rollout.created_at.isoformat(),
        )
        self.add(trajectory)
        return trajectory

    def get(self, trajectory_id: str) -> StoredTrajectory | None:
        row = self._conn.execute(
            "SELECT * FROM trajectories WHERE trajectory_id = ?", (trajectory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_trajectory(row)

    def query(
        self,
        *,
        prompt_contains: str | None = None,
        worker_id: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 1000,
    ) -> list[StoredTrajectory]:
        clauses: list[str] = []
        params: list = []

        if prompt_contains is not None:
            clauses.append("prompt LIKE ?")
            params.append(f"%{prompt_contains}%")
        if worker_id is not None:
            clauses.append("worker_id = ?")
            params.append(worker_id)
        if min_score is not None:
            clauses.append("outcome_score >= ?")
            params.append(min_score)
        if max_score is not None:
            clauses.append("outcome_score <= ?")
            params.append(max_score)
        if after is not None:
            clauses.append("created_at >= ?")
            params.append(after)
        if before is not None:
            clauses.append("created_at <= ?")
            params.append(before)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM trajectories WHERE {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_trajectory(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()
        return row[0]

    def sample(self, n: int, *, min_score: float | None = None) -> list[StoredTrajectory]:
        if min_score is not None:
            rows = self._conn.execute(
                "SELECT * FROM trajectories WHERE outcome_score >= ? ORDER BY RANDOM() LIMIT ?",
                (min_score, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trajectories ORDER BY RANDOM() LIMIT ?", (n,)
            ).fetchall()
        return [self._row_to_trajectory(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_trajectory(row: sqlite3.Row) -> StoredTrajectory:
        return StoredTrajectory(
            trajectory_id=row["trajectory_id"],
            task_id=row["task_id"],
            worker_id=row["worker_id"],
            prompt=row["prompt"],
            response=row["response"],
            steps=json.loads(row["steps"]),
            step_scores=json.loads(row["step_scores"]),
            outcome_score=row["outcome_score"],
            created_at=row["created_at"],
        )
