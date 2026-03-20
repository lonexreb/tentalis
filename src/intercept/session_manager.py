"""SessionManager — tracks active proxy sessions with conversation history."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    worker_id: str = ""
    model: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return len([m for m in self.messages if m.get("role") == "user"])


class SessionManager:
    """Manages active inference sessions for the intercept proxy.

    Tracks conversation history per session, enabling multi-turn
    data collection and per-session skill injection.
    """

    def __init__(self, *, timeout_seconds: float = 1800.0) -> None:
        self._sessions: dict[str, Session] = {}
        self._timeout = timeout_seconds

    def get_or_create(
        self,
        session_id: str | None,
        *,
        worker_id: str = "",
        model: str = "",
    ) -> tuple[Session, bool]:
        """Get an existing session or create a new one.

        Returns (session, is_new).
        """
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_active = time.time()
            return session, False

        new_id = session_id or str(uuid.uuid4())
        session = Session(
            session_id=new_id,
            worker_id=worker_id,
            model=model,
        )
        self._sessions[new_id] = session
        logger.info("New session %s (worker=%s)", new_id, worker_id)
        return session, True

    def add_turn(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        response: str,
    ) -> None:
        """Append a request/response turn to session history."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.messages.extend(messages)
        session.messages.append({"role": "assistant", "content": response})
        session.last_active = time.time()

    def end(self, session_id: str) -> Session | None:
        """End a session and return it."""
        return self._sessions.pop(session_id, None)

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def cleanup_expired(self) -> list[str]:
        """Remove sessions that have exceeded the timeout. Returns expired IDs."""
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active > self._timeout
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return expired

    @property
    def active_count(self) -> int:
        return len(self._sessions)
