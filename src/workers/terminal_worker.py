"""TerminalWorker — executes bash commands in Docker containers."""

from __future__ import annotations

import asyncio
import logging

from src.events.bus import EventBus
from src.events.types import ResultEvent, TaskEvent, TaskStatus
from src.workers.base import BaseWorker

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
DEFAULT_IMAGE = "python:3.10-slim"


class TerminalWorker(BaseWorker):
    """Executes task prompts as bash commands inside isolated Docker containers.

    Steps = individual commands. Each command is run sequentially.
    Security: read-only rootfs, no network, resource limits, timeout.
    """

    environment_type = "terminal"

    def __init__(
        self,
        worker_id: str,
        bus: EventBus,
        task_types: list[str] | None = None,
        *,
        docker_image: str = DEFAULT_IMAGE,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(worker_id, bus, task_types or ["terminal"])
        self._image = docker_image
        self._timeout = timeout

    async def process(self, task: TaskEvent) -> ResultEvent:
        commands = self._parse_commands(task.prompt)
        steps: list[str] = []
        outputs: list[str] = []

        for cmd in commands:
            steps.append(cmd)
            exit_code, output = await self._run_in_docker(cmd)
            outputs.append(output)
            if exit_code != 0:
                return ResultEvent(
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    result="\n".join(outputs),
                    status=TaskStatus.FAILED,
                    steps=steps,
                )

        return ResultEvent(
            task_id=task.task_id,
            worker_id=self.worker_id,
            result="\n".join(outputs),
            status=TaskStatus.SUCCESS,
            steps=steps,
        )

    def _parse_commands(self, prompt: str) -> list[str]:
        """Extract commands from prompt — one per line, skip comments/blanks."""
        return [
            line.strip()
            for line in prompt.strip().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    async def _run_in_docker(self, command: str) -> tuple[int, str]:
        """Run a single command in an isolated Docker container."""
        docker_cmd = [
            "docker", "run", "--rm",
            "--read-only",
            "--network=none",
            "--memory=256m",
            "--cpus=0.5",
            "--pids-limit=64",
            self._image,
            "bash", "-c", command,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
            return proc.returncode or 0, stdout.decode(errors="replace")
        except asyncio.TimeoutError:
            proc.kill()
            return 1, f"Command timed out after {self._timeout}s"
        except FileNotFoundError:
            return 1, "Docker not available"
