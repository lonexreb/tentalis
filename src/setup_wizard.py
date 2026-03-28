"""Interactive setup wizard for first-time project configuration."""

from __future__ import annotations

import socket
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()


class SetupWizard:
    """Multi-step interactive configuration wizard using Rich prompts."""

    def __init__(self, config_dir: Path = Path(".")) -> None:
        self._config_dir = config_dir
        self._config: dict[str, str] = {}

    def run(self) -> dict[str, str]:
        console.print(
            Panel(
                "[bold]Welcome to Tentalis setup wizard[/bold]\n"
                "This will guide you through configuring your ADHR framework.",
                title="Setup Wizard",
            )
        )

        self._step_nats()
        self._step_inference()
        self._step_training()
        self._step_skills()
        self._step_write_env()

        console.print("\n[bold green]Setup complete![/bold green]")
        return self._config

    def _step_nats(self) -> None:
        console.print("\n[bold]Step 1: NATS Message Bus[/bold]")
        nats_url = Prompt.ask(
            "NATS URL", default="nats://localhost:4222"
        )
        self._config["NATS_URL"] = nats_url

        # Validate connection
        try:
            host = nats_url.split("://")[1].split(":")[0]
            port = int(nats_url.split(":")[-1])
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((host, port))
            s.close()
            console.print("[green]  NATS connection verified[/green]")
        except (ConnectionRefusedError, OSError, ValueError, IndexError):
            console.print("[yellow]  NATS not reachable — start it before running[/yellow]")

    def _step_inference(self) -> None:
        console.print("\n[bold]Step 2: Inference Backend[/bold]")
        backend = Prompt.ask(
            "Backend", choices=["ollama", "openai"], default="ollama"
        )
        self._config["INFERENCE_BACKEND"] = backend

        if backend == "ollama":
            host = Prompt.ask("Ollama host", default="http://localhost:11434")
            self._config["OLLAMA_HOST"] = host
            model = Prompt.ask("Model", default="qwen2.5:1.5b")
            self._config["LLM_MODEL"] = model

            # Health check
            try:
                req = urllib.request.Request(f"{host}/api/tags")
                with urllib.request.urlopen(req, timeout=3):
                    console.print("[green]  Ollama is running[/green]")
            except Exception:
                console.print("[yellow]  Ollama not reachable — start it before running[/yellow]")
        else:
            base_url = Prompt.ask("OpenAI-compatible base URL")
            self._config["INFERENCE_BASE_URL"] = base_url
            api_key = Prompt.ask("API key", password=True, default="")
            if api_key:
                self._config["INFERENCE_API_KEY"] = api_key
            model = Prompt.ask("Model name", default="qwen2.5:1.5b")
            self._config["LLM_MODEL"] = model

    def _step_training(self) -> None:
        console.print("\n[bold]Step 3: Training Backend[/bold]")
        backend = Prompt.ask(
            "Backend",
            choices=["standalone", "openrlhf", "tinker"],
            default="standalone",
        )
        self._config["TRAINER_BACKEND"] = backend

        if backend == "tinker":
            api_key = Prompt.ask("Tinker API key", password=True)
            self._config["TINKER_API_KEY"] = api_key
            base_url = Prompt.ask(
                "Tinker base URL",
                default="https://api.tinker.thinkingmachines.ai",
            )
            self._config["TINKER_BASE_URL"] = base_url
            console.print("[green]  Tinker backend configured (no GPU required)[/green]")
        elif backend == "openrlhf":
            device = Prompt.ask("Device", choices=["cpu", "cuda"], default="cuda")
            self._config["TRAINING_DEVICE"] = device
            console.print("[green]  OpenRLHF backend configured[/green]")
        else:
            device = Prompt.ask("Device", choices=["cpu", "cuda"], default="cpu")
            self._config["TRAINING_DEVICE"] = device
            console.print("[green]  Standalone GRPO backend configured[/green]")

        model = Prompt.ask("Training model", default="distilgpt2")
        self._config["TRAINING_MODEL"] = model

    def _step_skills(self) -> None:
        console.print("\n[bold]Step 4: SkillRL[/bold]")
        enabled = Confirm.ask("Enable SkillRL (fast feedback via skill injection)?", default=False)
        self._config["SKILLS_ENABLED"] = str(enabled).lower()

        if enabled:
            threshold = Prompt.ask("Skill evolution threshold (0-1)", default="0.4")
            self._config["SKILL_EVOLUTION_THRESHOLD"] = threshold
            top_k = Prompt.ask("Skills to retrieve per task", default="3")
            self._config["SKILL_RETRIEVAL_TOP_K"] = top_k
            console.print("[green]  SkillRL enabled[/green]")

    def _step_write_env(self) -> None:
        console.print("\n[bold]Step 5: Write configuration[/bold]")
        env_file = self._config_dir / ".env"

        if env_file.exists():
            overwrite = Confirm.ask(
                f".env already exists at {env_file}. Overwrite?", default=False
            )
            if not overwrite:
                console.print("[yellow]Skipping .env write[/yellow]")
                return

        lines = [f"{k}={v}" for k, v in self._config.items()]
        env_file.write_text("\n".join(lines) + "\n")
        console.print(f"[green]Wrote .env to {env_file}[/green]")
