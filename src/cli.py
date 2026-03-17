"""CLI entry point: agentic-employees init|train|serve|status."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="agentic-employees",
    help="ADHR meta-RL framework — agents that learn from manager feedback.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init(
    model: str = typer.Option("qwen2.5:1.5b", help="Base model to pull via Ollama"),
    config_dir: str = typer.Option(".", help="Directory to initialize config in"),
) -> None:
    """Initialize project config and pull the base model."""
    config_path = Path(config_dir)

    # Write a default .env if it doesn't exist
    env_file = config_path / ".env"
    if not env_file.exists():
        env_file.write_text(
            f"LLM_MODEL={model}\n"
            f"NATS_URL=nats://localhost:4222\n"
            f"INFERENCE_BACKEND=ollama\n"
            f"TRAINER_BACKEND=standalone\n"
            f"META_RL_ENABLED=false\n"
        )
        console.print(f"[green]Created .env[/green] at {env_file}")
    else:
        console.print(f"[yellow].env already exists[/yellow] at {env_file}")

    # Create checkpoint directory
    ckpt_dir = config_path / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    console.print(f"[green]Checkpoint directory ready:[/green] {ckpt_dir}")

    # Try pulling the model via Ollama
    console.print(f"\nPulling model [bold]{model}[/bold] via Ollama...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            console.print(f"[green]Model {model} pulled successfully[/green]")
        else:
            console.print(
                f"[yellow]Ollama pull failed (exit {result.returncode}). "
                f"Is Ollama running?[/yellow]\n{result.stderr[:200]}"
            )
    except FileNotFoundError:
        console.print(
            "[yellow]Ollama not found.[/yellow] Install it from https://ollama.com "
            "or use Docker Compose: docker compose up -d"
        )
    except subprocess.TimeoutExpired:
        console.print("[yellow]Model pull timed out after 10 minutes[/yellow]")

    console.print("\n[bold green]Initialization complete.[/bold green]")
    console.print("Next steps:")
    console.print("  agentic-employees serve     # Start NATS + workers")
    console.print("  agentic-employees train     # Start training loop")


@app.command()
def train(
    backend: str = typer.Option(
        None, help="Training backend: 'standalone' (CPU) or 'openrlhf' (GPU)"
    ),
    env: str = typer.Option("terminal", help="Environment type for training"),
    model: str = typer.Option(None, help="Override training model name"),
    batch_size: int = typer.Option(None, help="Override training batch size"),
    device: str = typer.Option(None, help="Override device (cpu/cuda)"),
) -> None:
    """Start the training loop with the selected backend."""
    # Set overrides via env vars before importing Config
    if backend:
        os.environ["TRAINER_BACKEND"] = backend
    if model:
        os.environ["TRAINING_MODEL"] = model
    if batch_size is not None:
        os.environ["TRAINING_BATCH_SIZE"] = str(batch_size)
    if device:
        os.environ["TRAINING_DEVICE"] = device

    from src.config import Config

    cfg = Config()
    effective_backend = cfg.trainer_backend

    console.print(f"[bold]Training backend:[/bold] {effective_backend}")
    console.print(f"[bold]Environment:[/bold] {env}")
    console.print(f"[bold]Model:[/bold] {cfg.training_model}")
    console.print(f"[bold]Device:[/bold] {cfg.training_device}")

    if effective_backend == "openrlhf":
        try:
            import openrlhf  # noqa: F401
        except ImportError:
            console.print(
                "[red]OpenRLHF not installed.[/red] "
                'Install with: pip install -e ".[openrlhf]"'
            )
            raise typer.Exit(1)
        console.print("[green]Using OpenRLHF backend (Ray + vLLM + DeepSpeed)[/green]")
    else:
        console.print("[green]Using standalone backend (CPU-testable GRPO)[/green]")

    console.print("\nStarting training service...")
    from src.services.training import main as training_main

    asyncio.run(training_main())


@app.command()
def serve(
    docker: bool = typer.Option(False, help="Start via Docker Compose"),
    services: str = typer.Option(
        "nats,ollama,bridge,training",
        help="Comma-separated services to start (docker mode)",
    ),
) -> None:
    """Start the agent serving infrastructure."""
    if docker:
        service_list = [s.strip() for s in services.split(",")]
        console.print(
            f"[bold]Starting services via Docker Compose:[/bold] {', '.join(service_list)}"
        )
        cmd = ["docker", "compose", "up", "-d"] + service_list
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            console.print("[green]Services started.[/green]")
            console.print(result.stdout)
        else:
            console.print(f"[red]Docker Compose failed:[/red]\n{result.stderr[:500]}")
            raise typer.Exit(1)
    else:
        console.print("[bold]Starting demo loop (NATS required)...[/bold]")
        console.print("Tip: Use --docker to start everything via Docker Compose")
        from src.__main__ import main as demo_main

        asyncio.run(demo_main())


@app.command()
def status() -> None:
    """Show status of all services."""
    table = Table(title="agentic-employees Service Status")
    table.add_column("Service", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # Check NATS
    nats_status, nats_detail = _check_nats()
    table.add_row("NATS", nats_status, nats_detail)

    # Check Ollama
    ollama_status, ollama_detail = _check_ollama()
    table.add_row("Ollama", ollama_status, ollama_detail)

    # Check Bridge
    bridge_status, bridge_detail = _check_http("http://localhost:8100/health", "Bridge")
    table.add_row("Bridge", bridge_status, bridge_detail)

    # Check Intercept Proxy
    intercept_status, intercept_detail = _check_http(
        "http://localhost:8200/health", "Intercept Proxy"
    )
    table.add_row("Intercept Proxy", intercept_status, intercept_detail)

    # Check Docker
    docker_status, docker_detail = _check_docker()
    table.add_row("Docker Compose", docker_status, docker_detail)

    # Training backend info
    from src.config import Config

    cfg = Config()
    table.add_row(
        "Training Backend",
        f"[blue]{cfg.trainer_backend}[/blue]",
        f"model={cfg.training_model}, device={cfg.training_device}",
    )

    console.print(table)


def _check_nats() -> tuple[str, str]:
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(("localhost", 4222))
        s.close()
        return "[green]UP[/green]", "localhost:4222"
    except (ConnectionRefusedError, OSError):
        return "[red]DOWN[/red]", "Cannot connect to localhost:4222"


def _check_ollama() -> tuple[str, str]:
    try:
        import urllib.request

        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                import json

                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                return "[green]UP[/green]", f"Models: {', '.join(models[:3]) or 'none'}"
        return "[yellow]UNKNOWN[/yellow]", f"HTTP {resp.status}"
    except Exception:
        return "[red]DOWN[/red]", "Cannot connect to localhost:11434"


def _check_http(url: str, name: str) -> tuple[str, str]:
    try:
        import urllib.request

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return "[green]UP[/green]", url
        return "[yellow]UNKNOWN[/yellow]", f"HTTP {resp.status}"
    except Exception:
        return "[red]DOWN[/red]", f"{name} not reachable"


def _check_docker() -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            return "[green]RUNNING[/green]", f"{len(lines)} container(s)"
        return "[yellow]NO CONTAINERS[/yellow]", "No running containers"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "[red]UNAVAILABLE[/red]", "Docker not found"


if __name__ == "__main__":
    app()
