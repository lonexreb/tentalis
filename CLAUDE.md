# agentic-employees

Managers managing AI agents with appraisal-style feedback loops. Agents learn continuously from manager feedback using RLHF/GRPO — like performance reviews that actually improve performance.

## Architecture

Event-Driven Architecture (EDA):
- **Control Plane**: OpenClaw (identity, memory, channels, UI)
- **Event Broker**: NATS (pub/sub for agent coordination)
- **Training Plane**: OpenRLHF + OpenClaw-RL (GRPO/DAPO/AgentPRM)
- **Inference**: vLLM + Ray for distributed serving

## Language

**Python only.** Requires Python >= 3.10. All components — RL training, agent logic, NATS clients, PRM evaluator — are Python.

## Current Dependencies (pyproject.toml)

- [nats-py](https://github.com/nats-io/nats.py) — NATS client for event bus
- [pydantic](https://docs.pydantic.dev/) — event type serialization (v2)
- [ollama](https://github.com/ollama/ollama-python) — LLM inference via Ollama (async client)

Dev: pytest, pytest-asyncio, ruff

Training (optional): `pip install -e ".[training]"` — torch, transformers, peft

## Planned Dependencies (future phases)

- [OpenClaw](https://github.com/openclaw/openclaw) — control plane (identity, memory, channels)
- [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) — continuous learning from feedback (async GRPO)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — heavy RL training (PPO/GRPO/DAPO/REINFORCE++)
- [PicoClaw](https://github.com/sipeed/picoclaw) — future edge deployment (<10MB RAM)
- [M³HF](https://github.com/cooperativex/M3HF) — multi-phase feedback from mixed-quality humans
- Ray, vLLM, DeepSpeed — distributed compute and inference

## Directory Structure

```
src/
├── __main__.py    # Demo entry point (python -m src) — auto-detects Ollama, falls back to EchoWorker
├── config.py      # Frozen dataclass with env var defaults (NATS, Ollama, timeouts)
├── manager/
│   └── manager.py # Manager agent (assign tasks, wait for results, publish feedback)
├── workers/
│   ├── base.py         # BaseWorker ABC (subscribe, handle, process pattern)
│   ├── echo_worker.py  # EchoWorker — echoes prompt back with PRM steps (testing)
│   └── llm_worker.py   # LLMWorker — real LLM inference via Ollama with step parsing
├── events/
│   ├── types.py   # Pydantic v2 event models (TaskEvent, ResultEvent, FeedbackEvent, etc.)
│   ├── topics.py  # Topic constants and helpers
│   └── bus.py     # EventBus wrapping nats-py (connect, publish, subscribe, drain)
├── rewards/
│   ├── scorer.py        # StepScorer protocol + LLMJudgeScorer (LLM-as-judge PRM)
│   ├── prompts.py       # STEP_JUDGE_PROMPT template for step-level evaluation
│   └── prm_evaluator.py # PRMEvaluator — subscribes to results, scores steps, publishes rollouts
├── training/
│   ├── bridge.py        # RolloutBuffer + NATSTrainingBridge (batch rollouts for RL trainer)
│   ├── grpo.py          # GRPO math: compute_group_advantages, clipped_surrogate_loss, kl_penalty
│   ├── trainer.py       # Trainer protocol, TrainStepResult, MockTrainer, GRPOTrainer (LoRA)
│   └── loop.py          # TrainingLoop orchestrator (bridge → trainer → ModelUpdateEvent)
config/
└── openclaw/      # SOUL.md, IDENTITY.md templates per agent (future)
tests/
├── events/
│   ├── test_types.py  # Serialization roundtrip tests (standalone)
│   └── test_bus.py    # EventBus pub/sub tests (requires NATS)
├── rewards/
│   ├── test_scorer.py        # LLMJudgeScorer tests (mocked Ollama)
│   └── test_prm_evaluator.py # PRMEvaluator tests (mocked scorer)
├── training/
│   ├── test_bridge.py        # RolloutBuffer unit tests
│   ├── test_grpo.py          # GRPO advantage math + torch loss/KL tests
│   └── test_trainer.py       # MockTrainer + GRPOTrainer protocol/integration tests
└── test_integration.py # Full manager→worker→PRM→rollout loop (requires NATS)
docs/
└── architecture/  # Diagrams and ADRs
```

## Code Style

- PEP 8
- Type hints on all function signatures
- Docstrings on public APIs only (not internal helpers)
- No unnecessary abstractions — keep it simple

## Testing

- Framework: pytest + pytest-asyncio (asyncio_mode = "auto")
- `tests/` mirrors `src/` structure (e.g., `tests/events/` tests `src/events/`)
- Standalone (no NATS/Ollama): `tests/events/test_types.py`, `tests/rewards/`, `tests/training/`
- Requires NATS: `tests/events/test_bus.py`, `tests/test_integration.py`
- Mock strategy: scorer/evaluator tests mock Ollama client; integration tests use EchoWorker + mock scorer
- Run standalone: `pytest tests/events/test_types.py tests/rewards/ tests/training/ -v`
- Run standalone (skip slow torch tests): `pytest tests/training/ -v -k "not slow"`
- Run all: `pytest tests/ -v` (with NATS running)

## Commit Format

Conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code restructuring
- `test:` adding/updating tests
- `chore:` maintenance tasks

## Current Phase

**Phase 4 complete** — GRPO trainer integration. Standalone GRPO trainer (torch + transformers + peft) with LoRA fine-tuning, TrainingLoop orchestrator, MockTrainer fallback. Trains + saves LoRA checkpoints, publishes ModelUpdateEvent (workers ignore until Phase 5 vLLM migration).

**Phase 5 (next):** vLLM migration, weight hot-swap in workers, OpenRLHF integration, Semantic Router as inference gateway.

## Key Documents

- [PLAN.md](./PLAN.md) — Full technical research & architecture bible (papers, analysis, decisions)
- [LEARNING.md](./LEARNING.md) — Mistake/lesson tracking for autonomous decisions
- [RESEARCH-EXPERIMENT.md](./RESEARCH-EXPERIMENT.md) — Phase experiment records and findings
