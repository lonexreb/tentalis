# agentic-employees

Managers managing AI agents with appraisal-style feedback loops. Agents learn continuously from manager feedback using RLHF/GRPO — like performance reviews that actually improve performance.

## Architecture

Event-Driven Architecture (EDA):
- **Control Plane**: OpenClaw (identity, memory, channels, UI)
- **Event Broker**: NATS (pub/sub for agent coordination)
- **Training Plane**: Standalone GRPO (lightweight) + OpenRLHF (production GPU)
- **Inference**: InferenceClient protocol — Ollama (dev) or OpenAI-compatible (vLLM/Semantic Router)

## Language

**Python only.** Requires Python >= 3.10. All components — RL training, agent logic, NATS clients, PRM evaluator — are Python.

## Current Dependencies (pyproject.toml)

- [nats-py](https://github.com/nats-io/nats.py) — NATS client for event bus
- [pydantic](https://docs.pydantic.dev/) — event type serialization (v2)
- [ollama](https://github.com/ollama/ollama-python) — LLM inference via Ollama (async client)

Dev: pytest, pytest-asyncio, pytest-aiohttp, ruff

Optional extras:
- `pip install -e ".[training]"` — torch, transformers, peft
- `pip install -e ".[inference]"` — openai, httpx (for OpenAI-compatible servers / vLLM)
- `pip install -e ".[bridge]"` — aiohttp (Bridge HTTP API for OpenClaw integration)
- `pip install -e ".[vllm]"` — vLLM (GPU inference server)
- `pip install -e ".[openrlhf]"` — OpenRLHF (production GRPO training)
- `pip install -e ".[intercept]"` — fastapi, uvicorn, httpx (Intercept Proxy)

## Directory Structure

```
src/
├── __main__.py    # Demo entry point (python -m src) — uses InferenceClient factory
├── config.py      # Frozen dataclass with env var defaults (NATS, inference, training)
├── manager/
│   └── manager.py # Manager agent (assign tasks, wait for results, publish feedback)
├── workers/
│   ├── base.py              # BaseWorker ABC (subscribe, handle, process, model update hook, environment_type)
│   ├── echo_worker.py       # EchoWorker — echoes prompt back with PRM steps (testing)
│   ├── llm_worker.py        # LLMWorker — LLM inference via InferenceClient with step parsing
│   ├── terminal_worker.py   # TerminalWorker — Docker-based bash execution
│   ├── swe_worker.py        # SWEWorker — GitHub issue → plan/implement/test pipeline
│   └── gui_worker.py        # GUIWorker — screenshot + action pairs for GUI automation
├── events/
│   ├── types.py   # Pydantic v2 event models (TaskEvent, ResultEvent, FeedbackEvent, etc.)
│   ├── topics.py  # Topic constants and helpers
│   └── bus.py     # EventBus wrapping nats-py (connect, publish, subscribe, drain)
├── inference/
│   ├── client.py            # InferenceClient protocol + OllamaInferenceClient + OpenAIInferenceClient
│   ├── vllm_lora.py         # VLLMLoRAManager — dynamic LoRA hot-swap via vLLM admin endpoints
│   └── adapter_registry.py  # PerWorkerAdapterRegistry — per-worker LoRA adapter management
├── rewards/
│   ├── scorer.py            # StepScorer protocol + LLMJudgeScorer (LLM-as-judge PRM)
│   ├── prompts.py           # STEP_JUDGE_PROMPT template for step-level evaluation
│   ├── prm_evaluator.py     # PRMEvaluator — subscribes to results, scores steps, publishes rollouts
│   └── combined_scorer.py   # CombinedScorer — multi-scorer composition with per-environment weights
├── training/
│   ├── bridge.py              # RolloutBuffer + NATSTrainingBridge (batch rollouts for RL trainer)
│   ├── grpo.py                # GRPO math: advantages, clipped_surrogate, asymmetric_clip, combined_loss, kl_penalty
│   ├── trainer.py             # Trainer protocol, TrainStepResult, MockTrainer, GRPOTrainer (LoRA)
│   ├── combined_trainer.py    # CombinedTrainer — merged RL + OPD distillation loss
│   ├── meta_trainer.py        # ManagerMetaTrainer — outer-loop RL for manager feedback quality
│   ├── loop.py                # TrainingLoop orchestrator (bridge → trainer → ModelUpdateEvent, combined support)
│   └── openrlhf_launcher.py   # OpenRLHFLauncher — subprocess launcher for production GRPO training
├── intercept/
│   ├── __main__.py    # Entrypoint: python -m src.intercept
│   └── proxy.py       # InterceptProxy — FastAPI proxy logging SessionEvents to NATS
├── opd/
│   ├── hint_extractor.py    # HintExtractor — feedback → OPD hint + teacher logprobs
│   └── rollout_builder.py   # CombinedRolloutBuilder — joins RL + OPD by task_id with timeout
├── bridge/
│   ├── __main__.py    # Entrypoint: python -m src.bridge
│   ├── service.py     # BridgeService — connects HTTP API to NATS event bus
│   └── http_api.py    # HTTP endpoints for OpenClaw agents (assign, result, feedback, status, health)
├── services/
│   ├── __main__.py    # Entrypoint: python -m src.services.training
│   └── training.py    # Standalone training service (PRM Evaluator + Training Loop)
config/
└── openclaw/
    ├── AGENTS.md              # Agent registry (manager-01 + worker-01)
    ├── manager/
    │   ├── SOUL.md            # Manager behavior: decompose, assign, evaluate, score
    │   └── IDENTITY.md        # Manager identity (name, role, bio)
    ├── worker/
    │   ├── SOUL.md            # Worker behavior: step-by-step solving with <step>/<answer>
    │   └── IDENTITY.md        # Worker identity
    └── skills/
        ├── assign-task/SKILL.md     # exec: curl POST bridge:8100/tasks/assign
        ├── submit-result/SKILL.md   # exec: curl POST bridge:8100/tasks/result
        └── submit-feedback/SKILL.md # exec: curl POST bridge:8100/feedback
tests/
├── bridge/
│   ├── test_http_api.py # Bridge HTTP endpoint unit tests (mocked NATS)
│   └── test_service.py  # Bridge integration test (requires NATS)
├── events/
│   ├── test_types.py  # Serialization roundtrip tests (standalone)
│   └── test_bus.py    # EventBus pub/sub tests (requires NATS)
├── inference/
│   ├── test_client.py       # InferenceClient protocol + adapter tests (mocked)
│   └── test_vllm_lora.py    # VLLMLoRAManager tests (mocked httpx)
├── rewards/
│   ├── test_scorer.py        # LLMJudgeScorer tests (mocked InferenceClient)
│   └── test_prm_evaluator.py # PRMEvaluator tests (mocked scorer)
├── training/
│   ├── test_bridge.py        # RolloutBuffer unit tests
│   ├── test_grpo.py          # GRPO advantage math + torch loss/KL tests
│   └── test_trainer.py       # MockTrainer + GRPOTrainer protocol/integration tests
├── workers/
│   └── test_model_reload.py  # Worker model update subscription + reload tests
└── test_integration.py # Full manager→worker→PRM→rollout loop (requires NATS)
docker-compose.yml     # 5 services: NATS, Ollama, OpenClaw, Bridge, Training
Dockerfile             # Python 3.10 base for training service
Dockerfile.bridge      # Python 3.10 base for bridge service
scripts/
└── demo.sh            # One-command Docker Compose demo startup
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
- Standalone (no NATS/Ollama): `tests/events/test_types.py`, `tests/rewards/`, `tests/training/`, `tests/inference/`, `tests/workers/`, `tests/bridge/test_http_api.py`, `tests/opd/`, `tests/intercept/`
- Requires NATS: `tests/events/test_bus.py`, `tests/test_integration.py`, `tests/bridge/test_service.py`
- Requires optional deps: `tests/intercept/` (fastapi), `tests/bridge/test_http_api.py` (aiohttp)
- Mock strategy: scorer/evaluator tests mock InferenceClient; bridge tests mock EventBus; OPD tests mock bus+client; integration tests use EchoWorker + mock scorer
- Run standalone: `pytest tests/ -v --ignore=tests/bridge --ignore=tests/intercept` (100 pass, 14 skip without torch/NATS)
- Run standalone (skip slow torch tests): `pytest tests/training/ -v -k "not slow"`
- Run all: `pytest tests/ -v` (with NATS running + optional deps)

## Configuration (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| INFERENCE_BACKEND | ollama | `"ollama"` or `"openai"` (for vLLM/Semantic Router) |
| INFERENCE_BASE_URL | (auto) | Base URL for inference server |
| INFERENCE_API_KEY | (empty) | API key for inference server |
| VLLM_LORA_NAME | default | LoRA adapter name for vLLM |
| TRAINER_BACKEND | standalone | `"standalone"` (GRPOTrainer) or `"openrlhf"` (OpenRLHFLauncher) |
| BRIDGE_PORT | 8100 | Bridge HTTP API port |
| OPENCLAW_GATEWAY_URL | ws://localhost:18789 | OpenClaw gateway WebSocket URL |
| INTERCEPT_ENABLED | false | Enable intercept proxy |
| INTERCEPT_PORT | 8200 | Intercept proxy port |
| INTERCEPT_BACKEND_URL | http://localhost:11434 | Backend URL for intercept proxy |
| OPD_TEACHER_MODEL | qwen2.5:1.5b | Teacher model for OPD hint extraction |
| OPD_JOIN_TIMEOUT | 30.0 | Timeout (seconds) for joining RL + OPD rollouts |
| OPD_WEIGHT | 0.3 | Weight for OPD loss in combined training |
| RL_WEIGHT | 0.7 | Weight for RL loss in combined training |
| TRAINING_CLIP_EPSILON_HIGH | 0.28 | Asymmetric high clip bound |
| META_RL_ENABLED | false | Enable manager meta-RL training |
| META_RL_WINDOW_SIZE | 200 | Sliding window for meta-RL score tracking |
| META_RL_MIN_FEEDBACK | 200 | Min feedback events before meta-training |

## Commit Format

Conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code restructuring
- `test:` adding/updating tests
- `chore:` maintenance tasks

## Current Phase

**Phase 7 complete** — ADHR (Appraisal-Driven Hierarchical RL) with OpenClaw-RL integration.

- Intercept Proxy (`src/intercept/`) — FastAPI proxy logging SessionEvents to NATS
- OPD Hint Extraction (`src/opd/hint_extractor.py`) — manager textual feedback → corrective hints + teacher logprobs
- Combined Rollout Builder (`src/opd/rollout_builder.py`) — joins RL + OPD signals by task_id with timeout
- Combined Trainer (`src/training/combined_trainer.py`) — merged RL + OPD loss with asymmetric clipping
- Combined Scorer (`src/rewards/combined_scorer.py`) — multi-scorer composition with per-environment weight profiles
- Manager Meta-RL (`src/training/meta_trainer.py`) — outer-loop RL training for manager feedback quality
- Per-Worker Adapter Registry (`src/inference/adapter_registry.py`) — worker_id → LoRA adapter management
- Multi-Environment Workers — TerminalWorker (Docker bash), SWEWorker (issue→patch), GUIWorker (screenshot+action)
- `target_worker_id` on ModelUpdateEvent for per-worker model targeting
- `environment_type` on BaseWorker for environment-specific scoring
- Asymmetric clipped surrogate loss + combined loss in `grpo.py`
- Docker Compose updated with intercept-proxy service (6 services total)
- 106+ tests (100 pass standalone, 14 skip without torch)

Previous phases:
- Phase 6: OpenClaw integration, Bridge Service, Docker Compose demo
- Phase 5: InferenceClient protocol, weight hot-swap, OpenRLHF, Semantic Router readiness
- Phase 4: Standalone GRPO trainer, LoRA fine-tuning, TrainingLoop
- Phase 3: LLM workers (Ollama), PRM scoring (LLM-as-judge), training bridge
- Phase 2: Event loop, Manager/Worker agents, EchoWorker
- Phase 1: Scaffolding, docs, git

**Phase 8 (next):** Trained PRM model, DAPO graduation, multi-model routing in Semantic Router, HaluGate scorer.

## Key Documents

- [PLAN.md](./PLAN.md) — Full technical research & architecture bible (papers, analysis, decisions)
- [LEARNING.md](./LEARNING.md) — Mistake/lesson tracking for autonomous decisions
- [RESEARCH-EXPERIMENT.md](./RESEARCH-EXPERIMENT.md) — Phase experiment records and findings
