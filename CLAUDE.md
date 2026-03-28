# Tentalis

ADHR meta-RL framework built on top of OpenRLHF/OpenClaw-RL. Agents learn continuously from manager feedback — like performance reviews that actually improve performance.

## Architecture

Two-Layer Architecture (control plane / data plane split):
- **Orchestration Layer** (NATS): Agent coordination, task routing, manager feedback, meta-RL signals
- **Training Layer** (OpenRLHF): Production GRPO/DAPO training via Ray + vLLM + DeepSpeed
- **Control Plane**: OpenClaw (identity, memory, channels, UI)
- **Inference**: InferenceClient protocol — Ollama (dev) or OpenAI-compatible (vLLM/Semantic Router)
- **CLI**: `tentalis init|train|serve|status|experiment` (Typer + Rich)

## Language

**Python only.** Requires Python >= 3.10. All components — RL training, agent logic, NATS clients, PRM evaluator — are Python.

## Current Dependencies (pyproject.toml)

- [nats-py](https://github.com/nats-io/nats.py) — NATS client for event bus
- [pydantic](https://docs.pydantic.dev/) — event type serialization (v2)
- [ollama](https://github.com/ollama/ollama-python) — LLM inference via Ollama (async client)
- [typer](https://typer.tiangolo.com) — CLI framework
- [rich](https://rich.readthedocs.io) — CLI output formatting

Dev: pytest, pytest-asyncio, pytest-aiohttp, ruff

Optional extras:
- `pip install -e ".[training]"` — torch, transformers, peft
- `pip install -e ".[inference]"` — openai, httpx (for OpenAI-compatible servers / vLLM)
- `pip install -e ".[bridge]"` — aiohttp (Bridge HTTP API for OpenClaw integration)
- `pip install -e ".[vllm]"` — vLLM (GPU inference server)
- `pip install -e ".[openrlhf]"` — OpenRLHF (production GRPO training)
- `pip install -e ".[intercept]"` — fastapi, uvicorn, httpx (Intercept Proxy)
- `pip install -e ".[skills]"` — sentence-transformers (SkillRL embedding retrieval)
- `pip install -e ".[tinker]"` — Tinker SDK (cloud-managed RL training)
- `pip install -e ".[alignment]"` — streamlit (alignment experiment dashboard)

## Directory Structure

```
src/
├── __main__.py    # Demo entry point (python -m src) — uses InferenceClient factory
├── cli.py         # CLI entry point (tentalis init|train|serve|status)
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
│   ├── scheduler.py           # TrainingScheduler — time-window gated training (buffers outside hours)
│   ├── tinker_backend.py      # TinkerBackend — Trainer protocol adapter for Tinker cloud training
│   ├── openrlhf_launcher.py   # OpenRLHFLauncher — subprocess launcher (legacy, kept for direct CLI usage)
│   └── openrlhf_backend.py    # OpenRLHFBackend — Trainer protocol adapter for Ray+vLLM+DeepSpeed training
├── intercept/
│   ├── __main__.py          # Entrypoint: python -m src.intercept
│   ├── proxy.py             # InterceptProxy — session-stateful FastAPI proxy with skill injection
│   └── session_manager.py   # SessionManager — tracks active sessions with conversation history
├── opd/
│   ├── hint_extractor.py    # HintExtractor — feedback → OPD hint + teacher logprobs
│   └── rollout_builder.py   # CombinedRolloutBuilder — joins RL + OPD by task_id with timeout
├── bridge/
│   ├── __main__.py    # Entrypoint: python -m src.bridge
│   ├── service.py     # BridgeService — connects HTTP API to NATS event bus
│   └── http_api.py    # HTTP endpoints for OpenClaw agents (assign, result, feedback, status, health)
├── alignment/
│   ├── __init__.py
│   ├── scenarios.py         # AlignmentScenario dataclass + 4 scenario sets (~40 scenarios)
│   ├── behavioral_eval.py   # BehavioralEvaluator protocol, PatternBased + LLMJudge evaluators, harness
│   ├── hackable_scorer.py   # HackableScorer (StepScorer) + RewardHackingDetector
│   ├── misaligned_worker.py # MisalignedWorker (BaseWorker) — keyword stuffing/confidence/shortcut
│   ├── collusion_detector.py # CollusionDetector — Pearson correlation + Jaccard similarity
│   ├── audit_logger.py      # AuditLogger — subscribe_raw to all topics, write JSONL
│   ├── runner.py            # ExperimentRunner — orchestrates all 6 experiments
│   └── dashboard/
│       ├── __init__.py
│       └── app.py           # Streamlit dashboard for results + audit + constitution editor
├── skills/
│   ├── __init__.py
│   ├── store.py             # SkillStore — SQLite-backed CRUD with embedding persistence
│   ├── retriever.py         # SkillRetriever — embedding-based semantic search (SentenceTransformer)
│   └── evolver.py           # SkillEvolver — subscribes to feedback, extracts skills via LLM
├── setup_wizard.py          # Interactive Rich setup wizard for first-time config
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
│   ├── test_trainer.py       # MockTrainer + GRPOTrainer protocol/integration tests
│   ├── test_combined_trainer.py  # CombinedTrainer RL+OPD tests
│   ├── test_meta_trainer.py      # ManagerMetaTrainer tests
│   ├── test_tinker_backend.py    # TinkerBackend protocol conformance + mocked SDK tests
│   └── test_scheduler.py         # TrainingScheduler time-window + buffering tests
├── inference/
│   ├── test_client.py         # InferenceClient protocol + adapter tests (mocked)
│   ├── test_vllm_lora.py      # VLLMLoRAManager tests (mocked httpx)
│   └── test_adapter_registry.py  # PerWorkerAdapterRegistry tests
├── rewards/
│   ├── test_scorer.py         # LLMJudgeScorer tests (mocked InferenceClient)
│   ├── test_prm_evaluator.py  # PRMEvaluator tests (mocked scorer)
│   └── test_combined_scorer.py  # CombinedScorer per-environment weight tests
├── opd/
│   ├── test_hint_extractor.py   # HintExtractor tests (mocked client + bus)
│   └── test_rollout_builder.py  # CombinedRolloutBuilder join + timeout tests
├── intercept/
│   └── test_proxy.py          # Intercept proxy tests (mocked backend, requires fastapi)
├── skills/
│   ├── test_store.py          # SkillStore CRUD + SQLite tests
│   ├── test_retriever.py      # SkillRetriever cosine similarity + embedding retrieval tests
│   └── test_evolver.py        # SkillEvolver feedback → skill extraction tests
├── alignment/
│   ├── test_scenarios.py        # Scenario validation tests (fields, uniqueness, counts)
│   ├── test_behavioral_eval.py  # PatternBasedEvaluator, LLMJudgeEvaluator, harness tests
│   ├── test_hackable_scorer.py  # HackableScorer + RewardHackingDetector tests
│   ├── test_misaligned_worker.py # MisalignedWorker strategy tests
│   ├── test_collusion_detector.py # Pearson/Jaccard/CollusionDetector tests
│   ├── test_audit_logger.py     # AuditLogger JSONL + event detection tests
│   └── test_runner.py           # ExperimentRunner integration tests (mock mode)
├── workers/
│   ├── test_model_reload.py   # Worker model update subscription + reload tests
│   └── test_multi_env.py      # Terminal/SWE/GUI worker tests + target_worker_id filtering
└── test_integration.py # Full manager→worker→PRM→rollout loop (requires NATS)
docker-compose.yml     # 6 services: NATS, Ollama, OpenClaw, Bridge, Intercept Proxy, Training
Dockerfile             # Python 3.10 base for training service
Dockerfile.bridge      # Python 3.10 base for bridge service
Dockerfile.intercept   # Python 3.10 base for intercept proxy
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
- Standalone (no NATS/Ollama): `tests/events/test_types.py`, `tests/rewards/`, `tests/training/`, `tests/inference/`, `tests/workers/`, `tests/bridge/test_http_api.py`, `tests/opd/`, `tests/intercept/`, `tests/skills/`, `tests/alignment/`
- Requires NATS: `tests/events/test_bus.py`, `tests/test_integration.py`, `tests/bridge/test_service.py`
- Requires optional deps: `tests/intercept/` (fastapi), `tests/bridge/test_http_api.py` (aiohttp)
- Mock strategy: scorer/evaluator tests mock InferenceClient; bridge tests mock EventBus; OPD tests mock bus+client; integration tests use EchoWorker + mock scorer
- Run standalone: `pytest tests/ -v --ignore=tests/bridge --ignore=tests/intercept` (241 pass, 22 skip without torch/NATS)
- Run standalone (skip slow torch tests): `pytest tests/training/ -v -k "not slow"`
- Run all: `pytest tests/ -v` (with NATS running + optional deps)

## Configuration (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| INFERENCE_BACKEND | ollama | `"ollama"` or `"openai"` (for vLLM/Semantic Router) |
| INFERENCE_BASE_URL | (auto) | Base URL for inference server |
| INFERENCE_API_KEY | (empty) | API key for inference server |
| VLLM_LORA_NAME | default | LoRA adapter name for vLLM |
| TRAINER_BACKEND | standalone | `"standalone"` (GRPOTrainer) or `"openrlhf"` (OpenRLHFBackend) |
| BRIDGE_PORT | 8100 | Bridge HTTP API port |
| OPENCLAW_GATEWAY_URL | ws://localhost:18789 | OpenClaw gateway WebSocket URL |
| INTERCEPT_ENABLED | false | Enable intercept proxy |
| INTERCEPT_PORT | 8200 | Intercept proxy port |
| INTERCEPT_BACKEND_URL | http://localhost:11434 | Backend URL for intercept proxy |
| OPD_MODE | lightweight | `"lightweight"` (LLM hints) or `"openclaw"` (vLLM logprobs) |
| OPD_TEACHER_MODEL | qwen2.5:1.5b | Teacher model for OPD hint extraction |
| OPD_JOIN_TIMEOUT | 30.0 | Timeout (seconds) for joining RL + OPD rollouts |
| OPD_WEIGHT | 0.3 | Weight for OPD loss in combined training |
| RL_WEIGHT | 0.7 | Weight for RL loss in combined training |
| TRAINING_CLIP_EPSILON_HIGH | 0.28 | Asymmetric high clip bound |
| META_RL_ENABLED | false | Enable manager meta-RL training |
| META_RL_WINDOW_SIZE | 200 | Sliding window for meta-RL score tracking |
| META_RL_MIN_FEEDBACK | 200 | Min feedback events before meta-training |
| PRM_NUM_VOTES | 3 | Number of parallel LLM judge evaluations (majority voting) |
| SKILLS_ENABLED | false | Enable SkillRL skill injection |
| SKILLS_DIR | skills_data | Directory for skill SQLite database |
| SKILL_EVOLUTION_THRESHOLD | 0.4 | Score threshold below which skills are extracted |
| SKILL_RETRIEVAL_TOP_K | 3 | Number of skills to inject per task |
| TINKER_API_KEY | (empty) | API key for Tinker cloud training |
| TINKER_BASE_URL | https://api.tinker.thinkingmachines.ai | Tinker API base URL |
| TRAINING_SCHEDULE_ENABLED | false | Enable time-window gated training |
| TRAINING_SCHEDULE_HOURS | 02:00-06:00 | UTC training window (HH:MM-HH:MM) |
| ALIGNMENT_ENABLED | false | Enable alignment experiment infrastructure |
| ALIGNMENT_RESULTS_DIR | alignment_results | Directory for experiment result JSON files |
| ALIGNMENT_AUDIT_ALL | false | Enable full NATS event audit logging |

## Commit Format

Conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code restructuring
- `test:` adding/updating tests
- `chore:` maintenance tasks

## Current Phase

**Phase 8 in progress** — Adopt + Extend architecture reassessment.

Phase 8 additions:
- CLI entry point (`src/cli.py`) — `tentalis init|train|serve|status` via Typer + Rich
- OpenRLHF training backend (`src/training/openrlhf_backend.py`) — Trainer protocol adapter for Ray+vLLM+DeepSpeed
- OpenClaw-RL OPD mode (`src/opd/hint_extractor.py`) — per-token logprob extraction from vLLM teacher models
- Backend selection in `src/services/training.py` — standalone vs openrlhf
- `OPD_MODE` config var — "lightweight" or "openclaw"
- Docker Compose with commented OpenRLHF GPU trainer service
- Honest README positioning — two-layer architecture, novel vs adopted components

Phase 7 (complete):
- Intercept Proxy, OPD, CombinedScorer, Meta-RL, Adapter Registry, Multi-env Workers

Previous phases:
- Phase 6: OpenClaw integration, Bridge Service, Docker Compose demo
- Phase 5: InferenceClient protocol, weight hot-swap, OpenRLHF, Semantic Router readiness
- Phase 4: Standalone GRPO trainer, LoRA fine-tuning, TrainingLoop
- Phase 3: LLM workers (Ollama), PRM scoring (LLM-as-judge), training bridge
- Phase 2: Event loop, Manager/Worker agents, EchoWorker
- Phase 1: Scaffolding, docs, git

Phase 9a (complete — MetaClaw adoption):
- Majority Voting PRM (`src/rewards/scorer.py`) — parallel LLM judge evals with median aggregation
- SkillRL (`src/skills/`) — skill store, embedding retriever, evolver from feedback, skill injection in workers/proxy
- Tinker training backend (`src/training/tinker_backend.py`) — cloud-managed RL via Tinker SDK
- Interactive setup wizard (`src/setup_wizard.py`) — Rich multi-step config wizard
- Session-stateful intercept proxy (`src/intercept/session_manager.py`) — session tracking + skill injection
- Training scheduler (`src/training/scheduler.py`) — time-window gated training

Phase 9b (complete — Alignment experiments):
- Alignment scenario library (`src/alignment/scenarios.py`) — 40 scenarios across 4 categories
- Behavioral eval harness (`src/alignment/behavioral_eval.py`) — PatternBased + LLMJudge evaluators
- Hackable scorer (`src/alignment/hackable_scorer.py`) — deliberately weak scorer + divergence detector
- Misaligned worker (`src/alignment/misaligned_worker.py`) — keyword stuffing, confidence inflation, shortcut
- Collusion detector (`src/alignment/collusion_detector.py`) — Pearson + Jaccard cross-worker analysis
- Audit logger (`src/alignment/audit_logger.py`) — full NATS event capture to JSONL
- Experiment runner (`src/alignment/runner.py`) — 6 experiments (mock-mode, standalone)
- Streamlit dashboard (`src/alignment/dashboard/app.py`) — results viewer + constitution editor
- CLI `experiment` subcommand — `tentalis experiment run|results`
- `AlignmentEvalEvent` + `AuditLogEvent` event types, `subscribe_raw` on EventBus

**Phase 9c (next):** Trained PRM model, DAPO graduation, HaluGate scorer, CISPO contrastive loss, benchmarks.

## Key Documents

- [EXPERIMENT.md](./EXPERIMENT.md) — Alignment experiment tracking (6 experiments, hypotheses, metrics)
- [PLAN.md](./PLAN.md) — Full technical research & architecture bible (papers, analysis, decisions)
- [LEARNING.md](./LEARNING.md) — Mistake/lesson tracking for autonomous decisions
- [RESEARCH-EXPERIMENT.md](./RESEARCH-EXPERIMENT.md) — Phase experiment records and findings
