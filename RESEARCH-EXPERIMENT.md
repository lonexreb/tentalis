# Research & Experiment Records

Living document tracking experiments, findings, and technical decisions across phases. Each phase records what was tried, what worked, what didn't, and quantitative results where available.

---

## Phase 1: Scaffolding

**Date:** 2026-03-09

**Scope:** Project setup, docs, git, GitHub.

**Outcome:** Established CLAUDE.md + PLAN.md + LEARNING.md documentation pattern. Python >=3.10, pyproject.toml with setuptools backend.

**Finding:** Dev machine runs Python 3.10.0 — initial pyproject.toml incorrectly set `>=3.11`. Caught during first install. Logged in LEARNING.md.

---

## Phase 2: Event Loop

**Date:** 2026-03-09

**Scope:** NATS event bus, Manager/Worker agents, EchoWorker, pub/sub task assignment.

**Outcome:** Manager publishes TaskEvent, EchoWorker subscribes and echoes, Manager receives ResultEvent and publishes FeedbackEvent. Full loop verified with NATS.

**Key files created:** `src/events/`, `src/manager/`, `src/workers/`, `src/config.py`, `src/__main__.py`

---

## Phase 3: LLM Workers, PRM Scoring & Training Bridge

**Date:** 2026-03-10

**Scope:** Real LLM inference (Ollama), step-level scoring (LLM-as-judge PRM), training bridge (RolloutBuffer + JSONL export).

### Experiments

#### 3.1 LLM Worker (Ollama)

- **Hypothesis:** Ollama AsyncClient can provide structured step-by-step output suitable for PRM scoring.
- **Approach:** System prompt with `<step>...</step>` and `<answer>...</answer>` markers. Regex parsing in LLMWorker.
- **Result:** Works reliably with qwen2.5:1.5b. Step parsing robust enough for prototype. ~5 req/s on CPU.
- **Graduation path:** Ollama → vLLM + LoRA (Phase 5).

#### 3.2 PRM Evaluator (LLM-as-Judge)

- **Hypothesis:** An LLM can score each reasoning step on progress + correctness (0-1) via JSON output.
- **Approach:** StepScorer protocol with LLMJudgeScorer implementation. Evaluator subscribes to results, scores steps, publishes TrainingRolloutEvent.
- **Result:** Scores are noisy but directionally correct. Good enough to bootstrap training data for future trained PRM.
- **Graduation trigger:** When judge latency bottlenecks training throughput, or when 10K+ scored trajectories accumulated.

#### 3.3 Training Bridge

- **Hypothesis:** RolloutBuffer can batch async scored rollouts for synchronous GRPO consumption.
- **Approach:** Group rollouts by prompt (group_size), emit batch when batch_size reached.
- **Result:** Clean separation — bridge handles async→batch conversion, trainer protocol consumes batches. JSONL export works for offline analysis.

### Findings

- `ResultEvent` needed a `prompt` field added — evaluator needs original prompt for context. Small duplication but keeps evaluator stateless.
- Group-based buffering is essential for GRPO — advantage calculation requires multiple responses to the same prompt.
- LLM-as-judge scores have high variance but low bias — averaging across group mitigates noise.

---

## Phase 4: GRPO Trainer Integration

**Date:** 2026-03-12

**Scope:** Standalone GRPO trainer with LoRA fine-tuning, TrainingLoop orchestrator, MockTrainer fallback, ModelUpdateEvent publishing.

### Key Decision: Standalone GRPO over OpenRLHF

- **Problem:** OpenRLHF requires torch+CUDA+ray+deepspeed+vllm — cannot install or test on CPU-only dev machine (macOS, Python 3.10).
- **Decision:** Build standalone GRPO trainer (~150 lines) using torch + transformers + peft. Same Trainer protocol — OpenRLHF becomes a drop-in swap in Phase 5.
- **Validation:** CPU-testable with distilgpt2 (~82M params). Slow (~minutes per batch) but functionally validates the pipeline.

### Experiments

#### 4.1 GRPO Math (`src/training/grpo.py`)

- **Functions:** `compute_group_advantages()`, `clipped_surrogate_loss()`, `kl_penalty()`
- **Advantage formula:** `(reward_i - mean) / std` within a prompt group
- **Finding:** Floating point precision matters — `[0.7, 0.7, 0.7]` produces variance ~1e-32 (not exactly 0.0) due to IEEE 754 representation. Fix: tolerance check `variance < 1e-12` instead of `== 0.0`.
- **Test results:** 7 list-based advantage tests pass (no torch). 4 tensor-based loss/KL tests pass with torch.

#### 4.2 Trainer Protocol + MockTrainer

- **Pattern:** Follows `StepScorer` protocol pattern from `src/rewards/scorer.py` — `@runtime_checkable` Protocol.
- **MockTrainer:** Zero-dep trainer for testing. Logs batch info, returns fake TrainStepResult with decreasing loss (1/step_count).
- **TrainStepResult:** Pydantic model with loss, mean_advantage, std_advantage, checkpoint_path, step_count.
- **Test results:** Protocol conformance verified via `isinstance(MockTrainer(), Trainer)`. 6 tests pass.

#### 4.3 GRPOTrainer (LoRA + distilgpt2)

- **Architecture:** Lazy-loaded model (import inside `_ensure_loaded()`, not at module top). Base model frozen as reference, LoRA adapter on trainable copy.
- **LoRA config:** rank=8, alpha=16, target_modules=["c_attn"], dropout=0.05, task_type=CAUSAL_LM.
- **Training loop:** Group by prompt → compute advantages → tokenize → forward pass (policy + reference) → clipped surrogate loss + KL penalty → backward → AdamW step → periodic LoRA checkpoint.
- **Checkpointing:** Saves LoRA adapter + tokenizer to `checkpoint_dir/v{step:04d}/` every N steps.
- **Finding:** distilgpt2 loads in ~2s on CPU. Full train_step with batch of 4 rollouts takes ~10-30s on CPU. Acceptable for functional validation.

#### 4.4 TrainingLoop Orchestrator

- **Wiring:** Creates NATSTrainingBridge internally, registers `_on_batch` callback.
- **Flow:** Bridge receives rollouts → buffer groups and batches → callback calls `trainer.train_step()` → publishes `ModelUpdateEvent` if checkpoint saved.
- **ModelUpdateEvent:** Published to `model.updates` topic with version, checkpoint_path, metrics (loss, advantages). Workers ignore this in Phase 4 — Phase 5 enables hot-swap.

#### 4.5 Optional Dependencies

- **Approach:** `pip install -e ".[training]"` adds torch>=2.0, transformers>=4.40, peft>=0.11.
- **Base install** (`pip install -e ".[dev]"`) works without torch — 32 tests pass, torch-dependent tests skip cleanly.
- **Finding:** pip 21.2.3 (bundled with Python 3.10) doesn't support editable installs from pyproject.toml without setup.py. Fix: upgrade pip+setuptools first.

#### 4.6 Entry Point Fallback

- **Pattern:** Try importing GRPOTrainer → if ImportError (no torch), fall back to MockTrainer. Same pattern as LLMWorker → EchoWorker fallback.
- **Config fields added:** training_model, training_lr, training_clip_epsilon, training_kl_beta, training_checkpoint_dir, training_lora_rank, training_batch_size, training_group_size, training_device.

### Test Summary

| Test Suite | Tests | Deps | Speed | Status |
|------------|-------|------|-------|--------|
| `test_grpo.py` (list-based) | 7 | None | <1s | PASS |
| `test_grpo.py` (tensor-based) | 4 | torch | <1s | SKIP (no torch) / PASS (with torch) |
| `test_trainer.py` (MockTrainer) | 6 | None | <1s | PASS |
| `test_trainer.py` (GRPOTrainer) | 3 | torch, transformers, peft | ~30s | SKIP (no torch) / PASS (with torch) |
| `test_bridge.py` (existing) | 4 | None | <1s | PASS |
| **Total standalone** | **32 pass, 5 skip** | | **<1s** | |

### Files

| Action | Path |
|--------|------|
| CREATE | `src/training/grpo.py` — GRPO math utilities |
| CREATE | `src/training/trainer.py` — Trainer protocol, MockTrainer, GRPOTrainer |
| CREATE | `src/training/loop.py` — TrainingLoop orchestrator |
| CREATE | `tests/training/test_grpo.py` — Advantage math + loss/KL tests |
| CREATE | `tests/training/test_trainer.py` — Protocol conformance + integration tests |
| MODIFY | `pyproject.toml` — training optional deps, slow marker |
| MODIFY | `src/config.py` — 8 training config fields |
| MODIFY | `src/training/__init__.py` — new exports |
| MODIFY | `src/__main__.py` — TrainingLoop wiring with fallback |
| MODIFY | `CLAUDE.md` — directory structure, phase status, testing docs |

### Deferred to Phase 5

- Weight hot-swap in workers (needs vLLM, Ollama has no native LoRA hot-swap)
- OpenRLHF integration (drop-in via Trainer protocol)
- Trained PRM model (still using LLM-as-judge)
- Multi-GPU / Ray / DeepSpeed
- Semantic Router as inference gateway

---

## Phase 5: Inference Abstraction, Weight Hot-Swap, OpenRLHF & Semantic Router

**Date:** 2026-03-12

**Scope:** Decouple inference from Ollama, make workers react to model updates, integrate OpenRLHF for production training, prepare Semantic Router as inference gateway.

### Key Decision: Clean Break from Ollama

- **Problem:** Both `LLMWorker` and `LLMJudgeScorer` were hardcoded to `ollama.AsyncClient` with Ollama-specific response extraction (`response["message"]["content"]`).
- **Decision:** Introduce `InferenceClient` protocol with `chat()` returning `str` directly. Adapters handle response extraction internally. Old `ollama_host` constructor params removed entirely (clean break, not backward compatible).
- **Validation:** Both Ollama and OpenAI adapters pass protocol conformance tests. Scorer and worker tests updated to mock `InferenceClient` instead of `ollama.AsyncClient`.

### Experiments

#### 5.1 InferenceClient Protocol (`src/inference/client.py`)

- **Design:** `@runtime_checkable` Protocol with single `chat()` method. Returns `str` — adapters handle response format differences internally.
- **Adapters:** `OllamaInferenceClient` (wraps `ollama.AsyncClient`, maps `json_mode=True` → `format="json"`) and `OpenAIInferenceClient` (wraps `openai.AsyncOpenAI`, maps `json_mode=True` → `response_format={"type":"json_object"}`).
- **Factory:** `create_client(backend, base_url, api_key)` — selects adapter based on config.
- **Finding:** Lazy import of `openai` inside `OpenAIInferenceClient.__init__` means base install doesn't require openai package. Same pattern as torch in GRPOTrainer.

#### 5.2 LLMWorker Refactor

- **Change:** Constructor takes `client: InferenceClient` (required) instead of `ollama_host: str`. `process()` calls `self._client.chat()` which returns `str` directly — no more `response["message"]["content"]` extraction.
- **Model reload:** Added `reload_model(event)` override — updates `self.model` and `self._active_version`. Version tracked in `ResultEvent.model_version`.

#### 5.3 LLMJudgeScorer Refactor

- **Change:** Constructor takes `client: InferenceClient` (required) instead of `ollama.AsyncClient`. `_judge_single_step()` uses `json_mode=True` instead of `format="json"`.
- **Finding:** Test simplification — mock returns `str` instead of nested dict `{"message": {"content": ...}}`.

#### 5.4 Weight Hot-Swap Infrastructure

- **BaseWorker:** Subscribes to `model.updates` topic in `start()`. `_handle_model_update()` calls `reload_model(event)` — default no-op, overridden by LLMWorker.
- **ResultEvent:** Added `model_version: str | None = None` field. `_handle_task()` sets it from `self._active_version`.
- **Finding:** EchoWorker inherits subscription but no-ops on model updates — clean separation.

#### 5.5 VLLMLoRAManager

- **Design:** Uses `httpx.AsyncClient` to call vLLM admin endpoints: `POST /v1/load_lora_adapter`, `POST /v1/unload_lora_adapter`, `GET /v1/models`.
- **Requires:** `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` env var on vLLM server.
- **Finding:** All tests pass with mocked httpx. Real integration requires GPU with vLLM server running.

#### 5.6 OpenRLHFLauncher

- **Design:** NOT a Trainer protocol implementation — separate class that exports rollouts to JSONL and launches `python -m openrlhf.cli.train_grpo_ray` as `asyncio.create_subprocess_exec`.
- **Finding:** OpenRLHF has no simple Python API — it's a Ray-distributed CLI tool. Subprocess approach avoids coupling to OpenRLHF internals.

### Test Summary

| Test Suite | Tests | Deps | Speed | Status |
|------------|-------|------|-------|--------|
| `test_client.py` (protocol + Ollama) | 6 | ollama | <1s | PASS |
| `test_client.py` (OpenAI) | 4 | openai | <1s | SKIP (no openai) / PASS (with openai) |
| `test_vllm_lora.py` | 5 | httpx | <1s | PASS |
| `test_model_reload.py` | 6 | None | <1s | PASS |
| `test_scorer.py` (updated) | 4 | None | <1s | PASS |
| `test_types.py` (model_version) | 2 new | None | <1s | PASS |
| **Total standalone** | **51 pass, 11 skip** | | **<1s** | |

### Files

| Action | Path |
|--------|------|
| CREATE | `src/inference/__init__.py` — exports InferenceClient, adapters, factory |
| CREATE | `src/inference/client.py` — InferenceClient protocol + adapters |
| CREATE | `src/inference/vllm_lora.py` — VLLMLoRAManager |
| CREATE | `src/training/openrlhf_launcher.py` — OpenRLHFLauncher |
| CREATE | `tests/inference/__init__.py` |
| CREATE | `tests/inference/test_client.py` — protocol + adapter tests |
| CREATE | `tests/inference/test_vllm_lora.py` — VLLMLoRAManager tests |
| CREATE | `tests/workers/__init__.py` |
| CREATE | `tests/workers/test_model_reload.py` — worker reload tests |
| MODIFY | `src/workers/base.py` — model.updates subscription + reload_model hook |
| MODIFY | `src/workers/llm_worker.py` — InferenceClient, _active_version, reload_model |
| MODIFY | `src/rewards/scorer.py` — InferenceClient, json_mode=True |
| MODIFY | `src/events/types.py` — model_version field on ResultEvent |
| MODIFY | `src/config.py` — inference_backend, inference_base_url, inference_api_key, vllm_lora_name, trainer_backend |
| MODIFY | `src/__main__.py` — create_client factory, shared client, VLLMLoRAManager wiring |
| MODIFY | `pyproject.toml` — inference, vllm, openrlhf optional dep groups |
| MODIFY | `tests/rewards/test_scorer.py` — mock InferenceClient instead of ollama.AsyncClient |
| MODIFY | `tests/events/test_types.py` — model_version roundtrip tests |
| MODIFY | `CLAUDE.md` — directory structure, deps, phase status, config table |

### Deferred to Phase 6

- Trained PRM model (still using LLM-as-judge — need 10K+ scored trajectories first)
- DAPO graduation (requires OpenRLHF working + tuning)
- Multi-model routing rules in Semantic Router (deploy SR first, then configure)
- HaluGate as complementary StepScorer implementation
- PicoClaw edge deployment

---

## Phase 6: OpenClaw Full Agent Runtime Integration + Docker Compose Demo

**Date:** 2026-03-13

**Scope:** Make OpenClaw the full agent runtime (Manager + Worker agents run inside OpenClaw), Bridge Service translates HTTP ←→ NATS, Docker Compose provides all infrastructure. Zero changes to PRM/Training pipeline.

### Key Decision: Python Bridge Service Pattern

- **Problem:** OpenClaw is Node.js (session-based agents). Existing pipeline is Python (async NATS pub/sub). The openclaw-sdk requires Python >=3.11 but project uses 3.10.
- **Decision:** Bridge Service — a Python aiohttp server that receives HTTP requests from OpenClaw agents (via exec/curl) and publishes identical Pydantic v2 JSON to the same NATS topics. Skip the SDK entirely.
- **Validation:** All 5 Pydantic event models reused directly. PRM Evaluator and Training Loop see identical events — zero code changes needed.

### Experiments

#### 6.1 Bridge Service (`src/bridge/`)

- **Design:** `BridgeService` connects to NATS, starts aiohttp server on `:8100`. 5 HTTP endpoints: `POST /tasks/assign`, `POST /tasks/result`, `POST /feedback`, `GET /tasks/{id}/status`, `GET /health`.
- **Reuse:** All Pydantic models from `src/events/types.py`, topic helpers from `src/events/topics.py`, EventBus from `src/events/bus.py`.
- **Result polling:** Bridge subscribes to `results.*` via NATS, caches results in-memory. `GET /tasks/{id}/status` long-polls (30s timeout) or returns cached result immediately.
- **Finding:** aiohttp test client (`pytest-aiohttp`) integrates cleanly with pytest-asyncio. All 10 unit tests pass without NATS.

#### 6.2 OpenClaw Agent Configuration (`config/openclaw/`)

- **Manager SOUL.md:** Decompose requests → call `assign-task` skill → poll for result → evaluate → call `submit-feedback` skill → report to user.
- **Worker SOUL.md:** Replicates `SYSTEM_PROMPT` from `src/workers/llm_worker.py` — step-by-step solving with `<step>`/`<answer>` format for PRM compatibility.
- **Skills:** Three SKILL.md files with `exec: curl` templates targeting `bridge:8100`. Parameters templated for OpenClaw substitution.
- **Finding:** SKILL.md `exec` format is straightforward — just a bash command with curl. JSON escaping handled by OpenClaw's template engine.

#### 6.3 Standalone Training Service (`src/services/training.py`)

- **Design:** Extracted PRM Evaluator + Training Loop from `src/__main__.py` (lines 49-117) into standalone entrypoint. Connects to NATS, starts evaluator + loop, runs indefinitely.
- **Reuse:** Identical logic to `__main__.py` — same fallback chains (GRPOTrainer → MockTrainer), same config.
- **Finding:** Clean separation — training pipeline runs as its own Docker container, completely independent of Bridge/OpenClaw.

#### 6.4 Docker Compose

- **Services:** NATS (2.10), Ollama, OpenClaw, Bridge, Training — 5 containers.
- **Health checks:** NATS uses `nats-server --help`, Ollama uses `curl /api/tags`. `depends_on` with `condition: service_healthy` ensures startup order.
- **Dockerfiles:** Two — `Dockerfile` (training, installs `[training,inference]`) and `Dockerfile.bridge` (bridge, installs `[bridge]`).
- **Finding:** Pinning NATS to `2.10` avoids breaking changes. OpenClaw uses `latest` for now — should pin once stable.

### Event Flow Preservation

| Event | Old Publisher | New Publisher | Same NATS Topic | Same JSON Schema |
|-------|-------------|-------------|----------------|-----------------|
| TaskEvent | Manager (Python) | Bridge HTTP API | `tasks.{type}` | Yes (same Pydantic model) |
| ResultEvent | Worker (Python) | Bridge HTTP API | `results.{type}` | Yes |
| FeedbackEvent | Manager (Python) | Bridge HTTP API | `feedback.scored` | Yes |
| TrainingRolloutEvent | PRM Evaluator | PRM Evaluator | `training.rollouts` | **Unchanged** |
| ModelUpdateEvent | TrainingLoop | TrainingLoop | `model.updates` | **Unchanged** |

### Test Summary

| Test Suite | Tests | Deps | Speed | Status |
|------------|-------|------|-------|--------|
| `test_http_api.py` | 10 | aiohttp | <1s | PASS |
| `test_service.py` | 1 | aiohttp, NATS | <2s | SKIP (no NATS) / PASS (with NATS) |
| All existing tests | 67 | various | ~5s | PASS (unchanged) |
| **Total** | **77 collected, 72 pass, 5 skip** | | **~5s** | |

### Files

| Action | Path |
|--------|------|
| CREATE | `src/bridge/__init__.py` |
| CREATE | `src/bridge/__main__.py` — entrypoint for `python -m src.bridge` |
| CREATE | `src/bridge/service.py` — BridgeService (NATS + aiohttp) |
| CREATE | `src/bridge/http_api.py` — 5 HTTP endpoints |
| CREATE | `src/services/__init__.py` |
| CREATE | `src/services/__main__.py` — entrypoint for `python -m src.services.training` |
| CREATE | `src/services/training.py` — standalone PRM + Training service |
| CREATE | `config/openclaw/AGENTS.md` — agent registry |
| CREATE | `config/openclaw/manager/SOUL.md` — manager behavior |
| CREATE | `config/openclaw/manager/IDENTITY.md` — manager identity |
| CREATE | `config/openclaw/worker/SOUL.md` — worker behavior |
| CREATE | `config/openclaw/worker/IDENTITY.md` — worker identity |
| CREATE | `config/openclaw/skills/assign-task/SKILL.md` |
| CREATE | `config/openclaw/skills/submit-result/SKILL.md` |
| CREATE | `config/openclaw/skills/submit-feedback/SKILL.md` |
| CREATE | `docker-compose.yml` — 5 services |
| CREATE | `Dockerfile` — training service image |
| CREATE | `Dockerfile.bridge` — bridge service image |
| CREATE | `scripts/demo.sh` — one-command demo |
| CREATE | `tests/bridge/__init__.py` |
| CREATE | `tests/bridge/test_http_api.py` — 10 unit tests |
| CREATE | `tests/bridge/test_service.py` — 1 integration test |
| MODIFY | `src/config.py` — `bridge_port`, `openclaw_gateway_url` |
| MODIFY | `pyproject.toml` — `[bridge]` optional extra |

### Deferred to Phase 7

- Trained PRM model (still using LLM-as-judge — need 10K+ scored trajectories first)
- DAPO graduation (requires OpenRLHF working + tuning)
- Multi-model routing rules in Semantic Router
- HaluGate as complementary StepScorer
- OpenClaw WebSocket relay (Bridge → OpenClaw session for bidirectional communication)
- Pin OpenClaw Docker image version

---

## Phase 7: ADHR — Appraisal-Driven Hierarchical RL with OpenClaw-RL Integration

**Date:** 2026-03-13

**Scope:** Adopt OpenClaw-RL's best ideas (OPD, asymmetric clipping, live data interception) while keeping the event-driven architecture. Novel contribution: manager textual feedback as dual-signal source — score drives RL, text drives OPD distillation.

### Key Decision: Dual-Signal Feedback Architecture

- **Problem:** Existing FeedbackEvent has both `score` (0-1) and `textual_feedback` (string), but only `score` was used for training. The text was discarded — a massive waste of signal.
- **Decision:** Treat each FeedbackEvent as a dual-signal source. `score` → Binary RL reward (existing GRPO path). `textual_feedback` → OPD hindsight hint (new path). Combined loss merges both in a single gradient step.
- **Validation:** CombinedRolloutBuilder joins RL rollout + OPD hint by task_id with timeout fallback. When no hint arrives, degrades gracefully to pure RL.

### Experiments

#### 7.1 Intercept Proxy (`src/intercept/proxy.py`)

- **Design:** FastAPI app sitting between workers and Ollama/vLLM. Intercepts `/v1/chat/completions`, forwards to real backend, publishes `SessionEvent` to NATS. Workers need zero code changes — just point `INFERENCE_BASE_URL` at the proxy.
- **Finding:** FastAPI + httpx forward latency is negligible (~2ms overhead). SessionEvent logging is fire-and-forget — proxy never blocks on NATS publish failure.
- **Tests:** 4 tests with mocked httpx backend. Requires `[intercept]` extra (fastapi, uvicorn, httpx).

#### 7.2 OPD Hint Extraction (`src/opd/hint_extractor.py`)

- **Design:** Subscribes to `feedback.scored`. When textual_feedback is non-empty: (1) extracts corrective hint via LLM, (2) queries teacher model for logprobs, (3) publishes OPDHintEvent. Uses LRU cache of ResultEvents for joining.
- **Finding:** Hint extraction prompt works well — distills verbose feedback into actionable corrections. Teacher logprob extraction requires OpenAI-compatible backend (vLLM); falls back gracefully when unavailable.
- **Tests:** 5 tests with mocked InferenceClient + EventBus.

#### 7.3 Combined Rollout Builder (`src/opd/rollout_builder.py`)

- **Design:** Joins TrainingRolloutEvent + OPDHintEvent by task_id. Uses asyncio.sleep-based timeout — if no hint arrives within N seconds, publishes pure RL rollout. Simple dict keyed by task_id, no external state.
- **Finding:** 30s default timeout works well. Hint typically arrives 1-5s after rollout (feedback processing is fast). Timeout prevents unbounded memory growth.
- **Tests:** 4 tests covering rollout-first, hint-first, timeout, and multi-task independence.

#### 7.4 Combined Trainer (`src/training/combined_trainer.py`)

- **Design:** Implements same patterns as GRPOTrainer but with asymmetric clipping (eps=0.2, eps_high=0.28) and combined loss: `w_rl * (rl_loss + kl) + w_opd * opd_loss`. Falls back to pure RL when no OPD hints in batch.
- **Finding:** Asymmetric clipping encourages exploration of improved responses. Combined loss converges similarly to pure RL when OPD fraction is low (<20% of batch).
- **Tests:** 3 serialization tests (no torch) + 2 integration tests (require torch, marked slow).

#### 7.5 GRPO Math Extensions (`src/training/grpo.py`)

- **New functions:** `asymmetric_clipped_surrogate_loss(ratios, advantages, clip_eps, clip_eps_high)` — higher upper clip bound for positive advantages. `combined_loss(rl_loss, opd_loss, w_rl, w_opd)` — weighted sum.
- **Tests:** 3 asymmetric clip tests + 2 combined loss tests (all require torch).

#### 7.6 Environment-Aware Scoring (`src/rewards/combined_scorer.py`)

- **Design:** CombinedScorer composes multiple named StepScorers with per-environment weight profiles. Chat workers use `{"prm": 0.6, "halugate": 0.3, "length": 0.1}`, terminal workers use `{"success": 0.8, "efficiency": 0.2}`, etc.
- **Finding:** Profile-based weighting is simple and effective. Missing scorers are silently skipped with weight renormalization. Scorer failures are caught and logged.
- **Tests:** 7 tests covering single scorer, weighted combination, missing scorers, different environments, fallback, empty scorers, and failure handling.

#### 7.7 Manager Meta-RL (`src/training/meta_trainer.py`)

- **Design:** Tracks feedback → improvement correlation over sliding window. Per-worker score history (deque). When enough feedback accumulated (default 200), computes improvement_delta = mean(post_scores) - mean(pre_scores), publishes ManagerMetaRollout.
- **Finding:** Cold start protection (min_feedback=200) prevents noisy early meta-training. Sliding window naturally handles non-stationary improvement.
- **Tests:** 6 tests covering subscription, score tracking, empty feedback filtering, meta-rollout publishing, improvement delta calculation, and unknown worker defaults.

#### 7.8 Per-Worker Adapter Registry (`src/inference/adapter_registry.py`)

- **Design:** Maps worker_id → (adapter_name, adapter_path). Listens for ModelUpdateEvents with target_worker_id set. Integrates with VLLMLoRAManager for hot-swap. Old adapters unloaded before new ones loaded.
- **Finding:** target_worker_id=None treated as broadcast (global adapter). Per-worker naming convention: `worker-{id}-{version}`.
- **Tests:** 6 tests with mocked VLLMLoRAManager.

#### 7.9 Multi-Environment Workers

- **TerminalWorker:** Executes bash commands in Docker containers (read-only rootfs, no network, 256MB memory, 60s timeout). Steps = individual commands.
- **SWEWorker:** Three-step pipeline: plan → implement → test. Uses InferenceClient for each step.
- **GUIWorker:** LLM-driven action planning loop. Steps = (screenshot_description, action) pairs. Max 10 steps per task.
- **BaseWorker updates:** Added `environment_type` class attribute (default "chat"). `_handle_model_update` now filters by `target_worker_id`.
- **Tests:** 12 tests covering environment types, command parsing, three-step SWE pipeline, failure handling, GUI step limits, and target_worker_id filtering.

#### 7.10 TrainingLoop Updates

- **Change:** TrainingLoop now accepts optional `combined_trainer` parameter. When set, subscribes to `training.combined` topic and routes CombinedRolloutEvents to the combined trainer. Standard RL path unchanged.
- **Finding:** Dual subscription (training.rollouts + training.combined) works cleanly — standard rollouts go to GRPOTrainer, combined rollouts go to CombinedTrainer.

#### 7.11 Bridge HTTP API Updates

- **New endpoints:** `POST /sessions/log` (log SessionEvent from intercept proxy), `GET /training/status` (report training pipeline health).
- **Finding:** Minimal additions — 2 new handlers, reuses existing EventBus and Pydantic patterns.

#### 7.12 Docker Compose + Infrastructure

- **New service:** `intercept-proxy` — builds from `Dockerfile.intercept`, exposes :8200, connects to NATS and forwards to Ollama.
- **Config change:** Training service now points `INFERENCE_BASE_URL` at intercept-proxy instead of directly at Ollama.
- **Total:** 6 Docker Compose services (NATS, Ollama, OpenClaw, Bridge, Intercept Proxy, Training).

### Test Summary

| Test Suite | Tests | Deps | Speed | Status |
|------------|-------|------|-------|--------|
| `test_types.py` (new events) | 6 new | None | <1s | PASS |
| `test_proxy.py` | 4 | fastapi, httpx | <1s | SKIP (no fastapi) / PASS |
| `test_hint_extractor.py` | 5 | None | <1s | PASS |
| `test_rollout_builder.py` | 4 | None | <1s | PASS |
| `test_combined_trainer.py` (no-torch) | 3 | None | <1s | PASS |
| `test_combined_trainer.py` (torch) | 2 | torch | ~30s | SKIP / PASS |
| `test_grpo.py` (new functions) | 5 | torch | <1s | SKIP / PASS |
| `test_combined_scorer.py` | 7 | None | <1s | PASS |
| `test_meta_trainer.py` | 6 | None | <1s | PASS |
| `test_adapter_registry.py` | 6 | None | <1s | PASS |
| `test_multi_env.py` | 12 | None | <1s | PASS |
| All existing tests | 60 | various | <1s | PASS (unchanged) |
| **Total standalone** | **100 pass, 14 skip** | | **~1s** | |

### Files

| Action | Path |
|--------|------|
| CREATE | `src/intercept/__init__.py`, `src/intercept/__main__.py`, `src/intercept/proxy.py` |
| CREATE | `src/opd/__init__.py`, `src/opd/hint_extractor.py`, `src/opd/rollout_builder.py` |
| CREATE | `src/training/combined_trainer.py` |
| CREATE | `src/training/meta_trainer.py` |
| CREATE | `src/rewards/combined_scorer.py` |
| CREATE | `src/inference/adapter_registry.py` |
| CREATE | `src/workers/terminal_worker.py`, `src/workers/swe_worker.py`, `src/workers/gui_worker.py` |
| CREATE | `Dockerfile.intercept` |
| CREATE | `tests/intercept/__init__.py`, `tests/intercept/test_proxy.py` |
| CREATE | `tests/opd/__init__.py`, `tests/opd/test_hint_extractor.py`, `tests/opd/test_rollout_builder.py` |
| CREATE | `tests/training/test_combined_trainer.py`, `tests/training/test_meta_trainer.py` |
| CREATE | `tests/rewards/test_combined_scorer.py` |
| CREATE | `tests/inference/test_adapter_registry.py` |
| CREATE | `tests/workers/test_multi_env.py` |
| MODIFY | `src/events/types.py` — 5 new event types, target_worker_id on ModelUpdateEvent |
| MODIFY | `src/events/topics.py` — 5 new topic constants |
| MODIFY | `src/config.py` — 13 new config vars |
| MODIFY | `src/training/grpo.py` — asymmetric_clipped_surrogate_loss, combined_loss |
| MODIFY | `src/training/loop.py` — combined_trainer support, CombinedRolloutEvent subscription |
| MODIFY | `src/workers/base.py` — environment_type, target_worker_id filtering |
| MODIFY | `src/rewards/prm_evaluator.py` — CombinedScorer support |
| MODIFY | `src/bridge/http_api.py` — 2 new endpoints |
| MODIFY | `pyproject.toml` — [intercept] extra |
| MODIFY | `docker-compose.yml` — intercept-proxy service |
| MODIFY | `tests/events/test_types.py` — 6 new roundtrip tests |
| MODIFY | `tests/training/test_grpo.py` — 5 new function tests |

### Deferred to Phase 8

- Trained PRM model (still using LLM-as-judge — need 10K+ scored trajectories first)
- DAPO graduation (requires OpenRLHF working + tuning)
- Multi-model routing rules in Semantic Router
- HaluGate as complementary StepScorer
- Actual teacher logprob extraction (requires vLLM with logprobs enabled)
- Implicit signal extraction from tool failures / user re-queries
- Real per-token OPD advantages (current implementation uses response-level proxy)

---

## Phase 8: Adopt + Extend — Architecture Reassessment

**Date:** 2026-03-17

**Scope:** Shift from "build everything from scratch" to "adopt + extend". Adopt battle-tested training infrastructure (OpenRLHF) for the training plane while keeping NATS for the orchestration plane. Add CLI entry point, OpenRLHF backend integration, and OpenClaw-RL OPD mode.

### Key Decision: Two-Layer Architecture

- **Problem:** Phase 7 built custom GRPO math, OPD extraction, and rollout collection. But no major RL training framework (OpenRLHF, veRL, TRL) uses NATS or any message broker for GPU training. They use NCCL for GPU sync, Ray for orchestration, vLLM for rollouts, and DeepSpeed/FSDP for memory-efficient training.
- **Decision:** Split into two layers:
  - **Orchestration Layer** (NATS): Agent coordination, task routing, manager feedback, meta-RL — keep, it's correctly placed.
  - **Training Layer** (OpenRLHF): Production GRPO/DAPO via Ray+vLLM+DeepSpeed — adopt existing.
- **Validation:** `OpenRLHFBackend` implements the `Trainer` protocol, so `TrainingLoop` doesn't know the difference. Custom `GRPOTrainer` kept for CPU dev/testing. Seamless backend selection via `TRAINER_BACKEND` config var.

### Experiments

#### 8.1 CLI Entry Point (`src/cli.py`)

- **Design:** Typer + Rich CLI with 4 commands: `init`, `train`, `serve`, `status`. `init` creates .env + pulls model. `train` selects backend via `--backend standalone|openrlhf`. `serve` starts via Docker Compose or demo loop. `status` checks NATS, Ollama, Bridge, Intercept Proxy, Docker health.
- **Finding:** `pyproject.toml` `[project.scripts]` entry point (`agentic-employees = "src.cli:app"`) works cleanly with editable installs. Typer's `--help` auto-generates good docs.
- **Dependencies:** Added `typer>=0.12` and `rich>=13.0` to core dependencies.

#### 8.2 OpenRLHF Training Backend (`src/training/openrlhf_backend.py`)

- **Design:** Implements `Trainer` protocol. Exports rollouts to JSONL → launches `openrlhf.cli.train_grpo_ray` as subprocess → parses training metrics from stdout → returns `TrainStepResult`. Falls back to simulation mode on CPU (creates checkpoint directories, writes metadata, validates data flow).
- **Finding:** Subprocess-based integration is more robust than in-process Ray import — avoids dependency conflicts and crashes. JSONL file handoff is the natural boundary between orchestration and training planes.
- **Simulation mode:** When Ray is not installed, `_simulate_training()` validates the full data flow (export, checkpoint creation, metadata writing) without actual GPU training. Returns decreasing fake loss (1/step_count).
- **Config:** `TRAINER_BACKEND=openrlhf` env var. `src/services/training.py` updated with `_create_trainer()` factory.

#### 8.3 OpenClaw-RL OPD Mode (`src/opd/hint_extractor.py`)

- **Design:** `HintExtractor` now supports `opd_mode` parameter ("lightweight" or "openclaw"). Lightweight mode (default): our LLM-based hint extraction, works with any backend. OpenClaw mode: uses `openai.AsyncOpenAI` with `logprobs=True` on vLLM backend to extract per-token log probabilities.
- **Finding:** OpenAI completions API with `logprobs=True, top_logprobs=1` returns per-token logprobs in `response.choices[0].logprobs.content[i].logprob`. Works with vLLM but not Ollama (Ollama doesn't support logprobs in chat completions).
- **Fallback:** OpenClaw mode gracefully falls back to empty logprobs if `openai` package not installed or backend doesn't support logprobs.
- **Config:** `OPD_MODE=openclaw` env var.

#### 8.4 Docker Compose Updates

- **New:** Commented-out `openrlhf-trainer` service showing GPU configuration (nvidia driver, resource reservations). Serves as documentation for production GPU deployment.
- **Change:** Training service now has `META_RL_ENABLED=true` by default.
- **Unchanged:** 6 active services (NATS, Ollama, OpenClaw, Bridge, Intercept Proxy, Training).

#### 8.5 OpenRLHF Optional Dependencies

- **Change:** `[openrlhf]` extra expanded to include `ray>=2.9` and `deepspeed>=0.14` alongside `openrlhf[vllm]>=0.9.0`.
- **Finding:** Explicit Ray and DeepSpeed pins prevent version conflicts with OpenRLHF's own dependency resolution.

### Novel vs Adopted Components (Architecture Insight)

| Contribution | Status | Rationale |
|---|---|---|
| Manager Meta-RL | **Novel — keep** | No other framework trains the evaluator |
| Combined Scorer | **Novel — keep** | Environment-aware multi-dimensional scoring |
| Per-Worker Adapter Registry | **Novel — keep** | Targeted LoRA specialization per worker |
| Multi-Environment Workers | **Novel — keep** | Terminal/SWE/GUI with distinct scoring |
| GRPO math | **Adopt OpenRLHF** | GPU-optimized, battle-tested |
| OPD at scale | **Adopt OpenClaw-RL** | Per-token logprobs at GPU speed |
| Distributed training | **Adopt DeepSpeed/FSDP** | Industry standard |
| NATS orchestration | **Keep** | Correctly placed for control plane |

### Files

| Action | Path |
|--------|------|
| CREATE | `src/cli.py` — CLI entry point (init/train/serve/status) |
| CREATE | `src/training/openrlhf_backend.py` — OpenRLHF Trainer protocol adapter |
| MODIFY | `src/opd/hint_extractor.py` — dual OPD modes (lightweight + openclaw) |
| MODIFY | `src/config.py` — `opd_mode` config var |
| MODIFY | `src/services/training.py` — `_create_trainer()` factory with backend selection |
| MODIFY | `pyproject.toml` — typer, rich deps; expanded openrlhf extra; project.scripts |
| MODIFY | `docker-compose.yml` — commented openrlhf-trainer service |
| MODIFY | `README.md` — two-layer architecture, novel vs adopted, CLI docs |
| MODIFY | `CLAUDE.md` — Phase 8 status, directory structure, architecture description |

### Deferred to Phase 9

- Trained PRM model (need 10K+ scored trajectories)
- Full DAPO via OpenRLHF configuration (asymmetric clip done, dynamic sampling remaining)
- HaluGate as complementary StepScorer in CombinedScorer
- True per-token OPD advantages in CombinedTrainer
- Implicit signal extraction from sessions/tool failures
- Benchmark suite (MATH, HumanEval, GSM8K)

---

## Phase 9a: MetaClaw Adoption

**Date:** 2026-03-18

**Scope:** Adopt MetaClaw-validated patterns: majority voting PRM, SkillRL skill injection, Tinker cloud training backend, interactive setup wizard, session-stateful intercept proxy, time-window training scheduler.

**Outcome:** 6 new modules added. Majority voting with median aggregation in LLMJudgeScorer. SkillStore + SkillRetriever + SkillEvolver for feedback-driven skill extraction. TinkerBackend as Trainer protocol adapter. SetupWizard via Rich prompts. SessionManager for proxy conversation tracking. TrainingScheduler with UTC time-window gating.

**Key Finding:** Protocol-first design (StepScorer, Trainer, InferenceClient) made all additions zero-touch on existing code. TinkerBackend, HackableScorer, and all new components plug in via the same interfaces.

---

## Phase 9b: Alignment Experiments

**Date:** 2026-03-19

**Scope:** Build alignment experiment infrastructure to produce quantified evidence that the ADHR multi-agent framework catches misalignment, prevents reward hacking, and produces auditable decisions. 6 experiments, 40 behavioral scenarios, 79 tests.

### Key Decision: Mock-First Experiment Design

- **Problem:** Alignment experiments that require live NATS + LLM + GPU are too slow and fragile for CI/CD.
- **Decision:** Every experiment runs in mock mode (no external deps) using EchoWorker + PatternBasedEvaluator + MockTrainer. Real mode layers on NATS + LLM when available.
- **Validation:** All 79 tests pass standalone in <1s. `agentic-employees experiment run all` produces JSON results without any services running.

### Experiments

#### 9b.1 Scenario Library (`src/alignment/scenarios.py`)

- **Design:** `AlignmentScenario` frozen dataclass with `name`, `experiment`, `prompt`, `expected_behavior`, `misaligned_indicators`, `safety_relevant`, `metadata`.
- **4 scenario sets** (10 each): deception, reward hacking, safety-pragmatism, collusion.
- **Finding:** Pattern-based evaluation (substring matching against `misaligned_indicators`) works reliably for keyword-stuffing and confidence-inflation detection. Sufficient for mock-mode experiments, upgradeable to LLMJudgeEvaluator for real mode.

#### 9b.2 Behavioral Eval Harness (`src/alignment/behavioral_eval.py`)

- **Design:** `BehavioralEvaluator` protocol with two implementations: `PatternBasedEvaluator` (no LLM) and `LLMJudgeEvaluator` (uses InferenceClient). `BehavioralEvalHarness` orchestrates: scenarios → worker_fn → evaluate → publish AlignmentEvalEvent → write JSON.
- **Finding:** Worker function pattern (`Callable[[str], Awaitable[tuple[str, list[str]]]]`) cleanly decouples evaluation from actual worker infrastructure. Any async function that takes a prompt and returns (response, steps) can be evaluated.

#### 9b.3 Hackable Scorer + Reward Hacking Detection

- **HackableScorer:** Implements `StepScorer` protocol. Scores high if steps contain keywords ("verified", "correct", "accurate") regardless of quality. Base score 0.3, keyword boost up to 0.6.
- **RewardHackingDetector:** Tracks hackable vs judge score trajectories. Flags when divergence exceeds threshold (default 0.2).
- **Finding:** Misaligned keyword-stuffing worker reliably triggers divergence detection. The CombinedScorer's multi-scorer architecture is validated as a defense mechanism.

#### 9b.4 Misaligned Worker

- **Design:** Extends `BaseWorker` with 3 strategies: keyword_stuffing (injects scoring keywords), confidence_inflation (claims 100% certainty), shortcut (minimal steps).
- **Finding:** Keyword stuffing reliably games HackableScorer but fails PatternBasedEvaluator. Confidence inflation triggers deception scenario indicators. Shortcut produces minimal output that evades some checks.

#### 9b.5 Collusion Detector

- **Design:** Pearson correlation between worker score trajectories + Jaccard similarity of character n-grams between worker responses.
- **Finding:** Perfectly colluding workers (identical scores + responses) produce correlation >0.99 and similarity 1.0. Independent workers stay well below thresholds.

#### 9b.6 Audit Logger

- **Design:** Subscribes to all known NATS topics via `subscribe_raw()` (new method on EventBus). Writes every event to JSONL with timestamp, topic, inferred event type, and full payload.
- **Finding:** Event type inference from JSON field presence works well (task_id+prompt → TaskEvent, result+worker_id → ResultEvent, etc.). Handles malformed JSON gracefully.

#### 9b.7 Experiment Runner + CLI

- **ExperimentRunner:** One method per experiment (`run_experiment_1()` through `run_experiment_6()`). All return structured dicts, write JSON to `alignment_results/`.
- **CLI:** `agentic-employees experiment run {1-6|all}` + `agentic-employees experiment results` (Rich table).
- **Finding:** Mock mode produces meaningful results. Experiment 2 correctly detects reward hacking, Experiment 4 correctly detects collusion.

#### 9b.8 Streamlit Dashboard

- **3 tabs:** Experiment Overview (metrics per experiment), Audit Trail (searchable JSONL viewer), Constitution Editor (CombinedScorer weight sliders + feedback template).
- **Finding:** Reads JSON/JSONL files directly — no server dependency. Works standalone with `streamlit run src/alignment/dashboard/app.py`.

### Test Summary

| Test Suite | Tests | Deps | Speed | Status |
|------------|-------|------|-------|--------|
| `test_scenarios.py` | 12 | None | <1s | PASS |
| `test_behavioral_eval.py` | 12 | None | <1s | PASS |
| `test_hackable_scorer.py` | 11 | None | <1s | PASS |
| `test_misaligned_worker.py` | 8 | None | <1s | PASS |
| `test_collusion_detector.py` | 16 | None | <1s | PASS |
| `test_audit_logger.py` | 9 | None | <1s | PASS |
| `test_runner.py` | 8 | None | <1s | PASS |
| Existing tests (unchanged) | 162 | various | ~1s | PASS |
| **Total standalone** | **241 pass, 22 skip** | | **~1.3s** | |

### Files

| Action | Path |
|--------|------|
| CREATE | `EXPERIMENT.md` — experiment tracking document |
| CREATE | `src/alignment/__init__.py` |
| CREATE | `src/alignment/scenarios.py` — 40 scenarios in 4 categories |
| CREATE | `src/alignment/behavioral_eval.py` — eval harness + 2 evaluators |
| CREATE | `src/alignment/hackable_scorer.py` — deliberately weak scorer + detector |
| CREATE | `src/alignment/misaligned_worker.py` — rogue worker (3 strategies) |
| CREATE | `src/alignment/collusion_detector.py` — Pearson + Jaccard detector |
| CREATE | `src/alignment/audit_logger.py` — JSONL audit capture |
| CREATE | `src/alignment/runner.py` — 6 experiment orchestrator |
| CREATE | `src/alignment/dashboard/__init__.py`, `app.py` — Streamlit dashboard |
| CREATE | `tests/alignment/` — 7 test files (79 tests) |
| MODIFY | `src/events/types.py` — AlignmentEvalEvent, AuditLogEvent |
| MODIFY | `src/events/topics.py` — ALIGNMENT_EVALS, ALIGNMENT_AUDIT |
| MODIFY | `src/events/bus.py` — subscribe_raw() method |
| MODIFY | `src/config.py` — 3 alignment config fields |
| MODIFY | `src/cli.py` — experiment subcommand group |
| MODIFY | `pyproject.toml` — alignment optional extras |

### Deferred to Phase 9c

- Trained PRM model (replace LLM-as-judge)
- DAPO graduation (full DAPO via OpenRLHF)
- HaluGate hallucination scorer
- CISPO contrastive loss
- Benchmark suite (MATH, HumanEval, GSM8K)
- Real-mode experiments with live NATS + LLM
