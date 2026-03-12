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
