# NEXT-TO-DO

What to build next. Phase 8 shifts from "build everything from scratch" to "adopt + extend" — use battle-tested training infrastructure (OpenRLHF/veRL) and focus our effort on genuinely novel components.

> **Phase 7 completed** — OPD, Combined Training, Meta-RL, Multi-Environment Workers, Intercept Proxy, and Per-Worker Adapter Registry.
>
> **Phase 8 in progress** — CLI, OpenRLHF backend integration, OpenClaw-RL OPD mode, honest architecture positioning.

---

## Architecture Reassessment (Why Phase 8 Changed Direction)

**No major RL training framework uses NATS or any message broker for their core training loop.** Not OpenClaw-RL, not OpenRLHF, not veRL, not TRL. The industry uses NCCL for GPU sync, Ray for orchestration, vLLM for rollouts, and DeepSpeed/FSDP for memory-efficient training.

**What we got right:** NATS for agent orchestration (small JSON events, pub/sub, fault tolerance). This is the control plane — and it's well-placed.

**What we got wrong:** Building custom training infrastructure (GRPO math, OPD extraction, rollout collection) when GPU-optimized, battle-tested versions exist in OpenRLHF/veRL.

**Our genuinely novel contributions (worth keeping and publishing):**
1. Manager Meta-RL (outer loop that trains the evaluator)
2. Combined Scorer with environment-specific weight profiles
3. Per-worker LoRA adapter registry with targeted updates
4. Multi-environment workers (Terminal/SWE/GUI with distinct scoring)

**Two-layer architecture going forward:**
```
ORCHESTRATION LAYER (NATS — keep, it's correct here)
├── Agent coordination (task routing, worker management)
├── Manager feedback loop (our novel meta-RL)
├── Multi-environment scoring (our novel combined scorer)
└── Per-worker adapter management (our novel registry)

TRAINING LAYER (adopt existing — don't reinvent)
├── OpenRLHF or veRL for GRPO/DAPO training (Ray + vLLM + DeepSpeed)
├── OpenClaw-RL's OPD implementation (teacher logprobs at GPU speed)
└── Standard checkpoint management (HuggingFace Hub / S3)
```

---

## Phase 8 — Adopt + Extend

### 8.1 CLI Entry Point ✅ Done
- `agentic-employees init` — sets up config, pulls base model
- `agentic-employees train --backend openrlhf|standalone` — one command to train
- `agentic-employees serve --docker` — starts via Docker Compose
- `agentic-employees status` — shows all service health

### 8.2 OpenRLHF Training Backend ✅ Done
- `OpenRLHFBackend` implements `Trainer` protocol
- Exports our rollouts to JSONL → feeds to OpenRLHF Ray training
- Falls back to simulation mode on CPU (validates data flow without GPU)
- `src/services/training.py` uses backend selection: standalone vs openrlhf
- Keeps our custom GRPOTrainer for CPU testing/dev

### 8.3 OpenClaw-RL OPD Mode ✅ Done
- `HintExtractor` now supports `opd_mode="openclaw"` for vLLM logprob extraction
- Uses OpenAI completions API with `logprobs=True` on vLLM backend
- Extracts per-token log probabilities as directional training signal
- Falls back to lightweight mode if vLLM logprobs unavailable
- Config: `OPD_MODE=openclaw` env var

### 8.4 Trained PRM Model
- **What:** Replace LLM-as-judge with a trained process reward model (classifier head on frozen LLM)
- **Why:** LLM-as-judge is slow (~1s per step), expensive, inconsistent. Trained PRM is 10-100x faster.
- **Prerequisite:** Accumulate 10K+ scored trajectories via the Docker demo loop
- **Implementation:** Fine-tune small model (Qwen2.5:0.5b) with scalar reward head. Implement `TrainedPRMScorer` as `StepScorer` protocol adapter. Plug into CombinedScorer.
- **Research:** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), [AgentPRM](https://arxiv.org/abs/2502.10325)

### 8.5 DAPO Graduation
- **What:** Upgrade GRPO to full DAPO (Dynamic Advantage Policy Optimization)
- **Phase 7 progress:** Asymmetric clipping (Clip-Higher) done in `asymmetric_clipped_surrogate_loss()`.
- **Remaining:** Dynamic sampling (oversample, filter low-reward, keep diverse rollouts) + entropy bonus.
- **Implementation:** With OpenRLHF backend, this becomes configuration — OpenRLHF supports DAPO natively.
- **Research:** [DAPO (ByteDance)](https://arxiv.org/abs/2503.14476)

### 8.6 HaluGate Scorer
- **What:** Hallucination detection as complementary `StepScorer`
- **Why:** PRM scores reasoning quality but doesn't catch factual hallucinations.
- **Implementation:** `HaluGateScorer` as `StepScorer` protocol adapter. Score = similarity between step claims and retrieved evidence. Combine with PRM via CombinedScorer.

---

## Priority 2 — Signal Richness

### 2.1 True Per-Token OPD Advantages
- **What:** Extract actual per-token log-probability gaps from teacher vs policy forward passes.
- **Status:** OpenClaw OPD mode now extracts teacher logprobs. Next: compute `A_token[k] = log pi_teacher - log pi_theta` per token in CombinedTrainer.
- **Prerequisite:** vLLM with logprobs enabled.

### 2.2 Implicit Signal Extraction
- **What:** Automatically detect training signals from tool failures, user corrections, re-queries.
- **Why:** The next state after an agent action is a free reward signal.
- **Implementation:** `ImplicitSignalDetector` subscribes to `sessions.logged` and `results.*` topics. Classifies follow-up events as positive/negative signals.

### 2.3 Majority Voting for PRM
- **What:** Run `m` parallel judge calls per step and take majority vote.
- **Why:** Single LLM-as-judge calls have high variance. Trivial improvement.
- **Implementation:** Add `num_votes: int = 1` param to `LLMJudgeScorer`. Run `m` concurrent calls. Final score = median.

### 2.4 Session-Aware Trajectory Classification
- **What:** Classify agent interactions into main-line turns (trainable) vs side turns (noise).
- **Why:** Training on everything (memory organization, status checks) adds noise.

---

## Priority 3 — Production Readiness

### 3.1 Multi-Model Routing (Semantic Router)
- Route tasks to best-fit model based on task type/complexity. Already supported — just configure `INFERENCE_BASE_URL`.

### 3.2 Trajectory Data Pipeline
- Persist scored trajectories to queryable store (SQLite → PostgreSQL). Required for Trained PRM (8.4).

### 3.3 OpenClaw WebSocket Relay
- Bidirectional WebSocket Bridge ↔ OpenClaw session communication.

### 3.4 Pin OpenClaw Docker Image
- Use specific tagged release instead of `latest`.

---

## Priority 4 — Scale & Operations

### 4.1 Kubernetes Deployment
- Helm chart for K8s. NATS (nats-operator), vLLM (GPU node pools), Bridge (Deployment + Service).

### 4.2 Observability Stack
- Structured logging, Prometheus metrics, distributed tracing.

### 4.3 CI/CD Pipeline
- GitHub Actions: ruff check → pytest → Docker build → integration test → push to GHCR.

### 4.4 Benchmark Suite
- Reproducible benchmarks: MATH, HumanEval, GSM8K. Compare before/after GRPO training.

---

## What We're NOT Doing

- **Ripping out NATS** — it's correctly placed for orchestration
- **Deleting custom trainers** — useful for CPU testing/dev
- **Abandoning novel components** — they're our actual differentiation (meta-RL, combined scorer, adapter registry, multi-env workers)
- **Building custom NCCL/GPU training** — adopt OpenRLHF/veRL instead

---

## Sequencing (Phase 8 Forward)

```
Now ──────────────────────────────────────────────────────────────► Later

[8.1 CLI] ✅ ──► [8.2 OpenRLHF Backend] ✅ ──► [8.5 DAPO (via OpenRLHF)]
[8.3 OpenClaw OPD] ✅ ──► [2.1 Token OPD Advantages]

[3.2 Trajectory Store] ──► [8.4 Trained PRM] ──► [4.4 Benchmarks]

[2.3 Majority Voting] (trivial)
[8.6 HaluGate] ──► CombinedScorer integration

[3.3 WebSocket Relay] ──► [2.2 Implicit Signals]
[2.4 Turn Classification] ──► noise filtering

[4.1 K8s] ──► [4.2 Observability] ──► [4.3 CI/CD]
```

**Immediate next:**
1. **3.2 Trajectory Store** — unblocks Trained PRM (8.4)
2. **2.3 Majority Voting** — trivial improvement to scoring reliability
3. **8.6 HaluGate** — plug into CombinedScorer, no new infrastructure
4. **2.1 Token OPD Advantages** — build on OpenClaw OPD mode
