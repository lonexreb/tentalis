# NEXT-TO-DO

What to build next. Phase 9b completed alignment experiment infrastructure — 6 experiments, 40 scenarios, 79 tests, full audit trail. Now focusing on Phase 9c: trained models, advanced scorers, and benchmarks.

> **Phase 8 completed** — CLI, OpenRLHF backend, OpenClaw-RL OPD mode, honest positioning.
>
> **Phase 9a completed** — Majority Voting PRM, SkillRL, Tinker backend, Setup Wizard, Training Scheduler.
>
> **Phase 9b completed** — Alignment experiments, collusion detection, reward hacking detection, audit logger, Streamlit dashboard.
>
> **Phase 9c next** — Trained PRM, DAPO graduation, HaluGate, CISPO, benchmarks.

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

## Phase 8 — Adopt + Extend ✅ Complete

### 8.1 CLI Entry Point ✅ Done
### 8.2 OpenRLHF Training Backend ✅ Done
### 8.3 OpenClaw-RL OPD Mode ✅ Done

---

## Phase 9a — MetaClaw Adoption ✅ Complete

- Majority Voting PRM (parallel LLM judge evals with median aggregation)
- SkillRL (skill store, embedding retriever, evolver from feedback)
- Tinker training backend (cloud-managed RL via Tinker SDK)
- Interactive setup wizard (Rich multi-step config wizard)
- Session-stateful intercept proxy (session tracking + skill injection)
- Training scheduler (time-window gated training)

---

## Phase 9b — Alignment Experiments ✅ Complete

- 40 behavioral scenarios (deception, reward hacking, safety-pragmatism, collusion)
- PatternBasedEvaluator + LLMJudgeEvaluator with BehavioralEvalHarness
- HackableScorer + RewardHackingDetector
- MisalignedWorker (keyword stuffing, confidence inflation, shortcut)
- CollusionDetector (Pearson correlation + Jaccard n-gram similarity)
- AuditLogger (subscribe_raw → JSONL)
- ExperimentRunner (6 experiments, mock mode)
- Streamlit dashboard (experiment overview, audit timeline, constitution editor)
- CLI: `tentalis experiment run|results`
- See [EXPERIMENT.md](./EXPERIMENT.md) for details.

---

## Phase 9c — Advanced Scorers + Benchmarks (Next)

### 9c.1 Trained PRM Model
- **What:** Replace LLM-as-judge with a trained process reward model (classifier head on frozen LLM)
- **Why:** LLM-as-judge is slow (~1s per step), expensive, inconsistent. Trained PRM is 10-100x faster.
- **Prerequisite:** Accumulate 10K+ scored trajectories via the Docker demo loop
- **Implementation:** Fine-tune small model (Qwen2.5:0.5b) with scalar reward head. Implement `TrainedPRMScorer` as `StepScorer` protocol adapter. Plug into CombinedScorer.
- **Research:** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), [AgentPRM](https://arxiv.org/abs/2502.10325)

### 9c.2 DAPO Graduation
- **What:** Upgrade GRPO to full DAPO (Dynamic Advantage Policy Optimization)
- **Phase 7 progress:** Asymmetric clipping (Clip-Higher) done in `asymmetric_clipped_surrogate_loss()`.
- **Remaining:** Dynamic sampling (oversample, filter low-reward, keep diverse rollouts) + entropy bonus.
- **Implementation:** With OpenRLHF backend, this becomes configuration — OpenRLHF supports DAPO natively.
- **Research:** [DAPO (ByteDance)](https://arxiv.org/abs/2503.14476)

### 9c.3 HaluGate Scorer
- **What:** Hallucination detection as complementary `StepScorer`
- **Why:** PRM scores reasoning quality but doesn't catch factual hallucinations.
- **Implementation:** `HaluGateScorer` as `StepScorer` protocol adapter. Score = similarity between step claims and retrieved evidence. Combine with PRM via CombinedScorer.

### 9c.4 CISPO Contrastive Loss
- **What:** Contrastive loss between aligned vs misaligned trajectories.
- **Why:** Phase 9b alignment experiments produce natural contrastive pairs (MisalignedWorker = negatives, EchoWorker/LLMWorker = positives).

### 9c.5 Benchmark Suite
- **What:** Reproducible benchmarks on MATH, HumanEval, GSM8K.
- **Why:** Quantify improvement from GRPO training with standardized tasks.

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

## Sequencing (Phase 9c Forward)

```
Now ──────────────────────────────────────────────────────────────► Later

[9c.1 Trained PRM] ──► [9c.5 Benchmarks]
[9c.2 DAPO] (via OpenRLHF config)
[9c.3 HaluGate] ──► CombinedScorer integration
[9c.4 CISPO] ──► uses Phase 9b alignment data

[2.1 Token OPD Advantages] ──► CombinedTrainer upgrade
[3.2 Trajectory Store] ──► unblocks Trained PRM

[4.1 K8s] ──► [4.2 Observability] ──► [4.3 CI/CD]
```

**Immediate next:**
1. **3.2 Trajectory Store** — unblocks Trained PRM (9c.1)
2. **9c.3 HaluGate** — plug into CombinedScorer, no new infrastructure
3. **9c.4 CISPO** — leverage alignment experiment data for contrastive training
4. **2.1 Token OPD Advantages** — build on OpenClaw OPD mode
