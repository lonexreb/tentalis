# NEXT-TO-DO

What to build next to turn agentic-employees from a working prototype into a production-grade, differentiated product.

> **Phase 7 completed** — OPD, Combined Training, Meta-RL, Multi-Environment Workers, Intercept Proxy, and Per-Worker Adapter Registry are now implemented. Items marked ~~strikethrough~~ below are done.

---

## Priority 1 — Signal Richness (Inspired by OpenClaw-RL)

OpenClaw-RL ([arXiv:2603.10165](https://arxiv.org/abs/2603.10165), Princeton AI Lab / Gen-Verse) is the closest open-source competitor. Their key insight: **scalar rewards are not enough**. Their On-Policy Distillation (OPD) produces token-level directional training signals and outperforms scalar Binary RL by 3x (0.72 vs 0.23 accuracy at 16 steps). ~~We should adopt their best ideas while keeping our production architecture.~~ **Done in Phase 7.**

### ~~1.1 Hindsight-Guided On-Policy Distillation (OPD)~~ ✅ Phase 7
- **Status:** Implemented in `src/opd/hint_extractor.py` + `src/opd/rollout_builder.py` + `src/training/combined_trainer.py`.
- **What was built:** HintExtractor converts manager textual feedback → corrective hints. CombinedRolloutBuilder joins RL rollouts + OPD hints by task_id with timeout. CombinedTrainer merges RL + OPD in a single gradient step with asymmetric clipping.
- **Remaining work:** Per-token logprob extraction requires vLLM with logprobs enabled (current implementation uses response-level proxy). True token-level OPD advantages deferred to Phase 8.

### 1.2 Implicit Signal Extraction
- **What:** Automatically detect training signals from tool failures, user corrections, re-queries, and environment feedback — without requiring explicit manager scoring.
- **Why:** The *next state* after an agent action is a free reward signal. If a user re-asks the same question, the first answer was bad. If a tool returns an error, the tool call was wrong. The intercept proxy (Phase 7) now captures all inference sessions, making implicit signal extraction much easier.
- **Implementation:** Add `ImplicitSignalDetector` that subscribes to `sessions.logged` and `results.*` topics and classifies follow-up events:
  - User re-query on same topic → negative signal on prior response
  - Tool/API error in result → negative signal on the step that produced the call
  - User acceptance (moves to new topic) → positive signal
  - Publish implicit signals as `TrainingRolloutEvent` with source=`implicit`
- **Prerequisite:** Bridge WebSocket relay (2.2) for real-time user interaction tracking

### 1.3 Majority Voting for PRM
- **What:** Run `m` parallel judge calls per step and take majority vote for the final score.
- **Why:** Single LLM-as-judge calls have high variance (OpenClaw-RL uses m=1 for text, m=3 for GUI). Majority voting is a trivial improvement that significantly reduces scoring noise.
- **Implementation:** Add `num_votes: int = 1` parameter to `LLMJudgeScorer`. Run `m` concurrent `score_steps()` calls. Final score = median of `m` scores. Cost: `m`x judge calls, but parallelized so latency stays ~1x.
- **Note:** CombinedScorer (Phase 7) already supports multi-scorer composition — majority voting can be added as another scorer in the profile.

### 1.4 Trained PRM Model
- **What:** Replace LLM-as-judge with a trained process reward model (classifier head on frozen LLM)
- **Why:** LLM-as-judge is slow (~1s per step), expensive (full LLM call per evaluation), and inconsistent. A trained PRM is 10-100x faster and produces calibrated scores.
- **Prerequisite:** Accumulate 10K+ scored trajectories via the Docker demo loop
- **Implementation:** Fine-tune a small model (e.g., Qwen2.5:0.5b) with a scalar reward head on the scored step dataset. Replace `LLMJudgeScorer` with `TrainedPRMScorer` implementing the same `StepScorer` protocol. Can be plugged into CombinedScorer alongside other scorers.
- **Research:** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), [AgentPRM](https://arxiv.org/abs/2502.10325)

### 1.5 Session-Aware Trajectory Classification
- **What:** Classify agent interactions into main-line turns (trainable) vs side turns (auxiliary/noise).
- **Why:** OpenClaw-RL found that training on everything — including memory organization, status checks, and meta-queries — adds noise. Only "primary response + tool result" turns should become training samples.
- **Implementation:** Add a `TurnClassifier` to the rollout pipeline. Classify by event topic and content pattern: `tasks.*` and `results.*` = main-line; `model.updates`, health checks, status polls = side turns. Only main-line turns enter `RolloutBuffer`.
- **Note:** Intercept proxy (Phase 7) now logs all sessions via `SessionEvent`, making classification straightforward.

### 1.6 True Per-Token OPD Advantages (NEW)
- **What:** Extract actual per-token log-probability gaps from teacher vs policy forward passes.
- **Why:** Current OPD implementation uses response-level loss proxy. True token-level advantages (as described in OpenClaw-RL paper) provide much finer-grained training signal.
- **Prerequisite:** vLLM with logprobs enabled. Intercept proxy already captures `token_logprobs` field.
- **Implementation:** Extend CombinedTrainer to compute `A_token[k] = log pi_teacher(a|s_enhanced) - log pi_theta(a|s_original)` per token. Requires tokenizer alignment between teacher and policy models.

---

## Priority 2 — Close the Gap with Devin (Differentiation)

These items are what separate "interesting open-source project" from "credible alternative to closed-source RL-trained agents."

### 2.1 DAPO Graduation (Partially Done)
- **What:** Upgrade GRPO to DAPO (Dynamic Advantage Policy Optimization)
- **Why:** GRPO clips advantages symmetrically, which can suppress exploration. DAPO's Clip-Higher removes upper clipping and adds dynamic sampling — better training signal for harder tasks.
- **Phase 7 progress:** Asymmetric clipping (eps=0.2, eps_high=0.28) is now implemented in `asymmetric_clipped_surrogate_loss()` and used by CombinedTrainer. This is the "Clip-Higher" component of DAPO.
- **Remaining:** Dynamic sampling (oversample, filter low-reward, keep diverse rollouts) and entropy bonus. Requires OpenRLHF functional + GPU access.
- **Implementation:** Add `DAPOTrainer` implementing the `Trainer` protocol with full DAPO: asymmetric clip (done) + entropy bonus + dynamic response generation.
- **Research:** [DAPO (ByteDance)](https://arxiv.org/abs/2503.14476)

### 2.2 HaluGate Scorer
- **What:** Hallucination detection as a complementary `StepScorer`
- **Why:** PRM scores reasoning quality but doesn't catch factual hallucinations. HaluGate adds a second scoring dimension: "is this step factually grounded?"
- **Implementation:** Implement `HaluGateScorer` as a `StepScorer` protocol adapter. Score = similarity between step claims and retrieved evidence. Combine with PRM score via weighted average in `PRMEvaluator`.

---

## Priority 3 — Production Readiness

### 3.1 Multi-Model Routing (Semantic Router)
- **What:** Route tasks to the best-fit model based on task type/complexity
- **Why:** A coding task should go to a code-tuned model; a writing task to a general model. Currently all tasks hit the same model.
- **Implementation:** Deploy Semantic Router as the `INFERENCE_BASE_URL`. Configure route rules per task type. The `InferenceClient` protocol already supports this — just point `INFERENCE_BASE_URL` at the router.

### 3.2 OpenClaw WebSocket Relay
- **What:** Bidirectional WebSocket Bridge ↔ OpenClaw session communication
- **Why:** Currently Bridge uses HTTP request/response. WebSocket enables streaming results, real-time training status, and push notifications to the UI.
- **Implementation:** Add `websockets` dependency. Extend `BridgeService` with WebSocket endpoint. OpenClaw agents connect via ws:// instead of polling HTTP.

### 3.3 Pin OpenClaw Docker Image
- **What:** Use a specific tagged release instead of `latest`
- **Why:** `latest` can break at any time. CI and demos need reproducibility.
- **Implementation:** Find a stable OpenClaw release tag, update `docker-compose.yml`.

### 3.4 Trajectory Data Pipeline
- **What:** Persist scored trajectories to a queryable store (SQLite → PostgreSQL)
- **Why:** Trained PRM (1.1) needs a large dataset. Currently rollouts are ephemeral in NATS. Need a durable store that accumulates data across demo runs.
- **Implementation:** Add a `TrajectoryStore` that subscribes to `training.rollouts` and persists to SQLite. Export script for PRM training dataset generation.

---

## Priority 4 — Business Use Case Enablement

### 4.1 Domain-Specific Agent Templates
- **What:** Pre-built OpenClaw agent configs (SOUL.md + SKILL.md) for top business use cases
- **Why:** Users shouldn't have to write agent prompts from scratch. Ship templates for: customer support, code review, knowledge worker, content pipeline.
- **Templates to build:**
  - `config/templates/support-agent/` — ticket triage, troubleshooting steps, escalation rules
  - `config/templates/code-review-agent/` — PR review, bug detection, style enforcement
  - `config/templates/knowledge-agent/` — internal Q&A, source citation, confidence scoring
  - `config/templates/content-agent/` — drafting, brand voice adherence, revision cycles

### 4.2 Evaluation Dashboard
- **What:** Web UI showing PRM scores over time, training loss curves, model version history, agent performance trends
- **Why:** Business users need to see that agents are improving. "Loss went from 0.8 to 0.3 over 500 tasks" is a compelling story. Currently there's no visibility into training progress outside logs.
- **Implementation:** Lightweight dashboard (Streamlit or plain HTML + Chart.js) reading from TrajectoryStore (2.4). Show: score distributions per agent, per task type, per model version.

### 4.3 Human-in-the-Loop Feedback API
- **What:** HTTP endpoint for human reviewers to submit corrections/approvals on agent outputs
- **Why:** Manager-as-LLM scoring bootstraps the system, but human feedback is ground truth. Business teams need a way to correct agent reasoning and have those corrections feed into training.
- **Implementation:** Extend Bridge API with `POST /feedback/human` endpoint. Human feedback overrides LLM-judge scores in the training pipeline. Weight human-sourced rollouts higher in GRPO batches.

### 4.4 Multi-Tenant Isolation
- **What:** Namespace NATS topics and LoRA adapters per tenant/team
- **Why:** Enterprise deployments need isolation — team A's training data shouldn't affect team B's model.
- **Implementation:** Prefix all NATS topics with tenant ID. LoRA checkpoints stored per tenant. InferenceClient routes to tenant-specific adapter.

---

## Priority 5 — Competitive Moat

### 5.1 Benchmark Suite
- **What:** Reproducible benchmarks comparing agentic-employees against static agent frameworks
- **Why:** Claims like "agents improve over time" need proof. Run identical task sets on agentic-employees vs. CrewAI/LangGraph/AutoGen, measure quality at task 1 vs. task 100 vs. task 1000.
- **Benchmarks:**
  - MATH (step-by-step problem solving) — PRM advantage is clearest here
  - HumanEval (code generation) — compare before/after GRPO training
  - Customer support simulation — measure resolution rate improvement over time
  - Multi-step reasoning (GSM8K, ARC) — show step-level scoring catches errors earlier

### 5.2 Blog Posts / Case Studies
- **What:** Write-ups demonstrating concrete improvement metrics
- **Topics:**
  - "How Our Agents Improved 40% on MATH After 500 Training Steps"
  - "PRM vs ORM: Why Scoring Each Step Matters (With Data)"
  - "Zero-Downtime Model Updates: LoRA Hot-Swap in Production"
  - "From Ollama to vLLM: Scaling Inference Without Code Changes"

### 5.3 Plugin Ecosystem
- **What:** Allow users to add custom `StepScorer` implementations, custom `Trainer` backends, and custom event handlers
- **Why:** The protocol-based architecture (StepScorer, Trainer, InferenceClient) already supports this. Formalize it as a plugin system so the community can extend without forking.
- **Implementation:** Document the protocol interfaces. Add a `plugins/` directory with example scorers and trainers. Setuptools entry points for discovery.

---

## Priority 6 — Scale & Operations

### 6.1 Kubernetes Deployment
- **What:** Helm chart for production deployment on K8s
- **Why:** Docker Compose works for demos but not for production. K8s gives autoscaling, rolling updates, and resource limits.
- **Components:** NATS (use nats-operator), Ollama/vLLM (GPU node pools), Bridge (Deployment + Service), Training (Job or Deployment with GPU).

### 6.2 Observability Stack
- **What:** Structured logging, Prometheus metrics, distributed tracing
- **Metrics to expose:**
  - `prm_step_score` histogram — score distribution per task type
  - `grpo_loss` gauge — training loss over time
  - `lora_swap_duration_seconds` — hot-swap latency
  - `nats_messages_total` counter — event throughput per topic
  - `task_latency_seconds` histogram — end-to-end task time

### 6.3 CI/CD Pipeline
- **What:** GitHub Actions for lint, test, Docker build, integration test
- **Steps:**
  1. `ruff check` — linting
  2. `pytest tests/ -v -k "not slow"` — fast unit tests
  3. Docker build (Bridge + Training images)
  4. `docker compose up` + `pytest tests/ -v` — full integration
  5. Push images to GHCR on tag

---

## Sequencing (Updated Post-Phase 7)

```
Now ──────────────────────────────────────────────────────────────► Later

[1.3 Majority Voting] ──► [1.4 Trained PRM] ──► [5.1 Benchmarks]
         (trivial)                                      │
                                                        ▼
[3.4 Trajectory Store] ──► [1.6 Token OPD] ──► [1.2 Implicit Signals]
                                                  [5.2 Case Studies]
[3.3 Pin OpenClaw] ──► [4.1 Agent Templates]
                                                  [5.3 Plugin Ecosystem]
[1.5 Turn Classification] ──► [4.3 Human Feedback API]

[2.1 DAPO (remaining)] ──► [6.1 K8s Deployment]  [6.3 CI/CD]

[3.1 Semantic Router] ──► [4.4 Multi-Tenant]

[3.2 WebSocket Relay] ──► [1.2 Implicit Signals]
                     └──► [4.2 Dashboard]

[2.2 HaluGate] ──────────────────────────► [6.2 Observability]
```

**Phase 7 unblocked:**
- 1.1 OPD ✅ — foundation for 1.6 (token-level OPD) and 1.2 (implicit signals)
- 2.1 Asymmetric clipping ✅ — DAPO remaining = dynamic sampling + entropy bonus
- CombinedScorer ✅ — HaluGate (2.2) can plug in as a named scorer
- Intercept Proxy ✅ — unblocks 1.5 (turn classification via SessionEvent)
- Meta-RL ✅ — manager training loop operational, needs data volume

**Start with:**
1. **1.3 Majority Voting** — trivial change to `LLMJudgeScorer`, immediate reliability improvement
2. **3.4 Trajectory Store** — unblocks Trained PRM (1.4) and token-level OPD (1.6)
3. **1.5 Turn Classification** — use SessionEvent data from intercept proxy to filter noise
4. **2.2 HaluGate** — plug into CombinedScorer as a named scorer, no new infrastructure needed
