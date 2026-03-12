# Deep Research: Technical Stack, Architecture & State-of-the-Art for OpenClaw + OpenClaw-RL

## Part 1: The Papers — State of the Art

### Core RL Algorithms (Ranked by Relevance)

| Algorithm | Paper | Key Innovation | Best For |
|-----------|-------|----------------|----------|
| **GRPO** | [DeepSeekMath (2402.03300)](https://arxiv.org/abs/2402.03300) | Eliminates critic network; group-relative advantage = `(reward_i - mean) / std` | Memory-efficient agent training |
| **DAPO** | [ByteDance (2503.14476)](https://arxiv.org/abs/2503.14476) | Clip-Higher + Dynamic Sampling + Token-Level Loss; **50pts on AIME** beating DeepSeek-R1-Zero | Large-scale RL at production |
| **REINFORCE++** | [Cameron Wolfe writeup](https://cameronrwolfe.substack.com/p/reinforce) | PPO tricks without critic; ProRL V2 used it for SOTA 1.5B reasoning model | Lightweight RL training |
| **AgentPRM** | [Process Reward Models for LLM Agents (2502.10325)](https://arxiv.org/abs/2502.10325) | Monte Carlo rollouts for step-level rewards; **3B model beats GPT-4o** on ALFWorld | Multi-step agent tasks |
| **MARLHF** | [Multi-Agent RLHF (OpenReview)](https://openreview.net/forum?id=4vPC6Aj6N7) | First systematic study of multi-agent RLHF; proves single-policy coverage is inadequate; Nash equilibrium from preference data | Multi-agent alignment theory |
| **M³HF** | [ICML 2025 (2503.02077)](https://arxiv.org/abs/2503.02077) | Multi-phase feedback from mixed-quality humans (expert + non-expert); LLM parses feedback → reward updates | Your "manager appraisal" vision |
| **"GRPO is Secretly a PRM"** | [OpenReview paper](https://openreview.net/pdf/1f109913a199dad205fa51f554aaa5e2a5a782d5.pdf) | Proves GRPO implicitly induces process reward model behavior | Theory connecting GRPO↔PRM |

### The Key Insight: PRM > ORM for Agents

From [OpenAI's "Let's Verify Step by Step"](https://arxiv.org/abs/2502.10325): **Process supervision trains much more reliable reward models than outcome supervision** (78.2% on MATH vs significantly lower for ORM). For your manager-agent system, this means:

- **Don't just score final output** (outcome reward)
- **Score each step** (process reward) — like a manager reviewing each task milestone, not just the final deliverable

---

## Part 2: Critical Architecture Analysis — OpenClaw's Limitations

### The Node.js Single-Thread Problem

This is **the most critical finding** from the research. OpenClaw's Gateway has a documented architectural bottleneck:

> When calling tools from within an active LLM session, the tool call opens a new WebSocket connection back to the **same gateway that is currently busy processing the session's turn**. Since the gateway is **single-threaded Node.js**, it cannot respond to the second WS request while blocked on the first — resulting in a **10-second timeout**. ([Issue #6508](https://github.com/openclaw/openclaw/issues/6508))

**Impact on your vision:**
- Per-session serialization is by design (lane-aware FIFO queue)
- Cross-session parallelism works, but **one Gateway per host** (WhatsApp constraint)
- For manager overseeing 10+ worker agents simultaneously → you'll hit contention

### OpenClaw Performance Profile

| Metric | Value | Bottleneck |
|--------|-------|-----------|
| Gateway RAM | <1GB | Fine |
| Core package | ~8MB | Fine |
| Throughput | Dozens of short chats/sec | **LLM inference is the limiter** |
| GPU utilization | 72% → 89% (batch 4→16) | Needs tuning |
| Effective throughput | 45 → 63 tasks/min (with quantization) | Moderate |
| Scaling model | 1 Gateway per host | **Horizontal scaling constraint** |

---

## Part 3: Software Architecture Decision

### The Three Candidates

Based on software architecture fundamentals (coupling, cohesion, scalability, fault tolerance):

#### Option A: Monolithic OpenClaw (Current Default)
```
Single Gateway → All Agents → Single Event Loop
```
- **Pros**: Simple, fast to start, built-in identity/memory/skills
- **Cons**: Single-threaded bottleneck, can't scale horizontally, self-contention on WebSocket
- **Verdict**: Good for prototyping, **not for production multi-agent with RL training**

#### Option B: Microservices (Traditional)
```
API Gateway → Agent Service A → Agent Service B → Training Service
                    ↓                    ↓
              Shared Database      Message Queue
```
- **Pros**: Independent scaling, technology diversity
- **Cons**: Distributed monolith risk, synchronous coupling, complex debugging
- **Verdict**: Better, but the ["distributed monolith" antipattern](https://technode.global/2025/09/22/beware-the-distributed-monolith-why-agentic-ai-needs-event-driven-architecture-to-avoid-a-repeat-of-the-microservices-disaster/) is the #1 failure mode for agentic AI

#### Option C: Event-Driven Architecture (EDA) — RECOMMENDED
```
Event Broker (Redis Streams / NATS / Kafka)
    ├── Manager Agent (publishes tasks, subscribes to results)
    ├── Worker Agent A (subscribes to tasks, publishes results)
    ├── Worker Agent B (subscribes to tasks, publishes results)
    ├── PRM Evaluator (subscribes to completions, publishes scores)
    ├── RL Trainer (subscribes to scored rollouts, publishes weight updates)
    └── OpenClaw Gateway(s) (per-host, handles identity/memory/channels)
```

### Why EDA Wins

| Principle | EDA Advantage |
|-----------|--------------|
| **Loose coupling** | Agents don't call each other directly; they publish/subscribe events |
| **Async by nature** | LLM responses vary ms→minutes; EDA queues events instead of blocking |
| **Fault tolerance** | If worker crashes, events queue; manager continues with others |
| **Horizontal scaling** | Add workers by subscribing to the same event stream |
| **Observability** | Every interaction is an event with timestamp + context |
| **Dynamic registration** | New agents join by subscribing; no redeployment needed |
| **RL training decoupled** | Training runs in background consuming scored rollout events |

---

## Part 4: Recommended Technical Stack

### The Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CONTROL PLANE                      │
│  OpenClaw Gateway (identity, memory, channels, UI)   │
│  Per-host · Node.js · WebSocket :18789               │
└──────────────────────┬──────────────────────────────┘
                       │ Events
┌──────────────────────▼──────────────────────────────┐
│              EVENT BROKER (NATS / Redis Streams)     │
│  Topics: tasks.*, results.*, feedback.*, training.*  │
└──┬────────┬────────┬────────┬────────┬──────────────┘
   │        │        │        │        │
┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──────────────┐
│Mgr  │ │Wkr A│ │Wkr B│ │Wkr N│ │  RL TRAINING     │
│Agent│ │Agent│ │Agent│ │Agent│ │  PLANE           │
│     │ │     │ │     │ │     │ │                  │
│SOUL │ │SOUL │ │SOUL │ │SOUL │ │ OpenRLHF/        │
│IDENT│ │IDENT│ │IDENT│ │IDENT│ │ OpenClaw-RL      │
│MEMORY││MEMORY││MEMORY││MEMORY││                  │
└─────┘ └─────┘ └─────┘ └─────┘ │ • PRM Evaluator  │
                                  │ • GRPO/DAPO      │
                                  │ • Ray + vLLM     │
                                  │ • DeepSpeed ZeRO │
                                  └──────────────────┘
```

### Technology Choices

| Layer | Technology | Why |
|-------|-----------|-----|
| **Event Broker** | **NATS** (start) → Kafka (scale) | NATS: 10M+ msg/sec, <1ms latency, zero config. Kafka when you need replay/durability |
| **Control Plane** | **OpenClaw Gateway** | Identity (SOUL.md), memory, channel adapters, UI — already built |
| **Manager Agent** | **OpenClaw agent** with custom skills | Evaluates worker output, publishes feedback events, routes tasks |
| **Worker Agents** | **OpenClaw agents** (or PicoClaw for edge) | Each subscribes to task topics, publishes results |
| **RL Training** | **OpenRLHF** | Production-ready; Ray + vLLM + DeepSpeed; supports PPO/GRPO/DAPO/REINFORCE++ |
| **Process Reward Model** | **AgentPRM approach** | Step-level scoring via Monte Carlo rollouts; 3B model beats GPT-4o |
| **Feedback Integration** | **OpenClaw-RL** (async GRPO) | Turns manager feedback into training signals without interrupting serving |
| **Inference** | **vLLM** | 80% of RLHF time is generation; vLLM with AutoTP handles this |
| **Distributed Compute** | **Ray** | Separates Actor/Reward/Reference/Critic across GPUs |
| **Persistence** | **Redis** (state) + **S3/local** (transcripts) | Session state + JSONL transcript storage |

### Scaling Path

```
Phase 1: Prototype (1 machine)
├── 1 OpenClaw Gateway
├── NATS (embedded)
├── 1 Manager + 3 Workers (same host)
├── OpenClaw-RL with small model (7B)
└── Cost: $15/mo VPS + API costs

Phase 2: Production (multi-node)
├── 1+ OpenClaw Gateways (per-host)
├── NATS cluster (3 nodes)
├── N Worker agents (auto-scaling)
├── OpenRLHF on GPU nodes (Ray cluster)
└── Cost: ~$200-500/mo

Phase 3: Enterprise (Kubernetes)
├── K8s: CPU pods (Gateway) + GPU pods (inference/training)
├── Kafka (event replay, audit trail)
├── PicoClaw edge nodes reporting to manager
├── M³HF-style multi-phase feedback (expert + non-expert managers)
└── Horizontal auto-scaling on all layers
```

### The RL Training Loop (Your "Appraisal" System)

```
1. Manager publishes task → tasks.coding topic
2. Worker A picks up task, generates solution
3. Worker A publishes result → results.coding topic
4. Manager reviews result, publishes feedback → feedback.scored topic
   (thumbs up/down + textual corrections)
5. PRM Evaluator scores each step → training.rollouts topic
   (Monte Carlo rollouts for step-level rewards)
6. RL Trainer consumes scored rollouts:
   - GRPO: group-relative advantage across workers
   - DAPO: dynamic sampling + token-level loss
   - AgentPRM: step-level process rewards
7. Weight update published → model.updates topic
8. Workers hot-swap weights (graceful update)
9. Repeat — agents get better with every interaction
```

### Algorithm Selection Guide

| Scenario | Algorithm | Why |
|----------|-----------|-----|
| Starting out, limited GPU | **GRPO** | No critic needed, memory efficient |
| Need max performance | **DAPO** | SOTA results, dynamic sampling |
| Resource constrained (1.5B model) | **REINFORCE++** | Proven for small reasoning models |
| Multi-step agent tasks | **AgentPRM** | Step-level rewards beat outcome-only |
| Multiple managers with varying expertise | **M³HF** | Handles mixed-quality feedback |

---

## Part 5: Key Architectural Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture pattern | **Event-Driven** | Avoids distributed monolith; async matches LLM latency reality |
| Orchestration | **OpenClaw** (control plane only) | Use its identity/memory/channels; don't rely on it for agent coordination |
| Agent coordination | **NATS pub/sub** | Decouples agents; enables horizontal scaling |
| RL framework | **OpenRLHF** (training) + **OpenClaw-RL** (live feedback) | OpenRLHF for heavy training, OpenClaw-RL for continuous personalization |
| Reward model | **Process (PRM)** over Outcome (ORM) | 78.2% vs lower on MATH; step-level feedback = better agent learning |
| Primary RL algo | **GRPO → DAPO** (graduate) | Start simple (GRPO), scale to DAPO when you need peak performance |
| Edge deployment | **PicoClaw** (future) | <10MB RAM, 1s boot — lightweight workers on cheap hardware |

---

## Part 6: Phase 3 — LLM Workers, PRM Scoring & Training Bridge

### Status

- **Phase 1** (DONE): Scaffolding, docs, git, GitHub
- **Phase 2** (DONE): Event loop — manager publishes task, worker echoes, manager scores
- **Phase 3** (DONE): LLM workers (Ollama), PRM scoring (LLM-as-judge), training bridge (rollout buffer + JSONL)
- **Phase 4** (DONE): Standalone GRPO trainer (torch + LoRA + distilgpt2), TrainingLoop orchestrator, MockTrainer fallback, ModelUpdateEvent publishing
- **Phase 5** (NEXT): vLLM migration, weight hot-swap in workers, OpenRLHF integration (drop-in via Trainer protocol), Semantic Router as inference gateway

### Design Issue: ResultEvent Missing Prompt

The PRM evaluator needs the original prompt to score steps in context. `ResultEvent` doesn't carry it. **Fix: add `prompt` field to `ResultEvent`** — small duplication but keeps the evaluator stateless (no need to track/cache TaskEvents).

---

### 3A: LLM Worker (Ollama for Prototype)

**Decision: Ollama + AsyncClient** for prototype. Runs on CPU, async-native, zero config.

| Option | Throughput | GPU Required | Async | Best For |
|--------|-----------|-------------|-------|----------|
| Ollama + AsyncClient | ~5 req/s (CPU) | No | Native | Prototype |
| vLLM + AsyncLLMEngine | ~50-200 req/s | Yes | Native | Production |
| llama-cpp-python | ~3-10 req/s | No | Wrap in thread | Minimal deps |
| OpenAI-compatible API | Varies | No | Via openai SDK | Cloud fallback |

**Graduation path:** Ollama → vLLM + LoRA (when GPU available) → vLLM + Ray + tensor parallel (multi-GPU)

**Structured output for PRM:** Use step markers in system prompt (`<step>...</step>`, `<answer>...</answer>`). Parse with regex. Graduate to Pydantic schema-constrained generation (Ollama `format=` param) when reliability matters.

**Step marker system prompt:**
```
You are a precise problem solver. Break your reasoning into clear numbered steps.
Format your response exactly as:
<step>1. [Your first reasoning step]</step>
<step>2. [Your second reasoning step]</step>
...
<answer>[Your final answer]</answer>
```

**Key insight:** The `BaseWorker` abstraction already supports drop-in replacement. `LLMWorker` extends `BaseWorker` exactly like `EchoWorker`, just with real LLM calls in `process()`.

---

### 3B: PRM Evaluator (LLM-as-Judge First)

**Decision: Start with LLM-as-judge**, accumulate data, train dedicated PRM later.

| Phase | Scorer | Latency/Step | Training Data Needed | Hardware |
|-------|--------|-------------|---------------------|----------|
| **Now** | LLM-as-judge (GPT-4o-mini or local 1.5B) | 200-500ms | None | CPU or API |
| **>10K trajectories** | Trained PRM (Qwen2.5-1.5B + classification head) | 50-100ms | 10K+ scored trajectories | 1x GPU |
| **Production** | Batched PRM via vLLM | 10-30ms amortized | 50K+ | 1x A100 |

**Graduation trigger:** When LLM-as-judge latency bottlenecks training throughput, or when you need calibrated/consistent scores. The judge's own scores become PRM training labels — bootstrap from judge to trained model.

**Architecture in event bus:**
```
results.* ──► PRMEvaluator ──► training.rollouts
                  │
                  ├── Subscribes to: results.{task_type}
                  ├── Receives: ResultEvent (with steps + prompt)
                  ├── Scores each step via StepScorer protocol
                  └── Publishes: TrainingRolloutEvent (with step_scores)
```

**StepScorer protocol** — two implementations behind same interface:
1. `LLMJudgeScorer`: calls an LLM to score each step on progress + correctness (0-1)
2. `TrainedPRMScorer` (future): runs a trained PRM model for fast inference

**Step scoring prompt pattern:**
```
Rate step {n} of an agent solving a task.
TASK: {prompt}
STEPS SO FAR: {steps[:n]}
CURRENT STEP: {steps[n]}
Rate PROGRESS (0-1) and CORRECTNESS (0-1). Respond with JSON only.
```

**What constitutes a "step" in agent context:**

| Agent Action | Step Boundary | Score Meaning |
|-------------|--------------|---------------|
| Tool call | Invocation + result | "Did this advance toward the goal?" |
| Reasoning block | Each CoT segment | "Is this reasoning sound?" |
| Code generation | Each code block | "Does this move toward correctness?" |

---

### 3C: Training Bridge (NATS → OpenRLHF/OpenClaw-RL)

**Decision: Use both frameworks** for different purposes.

| Role | Framework | When |
|------|-----------|------|
| Continuous learning (live feedback loop) | **OpenClaw-RL** | Every scored rollout |
| Heavy retraining (new model version) | **OpenRLHF** | Periodic batch runs |

**Critical finding: OpenRLHF's GRPO is batch-synchronous.** It cannot consume rollouts one-at-a-time. Need a `RolloutBuffer` adapter:

```
NATS (training.rollouts)
    │ scored rollouts arrive async
    ▼
RolloutBuffer
    │ buffers until batch_size * group_size ready
    ▼
NATSTrainingBridge
    │ feeds batch to trainer
    ▼
GRPOTrainer (OpenRLHF)
    │ gradient step
    ▼
model.updates topic (LoRA checkpoint path)
```

**GRPO data requirements per batch:**
- Each prompt needs `group_size` (4-8) responses with rewards
- Advantage = `(reward_i - mean) / std` within the group
- Minimum batch: 8-64 prompts × 4-8 responses each = 32-512 scored rollouts

**Resource requirements for 7B GRPO:**

| Config | GPUs | VRAM | Feasible? |
|--------|------|------|-----------|
| Full fine-tune, ZeRO-3, group=8 | 4x A100 | 320GB | Yes |
| LoRA (rank 64), ZeRO-2, group=4 | 1x A100 | 80GB | Yes |
| QLoRA, group=2 | 1x RTX 4090 | 24GB | Tight but works |

**Prototype recommendation:** Use 3B model (Phi-3-mini or Qwen2.5-3B) with LoRA on single GPU. Graduate to 7B on cloud GPUs.

**Weight hot-swap strategy: LoRA adapters**
- Base model stays loaded (never changes)
- LoRA adapters are tiny (~10-50MB)
- vLLM supports loading new LoRA at runtime without restart
- Workers subscribe to `model.updates`, pull new adapter, switch
- Filesystem convention: `/models/adapters/v001/`, `/models/adapters/v002/`, `current -> v002`

---

### 3D: Build Order

#### Step 1: Fix ResultEvent (add prompt field)
- Add `prompt: str = ""` to `ResultEvent` in `src/events/types.py`
- Update `BaseWorker._handle_task()` to populate it from `task.prompt`
- Update tests

#### Step 2: LLMWorker (`src/workers/llm_worker.py`)
- New dep: `ollama>=0.4` in `pyproject.toml`
- `LLMWorker(BaseWorker)`: uses `ollama.AsyncClient` in `process()`
- System prompt with step markers, regex parsing into `steps` list
- Config: add `llm_model`, `ollama_host` to `Config` dataclass

#### Step 3: StepScorer protocol + LLMJudgeScorer (`src/rewards/scorer.py`)
- `StepScorer` protocol: `async score_steps(prompt, steps) -> list[float]`
- `LLMJudgeScorer`: calls LLM with step-judge prompt, returns progress×correctness scores
- `src/rewards/prompts.py`: judge prompt templates

#### Step 4: PRMEvaluator (`src/rewards/prm_evaluator.py`)
- Subscribes to `results.*`, scores steps via `StepScorer`
- Publishes `TrainingRolloutEvent` to `training.rollouts`
- Stateless — relies on prompt being in `ResultEvent`

#### Step 5: Re-export in `src/rewards/__init__.py`

#### Step 6: RolloutBuffer + NATSTrainingBridge (`src/training/bridge.py`)
- `RolloutBuffer`: buffers scored rollouts, emits batches when ready
- `NATSTrainingBridge`: subscribes to `training.rollouts`, feeds batches to trainer
- Trainer interface is abstract — plug in OpenRLHF or OpenClaw-RL later

#### Step 7: Tests
- `tests/rewards/test_scorer.py` — mock LLM judge, verify scoring
- `tests/rewards/test_prm_evaluator.py` — mock scorer, verify event flow
- `tests/training/test_bridge.py` — buffer fill + batch emission
- `tests/test_integration.py` — extend full loop: task → LLM result → PRM score → rollout

#### Step 8: Update `__main__.py`
- Add LLMWorker option (fallback to EchoWorker if Ollama unavailable)
- Add PRMEvaluator to the demo loop
- Print step scores alongside results

### 3E: Files Summary

| Action | Path |
|--------|------|
| MODIFY | `pyproject.toml` (add `ollama>=0.4`) |
| MODIFY | `src/config.py` (add `llm_model`, `ollama_host`) |
| MODIFY | `src/events/types.py` (add `prompt` to `ResultEvent`) |
| MODIFY | `src/workers/base.py` (populate `result.prompt`) |
| CREATE | `src/workers/llm_worker.py` |
| CREATE | `src/rewards/scorer.py` |
| CREATE | `src/rewards/prompts.py` |
| CREATE | `src/rewards/prm_evaluator.py` |
| MODIFY | `src/rewards/__init__.py` (add re-exports) |
| CREATE | `src/training/bridge.py` |
| MODIFY | `src/training/__init__.py` (add re-exports) |
| CREATE | `tests/rewards/__init__.py` |
| CREATE | `tests/rewards/test_scorer.py` |
| CREATE | `tests/rewards/test_prm_evaluator.py` |
| CREATE | `tests/training/__init__.py` |
| CREATE | `tests/training/test_bridge.py` |
| MODIFY | `tests/test_integration.py` (extend for PRM) |
| MODIFY | `src/__main__.py` (add LLMWorker + PRM) |

**10 new files, 8 modified.**

### 3F: Commit Sequence

1. `feat: add prompt field to ResultEvent for PRM context`
2. `feat: add LLM worker with Ollama backend`
3. `feat: add PRM evaluator with LLM-as-judge scorer`
4. `feat: add training bridge with rollout buffer`
5. `test: add reward, training, and extended integration tests`
6. `feat: update demo entry point with LLM worker and PRM`

### 3G: Verification

1. `pip install -e ".[dev]"` — installs with ollama dep
2. `ollama serve & ollama pull qwen2.5:1.5b` — model ready
3. `pytest tests/rewards/ -v` — scorer + evaluator tests pass (mocked, no NATS)
4. `pytest tests/training/test_bridge.py -v` — buffer tests pass (no NATS)
5. With NATS + Ollama running:
   - `pytest tests/test_integration.py -v` — full loop with PRM scoring
   - `python -m src` — prints task → LLM response with steps → step scores → feedback

---

## Sources

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [OpenClaw-RL GitHub](https://github.com/Gen-Verse/OpenClaw-RL)
- [PicoClaw GitHub](https://github.com/sipeed/picoclaw)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [GRPO Technical Deep Dive](https://cameronrwolfe.substack.com/p/grpo)
- [DeepSeekMath GRPO Paper](https://arxiv.org/abs/2402.03300)
- [DAPO Paper](https://arxiv.org/abs/2503.14476)
- [AgentPRM Paper](https://arxiv.org/abs/2502.10325)
- [MARLHF Paper](https://openreview.net/forum?id=4vPC6Aj6N7)
- [M³HF ICML 2025](https://arxiv.org/abs/2503.02077)
- [M³HF GitHub](https://github.com/cooperativex/M3HF)
- [GRPO is Secretly a PRM](https://openreview.net/pdf/1f109913a199dad205fa51f554aaa5e2a5a782d5.pdf)
- [OpenClaw WS Contention Issue #6508](https://github.com/openclaw/openclaw/issues/6508)
- [EDA for Agentic AI](https://technode.global/2025/09/22/beware-the-distributed-monolith-why-agentic-ai-needs-event-driven-architecture-to-avoid-a-repeat-of-the-microservices-disaster/)
- [OpenClaw Architecture Deep Dive (Substack)](https://trilogyai.substack.com/p/deep-dive-openclaw)
- [OpenClaw Gateway Deep Dive](https://practiceoverflow.substack.com/p/deep-dive-into-the-openclaw-gateway)
- [OpenClaw Design Patterns](https://kenhuangus.substack.com/p/openclaw-design-patterns-part-1-of)
- [OpenClaw Benchmarks](https://markaicode.com/benchmark-openclaw-performance-cpu-gpu-cloud/)
- [OpenClaw $15 VPS Production Stack](https://medium.com/@rentierdigital/the-complete-openclaw-architecture-that-actually-scales-memory-cron-jobs-dashboard-and-the-c96e00ab3f35)

---

## Part 7: vLLM Semantic Router / ClawOS Research

### What ClawOS Actually Is

**ClawOS is NOT a standalone project.** It's a feature ("Claw Mode") within [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) — a system-level intelligent router for Mixture-of-Models (MoM).

#### vLLM Semantic Router (the infrastructure)

- Sits between users and models as an [Envoy external processor](https://arxiv.org/abs/2603.04444)
- **6 signal types** extracted from each request: Domain (MMLU LoRA classifiers), Keyword (regex), Embedding (neural similarity), Factual (hallucination), Feedback (user satisfaction), Preference (personalization)
- **Decision engine**: configurable Boolean AND/OR rules → selects best model from pool
- **Plugins**: semantic-cache, jailbreak detection, PII protection, hallucination detection (HaluGate 3-stage pipeline), system prompt injection
- **MoM model family**: specialized routing models (mom-brain-flash/pro/max, mom-similarity-flash, mom-jailbreak-flash, mom-pii-flash)
- **OpenAI API-compatible** — drop-in replacement for any OpenAI client
- **Tech stack**: Go (42.6%), Python (16.4%), Rust (15.1%), TypeScript (13.5%)
- **v0.1 Iris** (Jan 2026) released, **v0.2 Athena** in development (adds RADAR difficulty-aware routing)

#### ClawOS / Claw Mode (the agent orchestration UI)

- A specialized mode in the Semantic Router's Playground chat UI
- **ClawOS** = "the orchestration/control mode in this chat"
- **Claw Team** = organizational unit (exactly one leader required)
- **Claw Worker** = individual anthropomorphic agent with domain, persona, speaking style, responsibilities
- **Claw Manager** = LLM persona that builds teams and recruits workers
- Provisioning via **MCP tool calls**: `claw_create_team`, `claw_create_worker`
- Integrates with **OpenClaw** as backend agent platform
- Confirmation-before-execution for all mutating actions

### Mapping to Our Architecture

| Semantic Router Concept | Our Equivalent | Our File |
|---|---|---|
| Model routing (6 signals) | Hardcoded single model in `LLMWorker` | `src/workers/llm_worker.py` |
| OpenAI-compatible API | `ollama.AsyncClient` | `src/workers/llm_worker.py`, `src/rewards/scorer.py` |
| Semantic cache | None — every prompt re-generated | — |
| HaluGate (hallucination detection) | None | — |
| Jailbreak / PII protection | None | — |
| Claw Team | Workers sharing same `task_types` | `src/workers/base.py` |
| Claw Worker | `BaseWorker` subclass instances | `src/workers/base.py` |
| Claw Manager | `Manager` class | `src/manager/manager.py` |
| MCP tool provisioning | Workers instantiated in code | `src/__main__.py` |
| Difficulty-aware routing (RADAR) | No task difficulty concept | `src/events/types.py` TaskEvent |

### What Helps Us

#### 1. Semantic Router as Inference Gateway (HIGH VALUE)

Our `LLMWorker.process()` and `LLMJudgeScorer._judge_single_step()` both call Ollama directly. Routing through the Semantic Router would give us:

- **Multi-model routing** — route easy tasks to small models, hard tasks to large ones (directly relevant for GRPO where you generate `group_size` responses per prompt)
- **Semantic caching** — avoid redundant generation for similar prompts during batch training
- **Safety guardrails** — jailbreak detection, PII protection (needed once multi-tenant)
- **Multi-provider fallback** — single API for vLLM, Ollama, cloud APIs
- **HaluGate as complementary PRM signal** — a new `StepScorer` implementation using hallucination detection alongside LLM-as-judge

#### 2. Difficulty-Aware Routing Concept (MEDIUM VALUE, borrowable now)

The RADAR idea from v0.2 Athena: estimate query difficulty, route to appropriate model. We can implement a lightweight version by adding `difficulty: float` to `TaskEvent` and having the Manager estimate it heuristically.

### What Does NOT Help Us

- **ClawOS / Claw Mode UI** — we're a headless Python backend with RL training. A web playground for creating agent personas is irrelevant until we have a user-facing product. Our planned `config/openclaw/SOUL.md` templates are simpler and better fit.
- **MoM specialized models** (mom-brain-flash, mom-pii-flash, etc.) — these are the router's internal signal-extraction models, not the models we're training via GRPO/DAPO.
- **MCP-based provisioning** — our workers are instantiated in Python code. Adding MCP→NATS bridge for provisioning is massive over-engineering at this stage.
- **Envoy deployment** — production K8s/service-mesh infrastructure, irrelevant for Phase 4.
- **PII/Preference signals** — our prompts are synthetic training tasks, not user data.

### Conflicts

- **Worker orchestration ownership**: Our Manager coordinates via NATS pub/sub; ClawOS coordinates via MCP. These are different patterns — bridging them would require a translation layer.
- **Language mismatch**: We're Python-only; Semantic Router is Go/Rust/TypeScript/Python. We'd consume it as an external service, never modify it.

### Decision: Do NOT Integrate Now

**Phase 4 (RL trainer) is the critical path. Plan Semantic Router integration for Phase 5.**

Adding the Semantic Router now means: deploying a Go/Rust service alongside NATS and Ollama, new operational complexity, new debugging surface — all before we have a working training loop. Phase 4's bottleneck is `RolloutBuffer → OpenRLHF GRPO`, not inference routing.

### Phase 5 Integration Plan (after RL trainer works)

1. Deploy Semantic Router as standalone service
2. Add `inference_url` to `Config` (default `http://localhost:8080/v1`)
3. Create `OpenAICompatibleWorker` or modify `LLMWorker` to use `openai.AsyncOpenAI` instead of `ollama.AsyncClient`
4. Route all inference (worker generation + PRM scoring) through the router
5. Enable semantic caching for GRPO group generation
6. Optionally add HaluGate as complementary `StepScorer` implementation

### Sources

- [DeepWiki: Claw Mode and System Prompt](https://deepwiki.com/vllm-project/semantic-router/7.4-claw-mode-and-system-prompt)
- [GitHub: vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- [Paper: arxiv.org/abs/2603.04444](https://arxiv.org/abs/2603.04444)
- [vLLM SR v0.1 Iris blog](https://blog.vllm.ai/2026/01/05/vllm-sr-iris.html)
