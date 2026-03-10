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
