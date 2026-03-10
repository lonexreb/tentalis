# agentic-employees

A system where **managers manage agentic employees** that learn from feedback — like performance appraisals that actually improve performance. Built on reinforcement learning from human feedback (RLHF), agents continuously improve through manager evaluations using GRPO/DAPO algorithms.

## Architecture

```
┌──────────────────────────────────┐
│   OpenClaw (Control Plane)       │
│   Identity · Memory · Channels   │
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│   NATS Event Broker              │
│   tasks.* · results.* · feedback.*│
└──┬───────┬───────┬───────┬───────┘
   │       │       │       │
 Manager  Wkr A  Wkr B  RL Trainer
 Agent    Agent  Agent   (OpenRLHF)
```

**Manager** assigns tasks → **Workers** execute → **Manager** reviews & scores → **RL Trainer** improves workers from feedback → repeat.

## Getting Started

> Phase 1 (current): Project scaffolding and documentation. Implementation coming in Phase 2.

```bash
# Prerequisites: Python 3.11+, NATS server
pip install -e .  # (coming soon)
```

## Documentation

- **[PLAN.md](./PLAN.md)** — Full technical research & architecture bible (papers, analysis, decisions)
- **[CLAUDE.md](./CLAUDE.md)** — Project conventions for Claude Code
- **[LEARNING.md](./LEARNING.md)** — Mistake/lesson tracking log
