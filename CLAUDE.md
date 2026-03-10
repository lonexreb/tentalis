# agentic-employees

Managers managing AI agents with appraisal-style feedback loops. Agents learn continuously from manager feedback using RLHF/GRPO — like performance reviews that actually improve performance.

## Architecture

Event-Driven Architecture (EDA):
- **Control Plane**: OpenClaw (identity, memory, channels, UI)
- **Event Broker**: NATS (pub/sub for agent coordination)
- **Training Plane**: OpenRLHF + OpenClaw-RL (GRPO/DAPO/AgentPRM)
- **Inference**: vLLM + Ray for distributed serving

## Language

**Python only.** All components — RL training, agent logic, NATS clients, PRM evaluator — are Python.

## Key Dependencies

- [OpenClaw](https://github.com/openclaw/openclaw) — control plane (identity, memory, channels)
- [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) — continuous learning from feedback (async GRPO)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — heavy RL training (PPO/GRPO/DAPO/REINFORCE++)
- [PicoClaw](https://github.com/sipeed/picoclaw) — future edge deployment (<10MB RAM)
- [M³HF](https://github.com/cooperativex/M3HF) — multi-phase feedback from mixed-quality humans
- [nats-py](https://github.com/nats-io/nats.py) — NATS client for Python
- Ray, vLLM, DeepSpeed — distributed compute and inference

## Directory Structure

```
src/
├── manager/    # Manager agent logic (task routing, evaluation, feedback)
├── workers/    # Worker agent implementations
├── events/     # NATS event bus, topic definitions, serialization
├── training/   # RL training loops (GRPO, DAPO, OpenRLHF integration)
└── rewards/    # PRM evaluator, reward functions, scoring
config/
└── openclaw/   # SOUL.md, IDENTITY.md templates per agent
tests/          # Mirrors src/ structure
docs/
└── architecture/  # Diagrams and ADRs
```

## Code Style

- PEP 8
- Type hints on all function signatures
- Docstrings on public APIs only (not internal helpers)
- No unnecessary abstractions — keep it simple

## Testing

- Framework: pytest
- `tests/` mirrors `src/` structure (e.g., `tests/manager/` tests `src/manager/`)
- Run: `pytest tests/`

## Commit Format

Conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code restructuring
- `test:` adding/updating tests
- `chore:` maintenance tasks

## Key Documents

- [PLAN.md](./PLAN.md) — Full technical research & architecture bible (papers, analysis, decisions)
- [LEARNING.md](./LEARNING.md) — Mistake/lesson tracking for autonomous decisions
