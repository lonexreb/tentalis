# Learning Log

Living document for tracking mistakes and lessons learned during autonomous development. Every mistake is a training signal.

## How to Use

When an autonomous decision turns out wrong or suboptimal, log it here with root cause and prevention strategy. Review before making similar decisions in the future.

---

## Architecture

| Date | Context | Mistake | Root Cause | Lesson | Prevention |
|------|---------|---------|------------|--------|------------|
| 2026-03-09 | Initial research | Sub-agents couldn't use tools | Assumed sub-agents inherit parent tool permissions | Sub-agents need explicit tool permissions — they don't inherit parent permissions | Always specify tool access when delegating to sub-agents |
| 2026-03-09 | Architecture selection | Could have jumped to a recommendation without analysis | Bias toward action over research | Always do deep research before proposing architecture — thoroughness > speed | Require comparative analysis (min 3 options) before any architecture decision |
| 2026-03-09 | Presenting findings | Could have presented only the recommended option | Anchoring bias — presenting one strong option | Present comparative analysis with trade-offs, not just recommendations — user wants to understand the reasoning | Always include a comparison table with pros/cons for each option |

## Code Patterns

| Date | Context | Mistake | Root Cause | Lesson | Prevention |
|------|---------|---------|------------|--------|------------|
| 2026-03-09 | Phase 2 pyproject.toml | Set `requires-python = ">=3.11"` but dev machine runs Python 3.10.0 | Copied from plan without checking local Python version | Always check `python3 --version` before setting requires-python | Run `python3 --version` before writing pyproject.toml |
| 2026-03-12 | Phase 4 GRPO math | `variance == 0.0` check failed for identical float inputs (0.7, 0.7, 0.7) — variance was ~1e-32 | IEEE 754 floating point: `sum([0.7,0.7,0.7])/3` is not exactly 0.7 | Never compare floats with `== 0.0`; use tolerance checks (`< 1e-12`) | Use epsilon comparisons for all float equality checks |
| 2026-03-12 | Phase 4 tests | `pytest.importorskip()` at module level skips the *entire* module, not just tests after it | pytest imports the whole module before collecting tests; importorskip raises Skip during import | Use `pytest.mark.skipif` on classes/functions for partial-module skipping; reserve `importorskip` for whole-module gating | When mixing torch/no-torch tests in one file, use `skipif` decorator pattern |

## Architecture (continued)

| Date | Context | Mistake | Root Cause | Lesson | Prevention |
|------|---------|---------|------------|--------|------------|
| 2026-03-13 | Phase 7 OPD design | Could have built OPD as a monolithic component | Tendency to build large integrated systems | Splitting OPD into 3 components (HintExtractor, CombinedRolloutBuilder, CombinedTrainer) keeps each testable and replaceable independently | Follow single-responsibility: one component per concern, joined by NATS events |
| 2026-03-13 | Phase 7 CombinedRolloutBuilder | Considered using a database for join state | Over-engineering instinct | Simple in-memory dict with asyncio timeout is sufficient — the join window is short (30s) and data volume is low. No external state needed. | Start with in-memory; add persistence only when data loss is actually a problem |
| 2026-03-13 | Phase 7 target_worker_id | Considered separate topics per worker for model updates | Topic proliferation makes subscriptions complex | Single `model.updates` topic with `target_worker_id` filter in BaseWorker is simpler and more flexible — broadcast by default, targeted when needed | Use payload-level filtering over topic multiplication when the message rate is low |

## Tool Usage

| Date | Context | Mistake | Root Cause | Lesson | Prevention |
|------|---------|---------|------------|--------|------------|

## Debugging

| Date | Context | Mistake | Root Cause | Lesson | Prevention |
|------|---------|---------|------------|--------|------------|

## User Preferences

| Date | Context | Mistake | Root Cause | Lesson | Prevention |
|------|---------|---------|------------|--------|------------|
