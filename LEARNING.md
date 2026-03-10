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

## Tool Usage

| Date | Context | Mistake | Root Cause | Lesson | Prevention |
|------|---------|---------|------------|--------|------------|

## Debugging

| Date | Context | Mistake | Root Cause | Lesson | Prevention |
|------|---------|---------|------------|--------|------------|

## User Preferences

| Date | Context | Mistake | Root Cause | Lesson | Prevention |
|------|---------|---------|------------|--------|------------|
