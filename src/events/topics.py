from __future__ import annotations

TASKS_CODING = "tasks.coding"
RESULTS_CODING = "results.coding"
FEEDBACK_SCORED = "feedback.scored"
TRAINING_ROLLOUTS = "training.rollouts"
TRAINING_COMBINED = "training.combined"
MODEL_UPDATES = "model.updates"

# Intercept proxy session logging
SESSIONS = "sessions.logged"

# On-Policy Distillation
OPD_HINTS = "opd.hints"

# Manager meta-RL
META_ROLLOUTS = "meta.rollouts"
META_UPDATES = "meta.updates"

# SkillRL
SKILLS_CREATED = "skills.created"

# Session lifecycle
SESSION_START = "sessions.start"
SESSION_END = "sessions.end"

# Alignment experiments
ALIGNMENT_EVALS = "alignment.evals"
ALIGNMENT_AUDIT = "alignment.audit"


def task_topic(task_type: str) -> str:
    return f"tasks.{task_type}"


def result_topic(task_type: str) -> str:
    return f"results.{task_type}"
