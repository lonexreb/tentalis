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


def task_topic(task_type: str) -> str:
    return f"tasks.{task_type}"


def result_topic(task_type: str) -> str:
    return f"results.{task_type}"
