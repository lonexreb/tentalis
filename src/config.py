from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    nats_url: str = os.environ.get("NATS_URL", "nats://localhost:4222")
    manager_id: str = os.environ.get("MANAGER_ID", "manager-01")
    worker_id: str = os.environ.get("WORKER_ID", "worker-01")
    task_timeout_seconds: float = float(os.environ.get("TASK_TIMEOUT_SECONDS", "30"))
    llm_model: str = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    training_model: str = os.environ.get("TRAINING_MODEL", "distilgpt2")
    training_lr: float = float(os.environ.get("TRAINING_LR", "1e-4"))
    training_clip_epsilon: float = float(os.environ.get("TRAINING_CLIP_EPSILON", "0.2"))
    training_kl_beta: float = float(os.environ.get("TRAINING_KL_BETA", "0.1"))
    training_checkpoint_dir: str = os.environ.get("TRAINING_CHECKPOINT_DIR", "checkpoints")
    training_lora_rank: int = int(os.environ.get("TRAINING_LORA_RANK", "8"))
    training_batch_size: int = int(os.environ.get("TRAINING_BATCH_SIZE", "8"))
    training_group_size: int = int(os.environ.get("TRAINING_GROUP_SIZE", "4"))
    training_device: str = os.environ.get("TRAINING_DEVICE", "cpu")
