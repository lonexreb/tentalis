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
    # Inference abstraction (Phase 5)
    inference_backend: str = os.environ.get("INFERENCE_BACKEND", "ollama")
    inference_base_url: str = os.environ.get("INFERENCE_BASE_URL", "")
    inference_api_key: str = os.environ.get("INFERENCE_API_KEY", "")
    vllm_lora_name: str = os.environ.get("VLLM_LORA_NAME", "default")
    # Training backend (Phase 5)
    trainer_backend: str = os.environ.get("TRAINER_BACKEND", "standalone")
    # Bridge service (Phase 6 — OpenClaw integration)
    bridge_port: int = int(os.environ.get("BRIDGE_PORT", "8100"))
    openclaw_gateway_url: str = os.environ.get("OPENCLAW_GATEWAY_URL", "ws://localhost:18789")
    # Intercept proxy (Phase 7)
    intercept_enabled: bool = os.environ.get("INTERCEPT_ENABLED", "false").lower() == "true"
    intercept_port: int = int(os.environ.get("INTERCEPT_PORT", "8200"))
    intercept_backend_url: str = os.environ.get("INTERCEPT_BACKEND_URL", "http://localhost:11434")
    # OPD (On-Policy Distillation)
    opd_teacher_model: str = os.environ.get("OPD_TEACHER_MODEL", "qwen2.5:1.5b")
    opd_join_timeout: float = float(os.environ.get("OPD_JOIN_TIMEOUT", "30.0"))
    opd_weight: float = float(os.environ.get("OPD_WEIGHT", "0.3"))
    rl_weight: float = float(os.environ.get("RL_WEIGHT", "0.7"))
    # OPD mode: "lightweight" (our LLM hint extraction) or "openclaw" (vLLM logprobs)
    opd_mode: str = os.environ.get("OPD_MODE", "lightweight")
    # Asymmetric clipping
    training_clip_epsilon_high: float = float(os.environ.get("TRAINING_CLIP_EPSILON_HIGH", "0.28"))
    # Meta-RL
    meta_rl_enabled: bool = os.environ.get("META_RL_ENABLED", "false").lower() == "true"
    meta_rl_window_size: int = int(os.environ.get("META_RL_WINDOW_SIZE", "200"))
    meta_rl_min_feedback: int = int(os.environ.get("META_RL_MIN_FEEDBACK", "200"))
