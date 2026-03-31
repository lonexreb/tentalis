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
    # PRM majority voting (Phase 9)
    prm_num_votes: int = int(os.environ.get("PRM_NUM_VOTES", "3"))
    # SkillRL (Phase 9)
    skills_enabled: bool = os.environ.get("SKILLS_ENABLED", "false").lower() == "true"
    skills_dir: str = os.environ.get("SKILLS_DIR", "skills_data")
    skill_evolution_threshold: float = float(os.environ.get("SKILL_EVOLUTION_THRESHOLD", "0.4"))
    skill_retrieval_top_k: int = int(os.environ.get("SKILL_RETRIEVAL_TOP_K", "3"))
    # Tinker training backend (Phase 9)
    tinker_api_key: str = os.environ.get("TINKER_API_KEY", "")
    tinker_base_url: str = os.environ.get("TINKER_BASE_URL", "https://api.tinker.thinkingmachines.ai")
    # Training scheduler (Phase 9)
    training_schedule_enabled: bool = os.environ.get("TRAINING_SCHEDULE_ENABLED", "false").lower() == "true"
    training_schedule_hours: str = os.environ.get("TRAINING_SCHEDULE_HOURS", "02:00-06:00")
    # Alignment experiments (Phase 9)
    alignment_enabled: bool = os.environ.get("ALIGNMENT_ENABLED", "false").lower() == "true"
    alignment_results_dir: str = os.environ.get("ALIGNMENT_RESULTS_DIR", "alignment_results")
    alignment_audit_all: bool = os.environ.get("ALIGNMENT_AUDIT_ALL", "false").lower() == "true"
    # Trajectory store (Phase 9c)
    trajectory_store_enabled: bool = os.environ.get("TRAJECTORY_STORE_ENABLED", "false").lower() == "true"
    trajectory_store_path: str = os.environ.get("TRAJECTORY_STORE_PATH", "trajectory_data/trajectories.db")
    # HaluGate scorer (Phase 9c)
    halugate_enabled: bool = os.environ.get("HALUGATE_ENABLED", "false").lower() == "true"
    halugate_model: str = os.environ.get("HALUGATE_MODEL", "qwen2.5:1.5b")
    # CISPO contrastive loss (Phase 9c)
    cispo_enabled: bool = os.environ.get("CISPO_ENABLED", "false").lower() == "true"
    cispo_weight: float = float(os.environ.get("CISPO_WEIGHT", "0.2"))
    cispo_margin: float = float(os.environ.get("CISPO_MARGIN", "0.5"))
    # DAPO (Phase 9c)
    dapo_entropy_beta: float = float(os.environ.get("DAPO_ENTROPY_BETA", "0.01"))
    dapo_min_reward_threshold: float = float(os.environ.get("DAPO_MIN_REWARD_THRESHOLD", "0.1"))
    # Trained PRM (Phase 9c)
    trained_prm_enabled: bool = os.environ.get("TRAINED_PRM_ENABLED", "false").lower() == "true"
    trained_prm_model: str = os.environ.get("TRAINED_PRM_MODEL", "Qwen/Qwen2.5-0.5B")
    trained_prm_checkpoint: str = os.environ.get("TRAINED_PRM_CHECKPOINT", "")
    # Benchmarks (Phase 9c)
    benchmark_dataset_dir: str = os.environ.get("BENCHMARK_DATASET_DIR", "benchmark_data")
    benchmark_results_dir: str = os.environ.get("BENCHMARK_RESULTS_DIR", "benchmark_results")
