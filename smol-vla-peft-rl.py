import os
import warnings

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["MUJOCO_GL"] = "egl"

# Config and Device setup
CORES = os.cpu_count()
MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"  # Drop-in replacement dataset
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

BATCH_SIZE = 8
print(f"Using device: {DEVICE}")

# Policy
policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(DEVICE)

# LORA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)

policy.model.vlm_with_expert = get_peft_model(policy.model.vlm_with_expert, lora_config)

trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in policy.parameters())
print(
    f"Trainable params: {trainable_params:,} | Total params: {total_params:,} | Trainable%: {100 * trainable_params / total_params:.4f}%"
)

env_cfg = LiberoEnvConfig(
    task="libero_goal",  # libero_spatial, libero_goal, libero_10, libero_object, libero_90,
    task_ids=[0],
    init_states=True,
    hard_reset=False,
    control_mode="relative",
)
