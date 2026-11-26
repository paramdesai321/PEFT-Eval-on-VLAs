import json
import math
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

BASE_MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"
OUTPUT_DIR = Path("./outputs/smolvla_qlora_libero")

BATCH_SIZE = 16          # Adjust to 8 or 16 on RTX 3080 (10GB VRAM) with grad_accum
GRAD_ACCUM_STEPS = 4     # Effective batch size = 64 (matches SmolVLA paper)
MAX_STEPS = 10_000       # 10k-20k optimizer steps for full LIBERO; use 500 for smoke test
LEARNING_RATE = 1e-4     # Standard LoRA/QLoRA LR (10x higher than full fine-tuning)
WARMUP_STEPS = 500
SAVE_EVERY_STEPS = 2500
LOG_EVERY_STEPS = 50

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


TRAINABLE_HEAD_PREFIXES = (
    "model.state_proj",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp_in",
    "model.action_time_mlp_out",
)

def replace_linear_with_4bit(module: nn.Module, skip_prefixes: tuple[str, ...], prefix: str = ""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if any(full_name.startswith(p) for p in skip_prefixes):
            continue
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            has_bias = child.bias is not None

            new_layer = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=has_bias,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type="nf4",
            )
            new_layer.weight = bnb.nn.Params4bit(
                child.weight.data.cpu(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            )
            if has_bias:
                new_layer.bias = nn.Parameter(child.bias.data.to(torch.bfloat16))
            setattr(module, name, new_layer)
        else:
            replace_linear_with_4bit(child, skip_prefixes, full_name)

def build_delta_timestamps(policy_cfg,fps:int) -> dict[str, list[float]]:

    delta_timestamps = {}
    if getattr(policy_cfg, "action_delta_indices", None) is not None:
        delta_timesteps["action"] = [i / fps for i in policy_cfg.action_delta_indices]
    if getattr(policy_cfg, "observation_delta_indices", None):
        for key in policy_cfg.image_features:
                delta_timestamps[key] = [i / fps for i in policy_cfg.observation_delta_indices]
   return delta_timestamps

def save_qlora_checkpoint(policy: SmolVLAPolicy, preprocess, postprocess, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    # 1. Save LoRA adapter weights + adapter_config.json
    policy.model.vlm_with_expert.save_pretrained(save_dir / "lora_adapter")

    # 2. Save unquantized projection heads
    head_state = {
        k: v.detach().cpu()
        for k, v in policy.state_dict().items()
        if any(k.startswith(p) for p in TRAINABLE_HEAD_PREFIXES)
    }
    torch.save(head_state, save_dir / "trainable_heads.pt")

    # 3. Save policy config & processor pipelines (with LIBERO dataset stats)
    policy.config.save_pretrained(save_dir)
    preprocess.save_pretrained(save_dir)
    postprocess.save_pretrained(save_dir)
    print(f"Saved QLoRA checkpoint to {save_dir}")

def main():
    assert torch.cuda.is_available(), "QLoRA training requires a CUDA GPU (e.g., RTX 3080)."
    device = torch.device("cuda")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load pretrained base policy
    print(f"Loading base policy: {BASE_MODEL_ID}")
    policy = SmolVLAPolicy.from_pretrained(BASE_MODEL_ID)
    policy.config.n_action_steps = 10  # Match closed-loop eval chunk execution

    # 2. Load LIBERO dataset with action-chunk delta timestamps
    temp_ds = LeRobotDataset(DATASET_ID)
    delta_timestamps = build_delta_timestamps(policy.config, temp_ds.fps)
    dataset = LeRobotDataset(DATASET_ID, delta_timestamps=delta_timestamps)

    # 3. Build preprocessor/postprocessor using LIBERO dataset statistics
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        BASE_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
