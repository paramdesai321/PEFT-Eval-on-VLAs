import os
import warnings
import platform

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "cgl" if platform.system() == "Darwin" else "egl"


try:
    import bitsandbytes as bnb

    HAS_BNB = True
except ImportError:
    HAS_BNB = False

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Config
MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"
BATCH_SIZE = 8
NUMBER_OF_BATCHES_TO_ITERATE = 5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")


def get_model_footprint_mb(model: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (
        1024**2
    )  # converting from bytes to megabytes


def replace_linear_with_4bit(module: nn.Module, compute_dtype=torch.bfloat16) -> int:
    # Recursively replaces nn.linear layers with bitsandbytes Linear4bit, all the nn.linear(K,Q,V,MLP) layers
    replaced_count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            quant_linear = bnb.nnLinear4bit(
                input_features=child.in_features,
                output_features=child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=True,  # Double quantization
                quant_type="nf4",
            )
            quant_linear.load_state_dict(child.state_dict())
            setattr(module, name, quant_linear)
            replaced_count += 1

        else:
            replaced_count += replace_linear_with_4bit(
                child, compute_dytpe=compute_dtype
            )
    return replaced_count


policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
initial_mb = get_model_footprint_mb(policy)
print(f"Base FP32/BF16 model footprint: {initial_mb:.2f} MB")

use_qlora = (DEVICE.type == "cuda") and HAS_BNB
if use_qlora:
    # Quantize VLM + Expert Linear layers to 4bit bit keeping action heads in 16bit(full precision)
    num_quantized = replace_linear_with_4bit(policy.model.vlm_with_expert)
    print(f"Replaced {num_quantized} linear laeyers with 4bit NF4")
    policy = policy.to(DEVICE)
else:
    print(
        f"[Warning] QLoRA 4-bit NF4 requires an NVIDIA CUDA GPU with bitsandbytes installed "
        f"(detected device='{DEVICE.type}', bitsandbytes={HAS_BNB}). Falling back to standard LoRA."
    )
    policy = policy.to(DEVICE)


for param in policy.model.vlm_with_expert.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)

policy.model.vlm_with_expert = get_peft_model(policy.model.vlm_with_expert, lora_config)


if use_qlora:
    for name, param in policy.model.vlm_with_expert.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)

env_cfg = LiberoEnvConfig(
    task="libero_spatial,libero_object,libero_goal,libero_10",
    task_ids=None,  # None = evaluate all 10 tasks per suite (40 tasks total)
    init_states=True,  # Standard fixed initial states
    hard_reset=False,
    control_mode="relative",
)
envs = make_env(env_cfg, n_envs=1, use_async_envs=False)


# 3. Create policy & environment pre/post-processors
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    MODEL_ID,
    preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
)

env_preprocess, env_postprocess = make_env_pre_post_processors(
    env_cfg=env_cfg,
    policy_cfg=policy.config,
)
# 4. Run closed-loop evaluation (10 episodes per task = 400 total rollouts)
policy.eval()
eval_info = eval_policy_all(
    envs=envs,
    policy=policy,
    env_preprocessor=env_preprocess,
    env_postprocessor=env_postprocess,
    preprocessor=preprocess,
    postprocessor=postprocess,
    n_episodes=10,
    max_episodes_rendered=2,
    videos_dir=Path("./eval_logs/qlora_videos"),
    start_seed=42,
)

print("Overall Success Rate (%):", eval_info["overall"]["pc_success"])
