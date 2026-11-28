import json
import math
from pathlib import Path
import sys

import bitsandbytes as bnb
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

BASE_MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"
OUTPUT_DIR = Path("./outputs/smolvla_qlora_libero")

BATCH_SIZE = 16  # Adjust to 8 or 16 on RTX 3080 (10GB VRAM) with grad_accum
GRAD_ACCUM_STEPS = 4  # Effective batch size = 64 (matches SmolVLA paper)
MAX_STEPS = 10_000  # 10k-20k optimizer steps for full LIBERO; use 500 for smoke test
LEARNING_RATE = 1e-4  # Standard LoRA/QLoRA LR (10x higher than full fine-tuning)
WARMUP_STEPS = 500
SAVE_EVERY_STEPS = 2500
LOG_EVERY_STEPS = 50

DRY_RUN = "--dry-run" in sys.argv
EPISODES = [0, 1, 2, 3] if DRY_RUN else None

if DRY_RUN:
    OUTPUT_DIR = Path("./outputs/smolvla_qlora_dry_run")
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 2
    MAX_STEPS = 20
    WARMUP_STEPS = 5
    SAVE_EVERY_STEPS = 20
    LOG_EVERY_STEPS = 5

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


def replace_linear_with_4bit(
    module: nn.Module, skip_prefixes: tuple[str, ...], prefix: str = ""
):
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if any(full_name.startswith(p) for p in skip_prefixes):
            continue

        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            has_bias = child.bias is not None
            quant_storage = child.weight.dtype

            newly_quantized_layer = bnb.nn.Linear4bit(
                input_features=in_features,
                output_features=out_features,
                bias=has_bias,
                compute_dtype=child.weight.dtype,
                quant_type="nf4",
                quant_storage=quant_storage,
            )

            newly_quantized_layer.weight = bnb.nn.Params4bit(
                child.weight.detach().clone(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
                quant_storage=quant_storage,
            )

            if has_bias:
                newly_quantized_layer.bias = nn.Parameter(
                    child.bias.detach().clone(),
                    requires_grad=child.bias.requires_grad,
                )

            setattr(module, name, newly_quantized_layer)

        else:
            replace_linear_with_4bit(child, skip_prefixes, full_name)


# def build_delta_timestamps(policy_cfg, fps: int) -> dict[str, list[float]]:

#     delta_timestamps = {}
#     if getattr(policy_cfg, "action_delta_indices", None) is not None:
#         delta_timestamps["action"] = [i / fps for i in policy_cfg.action_delta_indices]
#     if getattr(policy_cfg, "observation_delta_indices", None):
#         for key in policy_cfg.image_features:
#             delta_timestamps[key] = [
#                 i / fps for i in policy_cfg.observation_delta_indices
#             ]
#     return delta_timestamps


def build_delta_timestamps(policy_cfg, fps: float):
    chunk_size = getattr(policy_cfg, "chunk_size", None)

    if chunk_size is None:
        chunk_size = getattr(policy_cfg, "n_action_steps", 50)

    return {"action": [step_idx / fps for step_idx in range(chunk_size)]}


def save_qlora_checkpoint(
    policy: SmolVLAPolicy, preprocess, postprocess, save_dir: Path
):
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
    assert torch.cuda.is_available(), (
        "QLoRA training requires a CUDA GPU (e.g., RTX 3080)."
    )
    device = torch.device("cuda")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load pretrained base policy
    print(f"Loading base policy: {BASE_MODEL_ID}")
    policy = SmolVLAPolicy.from_pretrained(BASE_MODEL_ID)
    policy.config.n_action_steps = 10  # Match closed-loop eval chunk execution

    # 2. Load LIBERO dataset with action-chunk delta timestamps
    # temp_ds = LeRobotDataset(DATASET_ID)
    # delta_timestamps = build_delta_timestamps(policy.config, temp_ds.fps)
    # dataset = LeRobotDataset(DATASET_ID, delta_timestamps=delta_timestamps)
    temp_ds = LeRobotDataset(DATASET_ID, episodes=EPISODES)
    delta_timestamps = build_delta_timestamps(policy.config, temp_ds.fps)
    dataset = LeRobotDataset(
        DATASET_ID, episodes=EPISODES, delta_timestamps=delta_timestamps
    )

    # 3. Build preprocessor/postprocessor using LIBERO dataset statistics
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        BASE_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("Quantizing backbone Linear layers to 4-bit NF4...")
    replace_linear_with_4bit(policy, skip_prefixes=TRAINABLE_HEAD_PREFIXES)

    # 5. Freeze base parameters, inject LoRA into VLM + Action Expert, unfreeze I/O heads
    for param in policy.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        # task_type=TaskType.FEATURE_EXTRACTION, LETS SEE WHAT HAPPENS
    )
    policy.model.vlm_with_expert = get_peft_model(
        policy.model.vlm_with_expert, lora_config
    )

    for name, param in policy.named_parameters():
        if any(name.startswith(p) for p in TRAINABLE_HEAD_PREFIXES):
            param.data = param.data.to(torch.bfloat16)
            param.requires_grad = True

    policy.to(device)
    policy.model.vlm_with_expert.print_trainable_parameters()

    # 6. Optimizer & cosine LR schedule
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)

    def lr_lambda(step: int):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(max(1, WARMUP_STEPS))
        progress = float(step - WARMUP_STEPS) / float(max(1, MAX_STEPS - WARMUP_STEPS))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # 7. Training loop
    policy.train()
    optimizer.zero_grad()
    opt_step = 0
    micro_step = 0
    running_loss = 0.0
    RENAME_MAP = {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }

    print(f"Starting QLoRA training for {MAX_STEPS} optimizer steps...")
    while opt_step < MAX_STEPS:
        for raw_batch in dataloader:
            batch = preprocess(raw_batch)

            for src_key, dst_key in RENAME_MAP.items():
                if src_key in batch and dst_key not in batch:
                    batch[dst_key] = batch[src_key]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)
                scaled_loss = loss / GRAD_ACCUM_STEPS

            scaled_loss.backward()
            running_loss += loss.item()
            micro_step += 1

            for src_key, dst_key in RENAME_MAP.items():
                if src_key in batch and dst_key not in batch:
                    batch[dst_key] = batch[src_key]

            if micro_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step % LOG_EVERY_STEPS == 0:
                    avg_loss = running_loss / (LOG_EVERY_STEPS * GRAD_ACCUM_STEPS)
                    vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    print(
                        f"Step {opt_step}/{MAX_STEPS} | "
                        f"Flow-Matching Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Peak VRAM: {vram_gb:.2f} GB"
                    )
                    running_loss = 0.0

                if opt_step % SAVE_EVERY_STEPS == 0:
                    save_qlora_checkpoint(
                        policy, preprocess, postprocess, OUTPUT_DIR / f"step_{opt_step}"
                    )

                if opt_step >= MAX_STEPS:
                    break

            # Debugging
            print(
                f"micro_step={micro_step}, opt_step={opt_step}, loss={loss.item():.4f}",
                flush=True,
            )

    save_qlora_checkpoint(
        policy, preprocess, postprocess, OUTPUT_DIR / "checkpoint_final"
    )


if __name__ == "__main__":
    main()
