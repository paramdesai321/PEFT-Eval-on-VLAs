import os
import json
from pathlib import Path
import warnings

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.default import EvalConfig
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.scripts.lerobot_eval import eval_main
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from torch.utils.data import DataLoader
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Config and Device setup
CORES = os.cpu_count()
# MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"  # Drop-in replacement dataset
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Policy
MODEL_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
policy = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")
policy_cfg = PreTrainedConfig.from_pretrained(MODEL_CHECKPOINT)
policy_cfg.device = str(DEVICE)

# Instantiate model and load pretrained weights
policy = SmolVLAPolicy.from_pretrained(
    MODEL_CHECKPOINT,
    config=policy_cfg,
)

policy.device = "cuda"
BATCH_SIZE = 8
print(f"Using device: {DEVICE}")

# LORA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)

policy.model.vlm_with_expert = get_peft_model(policy.model.vlm_with_expert, lora_config)
policy.eval()


trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in policy.parameters())
print(
    f"Trainable params: {trainable_params:,} | Total params: {total_params:,} | Trainable%: {100 * trainable_params / total_params:.4f}%"
)

# Load Dataset and Preprocessors
dataset = LeRobotDataset(DATASET_ID)
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    MODEL_CHECKPOINT,
    preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
)
# Create Dataloader

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# Batch Inference loop

# Dynamic Camera/State Key mapping (Note: needed to Ensure dataset keys match what SmolVLA expects in input_features) for Batches

policy_input_keys = list(policy.config.input_features.keys())

NUMBER_OF_BATCHES_TO_ITERATE = 5
mean_squared_error_losses = []
l1_losses = []
for step, batch in enumerate(dataloader):
    # Move tensor features in batch to target device
    device_batch = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            device_batch[key] = val.to(DEVICE)
        else:
            device_batch[key] = val

    # Dynamic Key Remapping for SmolVLA
    processed_batch = {}
    for key, tensor in device_batch.items():
        if key in policy_input_keys:
            processed_batch[key] = tensor
        elif "images" in key:
            expected_img_keys = [k for k in policy_input_keys if "images" in k]
            if expected_img_keys:
                processed_batch[expected_img_keys[0]] = tensor

    # Inject batch of language task prompts if missing
    current_batch_size = next(iter(processed_batch.values())).shape[0]
    if (
        "task" not in processed_batch
        and "observation.language.tokens" not in processed_batch
    ):
        processed_batch["task"] = ["pick up the target object"] * current_batch_size
    # Preprocessing and Model Inference
    input_batch = preprocess(processed_batch)
    with torch.inference_mode():
        raw_actions = policy.select_action(input_batch)
        final_actions = postprocess(raw_actions)

    print(
        f"Batch {step + 1} | Input Batch Size: {current_batch_size} | Action Output Shape: {final_actions.shape}"
    )

    # Accumulate Statistics Across the dataset
    # Ground-truth actions from the dataset
    ground_truth_actions = batch.get("action")
    if ground_truth_actions is not None:
        ground_truth = ground_truth_actions.to(final_actions.device).float()
        # Align shapes if select_action returns a chunk:
        # final_actions is often (B, chunk, A); gt is (B, A).
        # Use only the first predicted action, or slice gt to match.
        if final_actions.ndim == 3:
            pred = final_actions[:, 0, :]  # first step of chunk
            ground_truth = ground_truth[:, : pred.shape[-1]]  # match action dim
        else:
            pred = final_actions
            ground_truth = ground_truth[:, : pred.shape[-1]]

        mean_squared_error = torch.mean((pred - ground_truth) ** 2).item()
        l1_loss = torch.mean(torch.abs(pred - ground_truth)).item()
        mean_squared_error_losses.append(mean_squared_error)
        l1_losses.append(l1_loss)

    print(
        f"Batch {step + 1} | MSE: {mean_squared_error:.5f} | L1: {l1_loss:.5f} "
        f"| running MSE: {np.mean(mean_squared_error):.5f}"
    )

    """
    !!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!
    PLEASE COMMENT THIS LINE TO RUN THROUHGH THE ENTIRE DATASET.
    NUMBER_OF_BATCHES_TO_ITERATE is set to a limited number for demo purposes.
    """
    if step >= (NUMBER_OF_BATCHES_TO_ITERATE - 1):
        break

# Inference Statistics
print("\n--- Inference Successful ---")
print(f"\nOffline metrics over {len(mean_squared_error_losses)} batches:")
print(f"  Mean Squared Error Loss: {np.mean(mean_squared_error_losses):.5f}")
print(f"  Mean Absolute Error Loss : {np.mean(l1_losses):.5f}")
