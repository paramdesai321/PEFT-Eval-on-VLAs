import os
import warnings
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader


warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Config and Device setup

MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"  # Drop-in replacement dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
policy.eval()

trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in policy.parameters())
print(f"Trainable params: {trainable_params:,} | Total params: {total_params:,} | Trainable%: {100 * trainable_params / total_params:.4f}%")

# Load Dataset and Preprocessors
dataset = LeRobotDataset(DATASET_ID)
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    MODEL_ID,
    preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
)
# Create Dataloader 

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

#
## Fetch Sample Frame (Note: Needed to know the exact data specs like camera name, image res etc.)
#sample_frame = dict(dataset[0])
#
## Adding batch dim (Note: Standard VLA are like this [Batch_Size, Channels, Height, Width] etc.)
#batched_frame = {}
#for key, value in sample_frame.items():
#    if isinstance(value, torch.Tensor):
#        batched_frame[key] = value.unsqueeze(0).to(DEVICE)
#    else:
#        batched_frame[key] = value

# Batch Inference loop

# Dynamic Camera/State Key mapping (Note: needed to Ensure dataset keys match what SmolVLA expects in input_features) for Batches

policy_input_keys = list(policy.config.input_features.keys())

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
    if "task" not in processed_batch and "observation.language.tokens" not in processed_batch:
        processed_batch["task"] = ["pick up the target object"] * current_batch_size
# Preprocessing and Model Inference
    input_batch = preprocess(processed_batch)
    with torch.inference_mode():
        raw_actions = policy.select_action(input_batch)
        final_actions = postprocess(raw_actions)

    print(f"Batch {step + 1} | Input Batch Size: {current_batch_size} | Action Output Shape: {final_actions.shape}")
    if step >= 2:
        break
print("\n--- Inference Successful ---")
