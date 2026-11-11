import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

from peft import LoraConfig, get_peft_model, inject_adapter_in_model

# Swap this import per-policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import warnings
import os

# Suppress all python warnings and torch deprecation alerts
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# load a policy
model_id = "lerobot/smolvla_base"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
# print("policy.model: ", policy.model)

# print("dir(policy.model): ", dir(policy.model))


# policy.model.vlm_with_expert = get_peft_model(policy.model.vlm_with_expert, lora_config)
# policy.model.vlm_with_expert.print_trainable_parameters()
inject_adapter_in_model(lora_config, policy.model.vlm_with_expert)
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in policy.parameters())
print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")


print("\nInput shape:")
for key, feat in policy.config.input_features.items():
    print(f" - {key}: shape {feat.shape}")

batch = {
    "observation.images.camera1": torch.randn(1, 3, 256, 256, device=device),
    "observation.images.camera2": torch.randn(1, 3, 256, 256, device=device),
    "observation.images.camera3": torch.randn(1, 3, 256, 256, device=device),
    "observation.state": torch.zeros(1, policy.config.input_features["observation.state"].shape[0], device=device),
    "observation.language.tokens": torch.zeros((1, 16), dtype=torch.long, device=device),
    "observation.language.attention_mask": torch.ones((1, 16), dtype=torch.bool, device=device),
}

print("\nRunning inference")
with torch.inference_mode():
    pred_action = policy.select_action(batch)
    print("Success! Predicted action output from dummy input:", pred_action)
    
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
# load a lerobotdataset
dataset = LeRobotDataset("lerobot/libero")
print("LeRobotDataset: ", dataset)
# pick an episode
episode_index = 0

# each episode corresponds to a contiguous range of frame indices
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]

# get a single frame from that episode (e.g. the first frame)
frame_index = from_idx
frame = dict(dataset[frame_index])

batch = preprocess(frame)
with torch.inference_mode():
    pred_action = policy.select_action(frame)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    pred_action = postprocess(pred_action)
    print("Predicted action output from lerobotdataset: ", pred_action)