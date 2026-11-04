import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
import numpy as np
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "lerobot/smolvla_base"

# Load model & metadata
policy = SmolVLAPolicy.from_pretrained(model_id).to(
    device=device,
    dtype=torch.float32
)
policy.eval()

dataset_meta = LeRobotDatasetMetadata("lerobot/svla_so100_stacking")

ds_features = dataset_meta.features
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    dataset_stats=dataset_meta.stats,
    preprocessor_overrides={"device_processor": {"device": str(device)}}
)

# Inputs
task = "pick up the red block"
state_names = ds_features["observation.state"]["names"]
observation = {name: 0.0 for name in state_names}
print("State dimensions:", state_names)

for key, feature in ds_features.items():
    if key.startswith("observation.images."):
        camera_name = key.removeprefix("observation.images.")
        observation[camera_name] = np.random.randint(
            0, 256, size=(480, 640, 3), dtype=np.uint8
        )

print("Raw observation keys:", list(observation))

# Build frame and process
frame = build_inference_frame(
    observation=observation,
    ds_features=ds_features,
    device=device,
    task=task,
)
processed_frame = preprocess(frame)
processed_frame["observation.images.camera1"] = processed_frame.pop(
    "observation.images.top"
)
processed_frame["observation.images.camera2"] = processed_frame.pop(
    "observation.images.wrist"
)

with torch.no_grad():
    raw_action = policy.select_action(processed_frame)

action = postprocess(raw_action)
robot_action = make_robot_action(action, ds_features)
print("Action tensor shape:", tuple(action.shape))
print("Named robot action:\n", robot_action)

from peft import LoraConfig, get_peft_model

# Define LoRA parameters
peft_config = LoraConfig(
    r=8,                           # Low rank matrix dimension
    lora_alpha=16,                 # Scaling factor
    target_modules=["q_proj", "v_proj"], # Apply to attention projections inside SmolVLA
    lora_dropout=0.05,
    bias="none",
)

# Apply LoRA to SmolVLA
peft_policy = get_peft_model(policy, peft_config)

# Check trainable parameter count
peft_policy.print_trainable_parameters()
