import torch
from lerobot.policies.smolvla import SmolVLAPolicy
# Local project imports
from utils.device_configuration import get_device
device = get_device()

print(f"Using device: {device}")

# Load model weights from Hugging Face
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device)
policy.eval()

print("SmolVLA loaded successfully!")
