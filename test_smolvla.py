import torch
from lerobot.policies.smolvla import SmolVLAPolicy

# Use Apple Silicon GPU ('mps') or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model weights from Hugging Face
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device)
policy.eval()

print("SmolVLA loaded successfully!")
