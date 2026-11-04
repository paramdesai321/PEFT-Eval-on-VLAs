import torch
from lerobot.policies.smolvla import SmolVLAPolicy
from utils.device_configuration import get_device

device = get_device()

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device)
policy.eval()

# Construct direct batch (Images normalized 0.0 to 1.0, shape: [B, C, H, W])
mock_batch = {
    "observation.images.top": torch.rand(1, 3, 480, 640, device=device),
    "observation.state": torch.zeros(1, 6, device=device),
    "task": ["pick up the red block"],
}

with torch.no_grad():
    action = policy.select_action(mock_batch)

print("Direct batch action output shape:", action.shape)
