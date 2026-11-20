import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader


try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

#Config
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

def get_model_footprint_mb(model:nn.Module) -> float: 
param_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
buffer_bytes = sum(b.numel()*b.element_size() for b in model.bufferes())
return (param_bytes + buffer_bytes)/(1024**2) # converting from bytes to megabytes

def replace_linear_with_4bit(module:nn.Module, compute_dtype=tircg.bfloat16) -> int: 
    # Recursively replaces nn.linear layers with bitsandbytes Linear4bit, all the nn.linear(K,Q,V,MLP) layers
    replaced_count = 0
    for name, child in module.named_children():
        
        if isinstance(child, nn.Linear):
            quant_linear = bnb.nnLinear4bit(
                input_features=child.in_features,
                output_features=child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=True, # Double quantization
                quant_type="nf4"
            )
            quant_linear.load_state_dict(child.state_dict())
            setattr(module, name, quant_linear)
            replaced_count +=1
        
        else:
            replaced_count += replace_linear_with_4bit(child, compute_dytpe=compute_dtype)
    return replaced_count

policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
inital_mb = get_model_footprint_mb(policy)
print(f"Base FP32/BF16 model footprint: {initial_mb:.2f} MB")

use_qlora = (DEVICE.type == "cuda") and HAS_BNB
if use_qlora:
    #Quantize VLM + Expert Linear layers to 4bit bit keeping action heads in 16bit(full precision)
    num_quantized = replace_linear_with_4bit(policy.model.vlm_with_expert)
    print(f"Replaced {num_quantized} linear laeyers with 4bit NF4")
    policy = policy.to(DEVICE)
else:
    print(
        f"[Warning] QLoRA 4-bit NF4 requires an NVIDIA CUDA GPU with bitsandbytes installed "
        f"(detected device='{DEVICE.type}', bitsandbytes={HAS_BNB}). Falling back to standard LoRA."
    )
    policy = policy.to(DEVICE)


for param in policy.model.vlm_with_expert_parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    taret_modules=["q_proj","v_proj","k_proj","out_proj"]
    lora_dropoout=0.05
    bias="none",
)

policy.model.vlm_with_expert = get_peft_model(policy.model.vlm_with_expert, lora_config)


if use_qlora:
    for name, param in policy.model.vlm_with_expert.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
policy.eval()

quantized_mb = get_model_footprint_mb(policy)
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in policy.parameters())
print(f"Post-PEFT model footprint: {quantized_mb:.2f} MB")
print(
    f"Trainable params: {trainable_params:,} | Total params: {total_params:,} "
    f"| Trainable%: {100 * trainable_params / total_params:.4f}%"
)

dataset = LeRobotDataset(DATASET_ID)
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    MODEL_ID,
    preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

policy_input_keys = list(policy.config.input_features.keys())
mean_squared_error_losses = []
l1_losses = []

for step, batch in enumerate(dataloader):
    device_batch = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    processed_batch = {}
    for key, tensor in device_batch.items():
        if key in policy_input_keys:
            processed_batch[key] = tensor
        elif "images" in key:
            expected_img_keys = [k for k in policy_input_keys if "images" in k]
            if expected_img_keys:
                processed_batch[expected_img_keys[0]] = tensor

    current_batch_size = next(iter(processed_batch.values())).shape[0]
    if (
        "task" not in processed_batch
        and "observation.language.tokens" not in processed_batch
    ):
        processed_batch["task"] = ["pick up the target object"] * current_batch_size

    input_batch = preprocess(processed_batch)

    # Use bfloat16 autocast on CUDA to match Linear4bit compute_dtype
    with torch.inference_mode(), torch.amp.autocast(
        device_type=DEVICE.type, dtype=torch.bfloat16, enabled=use_qlora
    ):
        raw_actions = policy.select_action(input_batch)
        final_actions = postprocess(raw_actions)

    ground_truth_actions = batch.get("action")
    if ground_truth_actions is not None:
        ground_truth = ground_truth_actions.to(final_actions.device).float()
        pred = final_actions[:, 0, :].float() if final_actions.ndim == 3 else final_actions.float()
        ground_truth = ground_truth[:, : pred.shape[-1]]

        mean_squared_error = torch.mean((pred - ground_truth) ** 2).item()
        l1_loss = torch.mean(torch.abs(pred - ground_truth)).item()
        mean_squared_error_losses.append(mean_squared_error)
        l1_losses.append(l1_loss)

        print(
            f"Batch {step + 1} | MSE: {mean_squared_error:.5f} | L1: {l1_loss:.5f} "
            f"| running MSE: {np.mean(mean_squared_error_losses):.5f}"
        )

    if step >= (NUMBER_OF_BATCHES_TO_ITERATE - 1):
        break

print("\n--- QLoRA Evaluation Successful ---")
print(f"Offline metrics over {len(mean_squared_error_losses)} batches:")
print(f"  Mean Squared Error Loss: {np.mean(mean_squared_error_losses):.5f}")
print(f"  Mean Absolute Error Loss : {np.mean(l1_losses):.5f}")
