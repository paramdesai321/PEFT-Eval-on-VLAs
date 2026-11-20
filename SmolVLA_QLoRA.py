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


