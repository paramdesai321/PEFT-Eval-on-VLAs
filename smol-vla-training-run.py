import json
import os
import numpy as np
import torch
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.scripts.lerobot_eval import eval_main
from peft import LoraConfig, get_peft_model



# Policy
MODEL_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
policy_config = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")

policy_config.num_steps = 10
policy_config.n_action_steps = 10
policy_config.device = "cuda"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ],
    bias="none",
)
policy = get_peft_model(policy_config, lora_config)

policy.print_trainable_parameters()


# train with AdamW optimizaer
optimizer = torch.optim.AdamW(
    [p for p in policy.parameters() if p.requires_grad],
    lr=1e-5,
)


# training not evaluation
# cfg = EvalPipelineConfig(
#     policy=policy_config,
#     env=LiberoEnvConfig(
#         task="libero_goal",
#         task_ids=[0],
#         init_states=True,
#         hard_reset=False,
#     ),
#     eval=EvalConfig(
#         n_episodes=1,
#         batch_size=1,
#     ),
# )

# # Run evaluation
# eval_main(cfg)