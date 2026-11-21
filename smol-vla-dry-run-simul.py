import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.scripts.lerobot_eval import eval_main

policy_cfg = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")

policy_cfg.num_steps = 10
policy_cfg.n_action_steps = 10
policy_cfg.device = "cuda"

cfg = EvalPipelineConfig(
    policy=policy_cfg,
    env=LiberoEnvConfig(
        task="libero_goal",
        task_ids=[0],
        init_states=True,
        hard_reset=False,
    ),
    eval=EvalConfig(
        n_episodes=1,
        batch_size=1,
    ),
)

# Run evaluation
eval_main(cfg)

# Load evaluation statistics
results_path = Path(cfg.output_dir) / "eval_info.json"

with open(results_path, "r") as f:
    eval_results = json.load(f)

# Print overall statistics
overall = eval_results["overall"]
print(overall)
