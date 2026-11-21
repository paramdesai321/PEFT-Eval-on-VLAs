import os

os.environ["MUJOCO_GL"] = "egl"

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.default import EvalConfig
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.scripts.lerobot_eval import eval_main

policy_cfg = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")
policy_cfg.pretrained_path = "HuggingFaceVLA/smolvla_libero"
policy_cfg.num_steps = 10  # SmolVLA steps
policy_cfg.n_action_steps = 10  # actions steps
policy_cfg.device = "cuda"

cfg = EvalPipelineConfig(
    policy=policy_cfg,
    env=LiberoEnvConfig(
        task="libero_goal",
        task_ids=[0],
        init_states=True,
        hard_reset=False,
    ),
    eval=EvalConfig(n_episodes=1, batch_size=2),
)

result = eval_main(cfg)
print(result)
