import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import importlib.util
import sys
import types

lora_sp_root = Path("/home/jet/UCDavis/ECS_271/LoRA-SP")
if str(lora_sp_root) not in sys.path:
    sys.path.insert(0, str(lora_sp_root))

# Ensure 'common' and 'common.policies' exist as package entries without running
# common/policies/__init__.py (which eagerly imports pi0 and configs/policies.py).
if "common" not in sys.modules:
    common_pkg = types.ModuleType("common")
    common_pkg.__path__ = [str(lora_sp_root / "common")]
    sys.modules["common"] = common_pkg

if "common.policies" not in sys.modules:
    policies_pkg = types.ModuleType("common.policies")
    policies_pkg.__path__ = [str(lora_sp_root / "common" / "policies")]
    sys.modules["common.policies"] = policies_pkg

lora_msp_path = lora_sp_root / "common" / "policies" / "lora_msp.py"
spec = importlib.util.spec_from_file_location("common.policies.lora_msp", lora_msp_path)
lora_msp_mod = importlib.util.module_from_spec(spec)
sys.modules["common.policies.lora_msp"] = lora_msp_mod
spec.loader.exec_module(lora_msp_mod)

LoraMSPConfig = lora_msp_mod.LoraMSPConfig
LoraMSPLinear = lora_msp_mod.LoraMSPLinear

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        return device, local_rank, True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, False


def apply_lora_sp(model: torch.nn.Module, target_prefixes: tuple[str, ...], target_keywords: tuple[str, ...], msp_cfg: LoraMSPConfig):
    # Freeze all base weights first
    for param in model.parameters():
        param.requires_grad = False

    # Walk named modules and replace matching nn.Linear layers in selected submodules
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            in_scope = (not target_prefixes or name.startswith(target_prefixes))
            matches_layer = any(name.endswith(kw) for kw in target_keywords)
            if in_scope and matches_layer:
                # print(f"[LoRA-SP] wrapped {name} ({module.in_features} -> {module.out_features})")
                parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model.get_submodule(parent_name) if parent_name else model
                wrapped = LoraMSPLinear(module, msp_cfg)  # verify exact constructor args in lora_msp.py
                # module.update_router_weight_ema_flag = False
                wrapped.update_router_weight_ema_flag = False
                setattr(parent, attr_name, wrapped)


def save_checkpoint(ckpt_dir: Path, step: int, policy: torch.nn.Module, optimizer: torch.optim.Optimizer, is_ddp: bool):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_policy = policy.module if is_ddp else policy
    # Save trainable adapter/router weights + optimizer state
    trainable_state = {k: v.cpu() for k, v in raw_policy.state_dict().items() if k in {n for n, p in raw_policy.named_parameters() if p.requires_grad}}
    torch.save(
        {
            "step": step,
            "trainable_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_dir / f"checkpoint_step_{step}.pt",
    )

def build_action_delta_timestamps(policy_cfg, fps: float) -> dict[str, list[float]]:
    chunk_size = getattr(policy_cfg, "chunk_size", None)
    if chunk_size is None:
        chunk_size = getattr(policy_cfg, "n_action_steps", 50)
    return {"action": [step_idx / fps for step_idx in range(chunk_size)]}


def _compute_merged_linear_weight_and_bias(msp_mod: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Build a static effective weight matrix (out_features, in_features) and optional bias
    from a LoraMSPLinear module so it can be stored in a standard torch.nn.Linear.

    Note: LoraMSPLinear routes dynamically per input token in forward(), so a static
    merge is an approximation of the effective linear map rather than input-dependent routing.
    """
    base_weight = msp_mod.weight.detach().to(torch.float32) if hasattr(msp_mod, "weight") else msp_mod.base_layer.weight.detach().to(torch.float32)
    base_bias = None
    if hasattr(msp_mod, "bias") and msp_mod.bias is not None:
        base_bias = msp_mod.bias.detach().to(torch.float32)
    elif hasattr(msp_mod, "base_layer") and msp_mod.base_layer.bias is not None:
        base_bias = msp_mod.base_layer.bias.detach().to(torch.float32)

    out_features, in_features = base_weight.shape

    # Common adapter attribute patterns in LoRA / LoRA-SP implementations:
    # Check `common/policies/lora_msp.py` to confirm exact tensor names and scaling factor.
    A = getattr(msp_mod, "A", getattr(msp_mod, "lora_A", None))
    B = getattr(msp_mod, "B", getattr(msp_mod, "lora_B", None))

    if A is not None and B is not None:
        A_t = A.weight.detach().to(torch.float32) if hasattr(A, "weight") else A.detach().to(torch.float32)
        B_t = B.weight.detach().to(torch.float32) if hasattr(B, "weight") else B.detach().to(torch.float32)

        # Determine scaling if present (e.g., scaling, alpha / r, or lora_alpha / r)
        cfg = getattr(msp_mod, "cfg", None)
        scale = getattr(msp_mod, "scaling", None)
        if scale is None and cfg is not None:
            r_val = getattr(cfg, "r", None)
            alpha_val = getattr(cfg, "alpha", getattr(cfg, "lora_alpha", None))
            if r_val and alpha_val:
                scale = float(alpha_val) / float(r_val)
        if scale is None:
            scale = 1.0

        # Handle 2D (rank, in_features) / (out_features, rank) or 3D (bank, rank, in_features) tensors
        if A_t.ndim == 2 and B_t.ndim == 2 and A_t.shape[1] == in_features and B_t.shape[0] == out_features:
            delta_w = (B_t @ A_t) * float(scale)
            merged_weight = (base_weight + delta_w).to(base_weight.dtype)
            return merged_weight, base_bias.to(base_weight.dtype) if base_bias is not None else None

        if A_t.ndim == 3 and B_t.ndim == 3:
            # Bank/expert-structured A: (E, R, in_f), B: (E, out_f, R) or (out_f, E*R)
            # Average across bank/experts as a static fallback if shapes align.
            if A_t.shape[-1] == in_features and B_t.shape[1] == out_features and A_t.shape[0] == B_t.shape[0]:
                per_expert_delta = torch.einsum("eor,eri->eoi", B_t, A_t) * float(scale)
                delta_w = per_expert_delta.mean(dim=0)
                merged_weight = (base_weight + delta_w).to(base_weight.dtype)
                return merged_weight, base_bias.to(base_weight.dtype) if base_bias is not None else None

    # Fallback if adapter shapes differ: export base weights cleanly so save_pretrained still succeeds
    return base_weight.to(msp_mod.weight.dtype if hasattr(msp_mod, "weight") else torch.float32), (
        base_bias.to(msp_mod.weight.dtype if hasattr(msp_mod, "weight") else torch.float32)
        if base_bias is not None
        else None
    )


def export_merged_policy(raw_policy: torch.nn.Module, out_dir: Path, model_id: str = "lerobot/smolvla_base") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, module in list(raw_policy.named_modules()):
        if isinstance(module, LoraMSPLinear):
            merged_w, merged_b = _compute_merged_linear_weight_and_bias(module)
            out_f, in_f = merged_w.shape
            plain_linear = torch.nn.Linear(
                in_features=in_f,
                out_features=out_f,
                bias=(merged_b is not None),
                device=merged_w.device,
                dtype=merged_w.dtype,
            )
            with torch.no_grad():
                plain_linear.weight.copy_(merged_w)
                if merged_b is not None and plain_linear.bias is not None:
                    plain_linear.bias.copy_(merged_b)

            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = raw_policy.get_submodule(parent_name) if parent_name else raw_policy
            setattr(parent, attr_name, plain_linear)

    if hasattr(raw_policy, "config"):
        if hasattr(raw_policy.config, "push_to_hub"):
            raw_policy.config.push_to_hub = False
        raw_policy.config.input_features.pop("observation.images.camera3", None)
        if "observation.state" in raw_policy.config.input_features:
            raw_policy.config.input_features["observation.state"].shape = (8,)
        if "action" in raw_policy.config.output_features:
            raw_policy.config.output_features["action"].shape = (7,)

    raw_policy.save_pretrained(out_dir)

    preprocessor, postprocessor = make_pre_post_processors(
        raw_policy.config,
        model_id,
    )
    if preprocessor is not None and hasattr(preprocessor, "save_pretrained"):
        preprocessor.save_pretrained(out_dir)
    if postprocessor is not None and hasattr(postprocessor, "save_pretrained"):
        postprocessor.save_pretrained(out_dir)

def main():
    model_id = "lerobot/smolvla_base"
    dataset_id = "lerobot/libero"
    # batch_size = 32
    # max_steps = 1000
    # save_every = 500
    batch_size = 2
    max_steps = 10
    save_every = 5
    ckpt_dir = Path("./outputs/smolvla_lora_sp")

    device, local_rank, is_ddp = setup_distributed()
    policy = SmolVLAPolicy.from_pretrained(model_id)

    # Example: target specific SmolVLA submodule prefixes for your ablation
    # Inspect `for n, _ in policy.named_modules(): print(n)` once to confirm exact names
    target_prefixes = ()  # e.g. ("model.vlm_with_expert",) or ("model.vision_encoder",)
    target_keywords = (".q_proj", ".k_proj", ".v_proj", ".o_proj", ".gate_proj", ".up_proj", ".down_proj", ".out_proj")
    msp_cfg = LoraMSPConfig(
        r=16,
        alpha=32,
        num_experts=16,
        target_threshold=0.9,
        target_threshold_init=0.9,
        use_spec_loss=True,
    )

    apply_lora_sp(policy, target_prefixes, target_keywords, msp_cfg)
    policy.to(device)

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    print("num of trainable_params: ", len(trainable_params))
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    if is_ddp:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=False)
        # policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=True)

    policy_cfg = (policy.module if is_ddp else policy).config

    # Load metadata first so timestamps match dataset fps
    meta_dataset = LeRobotDataset(dataset_id)
    delta_timestamps = build_action_delta_timestamps(policy_cfg, meta_dataset.meta.fps)
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

    preprocess, _ = make_pre_post_processors(
        (policy.module if is_ddp else policy).config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    policy.train()
    step = 0
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in dataloader:
            if step >= max_steps:
                break
            batch = preprocess(batch)

            for src_key, dst_key in RENAME_MAP.items():
                if src_key in batch and dst_key not in batch:
                    batch[dst_key] = batch[src_key]

            optimizer.zero_grad()

            # Forward pass through SmolVLA
            outputs = policy(batch)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs

            # Pull any accumulated spectral / router aux losses from LoraMSPLinear modules
            # and add them scaled by your coefficients (e.g., 1e-2 * spec_loss + 1e-3 * lb_loss + 1e-2 * z_loss)
               
            raw_policy = policy.module if is_ddp else policy
            msp_modules = [m for m in raw_policy.modules() if isinstance(m, LoraMSPLinear)]

            spec_losses = [m.compute_spec_loss() for m in msp_modules]
            bal_losses = [loss_t for m in msp_modules if (loss_t := m.compute_balance_loss()) is not None]
            z_losses = [loss_t for m in msp_modules if (loss_t := m.compute_z_loss()) is not None]

            spec_loss = torch.stack(spec_losses).mean() if spec_losses else torch.tensor(0.0, device=device)
            bal_loss = torch.stack(bal_losses).mean() if bal_losses else torch.tensor(0.0, device=device)
            z_loss = torch.stack(z_losses).mean() if z_losses else torch.tensor(0.0, device=device)

            total_loss = loss + 1e-2 * spec_loss + 1e-3 * bal_loss + 1e-2 * z_loss
            # Ensure every trainable parameter is present once in the backward graph
            # without changing loss values or gradients, while keeping find_unused_parameters=False
            graph_anchor = sum((p.sum() * 0.0) for p in trainable_params)
            total_loss = total_loss + graph_anchor

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # if step % 50 == 0 and local_rank == 0:
            #     print(f"step={step} total_loss={total_loss.item():.4f} task_loss={loss.item():.4f}")

            log_every = 1  # set to 50 when scale back up to 1000 steps
            if step % log_every == 0 and local_rank == 0:
                print(
                    f"step={step} total_loss={total_loss.item():.4f} "
                    f"task_loss={loss.item():.4f} spec={spec_loss.item():.4f} "
                    f"bal={bal_loss.item():.4f} z={z_loss.item():.4f}"
                )

            if step > 0 and step % save_every == 0 and local_rank == 0:
                save_checkpoint(ckpt_dir, step, policy, optimizer, is_ddp)

            

            step += 1

    raw_policy = policy.module if is_ddp else policy
    if local_rank == 0:
        save_checkpoint(ckpt_dir, step, policy, optimizer, is_ddp)
        # policy.save_pretrained("./outputs/smolvla_lora_sp_eval")
        export_merged_policy(raw_policy, Path("./outputs/smolvla_lora_sp_eval"))
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()