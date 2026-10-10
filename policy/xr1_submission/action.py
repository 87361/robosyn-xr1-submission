import numpy as np
from .robot_bridge import make_physical_action, runtime_joint_limits

def _direct_joint_action(raw_env, values, task_name: str):
    """Prepare a raw dataset joint target for env.step without arm IK."""
    import torch

    action = np.asarray(values, dtype=np.float32).reshape(14).copy()
    if not np.isfinite(action).all():
        raise ValueError("direct joint action contains non-finite values")
    # Eight official tasks store grippers in [0, 1]. table_rearrangement stores
    # physical [0, 0.05], and click_bell stores zeros. Never apply one global
    # conversion across these three source contracts.
    normalized_gripper_tasks = {
        "handle_basket",
        "water_pouring",
        "items_handover",
        "drawer_open_place",
        "mixer_operating",
        "item_assembly",
        "manipulate_pipette",
        "sample_loading",
    }
    if task_name in normalized_gripper_tasks:
        return make_physical_action(raw_env, action)
    if task_name not in {"click_bell", "table_rearrangement"}:
        raise ValueError(f"unknown direct-joint gripper contract for task {task_name!r}")
    active_ids, limits = runtime_joint_limits(raw_env)
    if len(active_ids) != 14:
        raise ValueError(f"environment must expose 14 active joints, got {len(active_ids)}")
    low = limits[active_ids, 0]
    high = limits[active_ids, 1]
    action = np.clip(action, low, high)
    return torch.as_tensor(action, dtype=torch.float32, device=raw_env.device).reshape(1, 14)

