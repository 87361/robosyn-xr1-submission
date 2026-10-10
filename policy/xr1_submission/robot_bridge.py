"""Shared RoboSynChallenge observation and action bridge for XR1 and OpenWAM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


IMAGE_NAMES = ("ego", "left_wrist", "right_wrist")
SENSOR_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def as_numpy(value: Any) -> np.ndarray:
    """Copy a NumPy/CPU/CUDA tensor-like value to host NumPy."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def coerce_rgb(value: Any, *, name: str) -> np.ndarray:
    """Validate a batched or unbatched RGB(A) image and return HWC uint8 RGB."""
    image = as_numpy(value)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"{name} must contain one environment, got {image.shape}")
        image = image[0]
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"{name} must be HxWxRGB(A), got {image.shape}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError(f"{name} contains non-finite pixels")
        scale = 255.0 if image.size and float(image.max()) <= 1.0 else 1.0
        image = np.rint(np.clip(image * scale, 0.0, 255.0)).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def extract_observation(
    obs: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract the three XR-1 RGB inputs and canonical CobotMagic 14D state."""
    try:
        images = {
            output_name: coerce_rgb(
                obs["sensor"][sensor_name]["color"], name=sensor_name
            )
            for output_name, sensor_name in zip(IMAGE_NAMES, SENSOR_NAMES)
        }
        state = as_numpy(obs["robot"]["qpos"])
    except KeyError as exc:
        raise KeyError(f"live observation is missing required field {exc}") from exc
    if state.ndim == 2:
        if state.shape[0] != 1:
            raise ValueError(f"qpos must contain one environment, got {state.shape}")
        state = state[0]
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (14,):
        raise ValueError(
            "qpos must be [left_joint6, left_gripper, right_joint6, right_gripper], "
            f"got {state.shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError("qpos contains non-finite values")
    return images, state.copy()


def split_state(state: np.ndarray) -> dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (14,):
        raise ValueError(f"state must have shape (14,), got {state.shape}")
    return {
        "left_arm": state[:6],
        "left_gripper": state[6:7],
        "right_arm": state[7:13],
        "right_gripper": state[13:14],
    }


def normalize_pose_matrix(value: Any, *, name: str) -> np.ndarray:
    pose = as_numpy(value)
    if pose.shape == (1, 4, 4):
        pose = pose[0]
    if pose.shape != (4, 4):
        raise ValueError(f"{name} pose must have shape (4, 4), got {pose.shape}")
    pose = np.asarray(pose, dtype=np.float32)
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} pose contains non-finite values")
    return pose


def compute_current_fk(
    robot: Any, state: np.ndarray, device: Any
) -> dict[str, np.ndarray]:
    """Compute both current EE poses with the live environment's robot object."""
    import torch

    parts = split_state(state)
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        qpos = torch.as_tensor(
            parts[f"{side}_arm"], dtype=torch.float32, device=device
        ).reshape(1, 6)
        env_ids = torch.zeros(1, dtype=torch.long, device=device)
        try:
            pose_value = robot.compute_fk(
                name=f"{side}_arm", qpos=qpos, env_ids=env_ids, to_matrix=True
            )
        except TypeError:
            pose_value = robot.compute_fk(
                name=f"{side}_arm", qpos=qpos, to_matrix=True
            )
        pose = normalize_pose_matrix(pose_value, name=f"{side}_arm FK")
        result[f"{side}_ee_pos"] = pose[:3, 3].copy()
        result[f"{side}_ee_rotm"] = pose[:3, :3].copy()
    return result


def make_request_payload(
    images: Mapping[str, np.ndarray],
    state: np.ndarray,
    instruction: str,
    ee_poses: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    payload = {name: coerce_rgb(images[name], name=name) for name in IMAGE_NAMES}
    payload["state"] = np.asarray(state, dtype=np.float32).reshape(14)
    payload["instruction"] = np.asarray(instruction)
    for side in ("left", "right"):
        payload[f"{side}_ee_pos"] = np.asarray(
            ee_poses[f"{side}_ee_pos"], dtype=np.float32
        ).reshape(3)
        payload[f"{side}_ee_rotm"] = np.asarray(
            ee_poses[f"{side}_ee_rotm"], dtype=np.float32
        ).reshape(3, 3)
    return payload


def load_inference_targets(path: Path, target_step: int) -> dict[str, np.ndarray]:
    """Load and validate one absolute target from XR-1's inference result."""
    if target_step < 0:
        raise ValueError("target_step must be non-negative")
    required = (
        "raw_action",
        "target_left_ee_pos",
        "target_left_ee_rotm",
        "target_left_gripper_pos",
        "target_right_ee_pos",
        "target_right_ee_rotm",
        "target_right_gripper_pos",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"XR-1 result is missing fields: {missing}")
        raw_action = np.asarray(data["raw_action"], dtype=np.float32)
        if raw_action.shape != (30, 60):
            raise ValueError(
                f"raw_action must have shape (30, 60), got {raw_action.shape}"
            )
        if target_step >= raw_action.shape[0]:
            raise ValueError(
                f"target_step {target_step} is outside the 30-step horizon"
            )
        targets = {
            key.removeprefix("target_"): np.asarray(data[key])[target_step]
            for key in required[1:]
        }
    expected = {
        "left_ee_pos": (3,),
        "left_ee_rotm": (3, 3),
        "left_gripper_pos": (1,),
        "right_ee_pos": (3,),
        "right_ee_rotm": (3, 3),
        "right_gripper_pos": (1,),
    }
    for key, shape in expected.items():
        targets[key] = np.asarray(targets[key], dtype=np.float32).reshape(shape)
        if not np.isfinite(targets[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    targets["raw_action"] = raw_action
    return targets


def pose_from_target(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    if not np.isfinite(pose).all():
        raise ValueError("target pose contains non-finite values")
    return pose


def ik_succeeded(value: Any) -> bool:
    value = as_numpy(value)
    return bool(value.all())


class IKSolveError(RuntimeError):
    """A recoverable IK result failure, distinct from simulator/runtime faults."""


def solve_arm_ik(
    robot: Any, side: str, pose: np.ndarray, seed: np.ndarray, device: Any
) -> np.ndarray:
    """Use the EmbodiChain IK API variants present in the supported releases."""
    import torch

    target = torch.as_tensor(pose, dtype=torch.float32, device=device)
    qpos_seed = torch.as_tensor(seed, dtype=torch.float32, device=device)
    name = f"{side}_arm"
    try:
        success, qpos = robot.compute_ik(name=name, pose=target, qpos_seed=qpos_seed)
    except TypeError:
        try:
            success, qpos = robot.compute_ik(
                name=name, pose=target, joint_seed=qpos_seed
            )
        except TypeError:
            success, qpos = robot.compute_ik(target, qpos_seed, name)
    if not ik_succeeded(success):
        raise IKSolveError(f"EmbodiChain IK failed for {name}")
    qpos = np.asarray(as_numpy(qpos), dtype=np.float32).reshape(-1)
    if qpos.shape != (6,) or not np.isfinite(qpos).all():
        raise IKSolveError(
            f"EmbodiChain IK returned invalid {name} qpos: {qpos.shape}"
        )
    return qpos


def compose_normalized_action(
    left_arm: np.ndarray,
    left_gripper: np.ndarray,
    right_arm: np.ndarray,
    right_gripper: np.ndarray,
) -> np.ndarray:
    """Pack IK arms plus XR-1 normalized grippers in canonical 14D order."""
    action = np.concatenate(
        (
            np.asarray(left_arm, dtype=np.float32).reshape(6),
            np.clip(np.asarray(left_gripper, dtype=np.float32).reshape(1), 0.0, 1.0),
            np.asarray(right_arm, dtype=np.float32).reshape(6),
            np.clip(np.asarray(right_gripper, dtype=np.float32).reshape(1), 0.0, 1.0),
        )
    ).astype(np.float32)
    if not np.isfinite(action).all():
        raise ValueError("composed action contains non-finite values")
    return action


def map_normalized_grippers(
    action: np.ndarray,
    active_joint_ids: list[int],
    left_eef_ids: list[int],
    right_eef_ids: list[int],
    qpos_limits: np.ndarray,
) -> np.ndarray:
    """Map normalized grippers to live robot limits without assuming 0.05 m."""
    physical = np.asarray(action, dtype=np.float32).reshape(14).copy()
    limits = np.asarray(qpos_limits, dtype=np.float32)
    if limits.ndim == 3 and limits.shape[0] == 1:
        limits = limits[0]
    if limits.ndim != 2 or limits.shape[1] != 2:
        raise ValueError(f"qpos_limits must be (joints, 2), got {limits.shape}")
    active_lookup = {joint_id: index for index, joint_id in enumerate(active_joint_ids)}
    gripper_ids = list(left_eef_ids) + list(right_eef_ids)
    if len(gripper_ids) != 2:
        raise ValueError(f"expected one non-mimic joint per gripper, got {gripper_ids}")
    for joint_id in gripper_ids:
        if joint_id not in active_lookup:
            raise ValueError(f"gripper joint {joint_id} is not active")
        action_index = active_lookup[joint_id]
        low, high = limits[joint_id]
        physical[action_index] = low + np.clip(physical[action_index], 0.0, 1.0) * (
            high - low
        )
    return physical


def runtime_joint_limits(raw_env: Any) -> tuple[list[int], np.ndarray]:
    active_joint_ids = [int(value) for value in raw_env.active_joint_ids]
    limits = np.asarray(as_numpy(raw_env.robot.body_data.qpos_limits), dtype=np.float32)
    if limits.ndim == 3:
        limits = limits[0]
    return active_joint_ids, limits


def make_physical_action(raw_env: Any, normalized_action: np.ndarray):
    """Use an env capability when available, otherwise runtime robot limits."""
    import torch

    action = torch.as_tensor(
        normalized_action, dtype=torch.float32, device=raw_env.device
    ).reshape(1, 14)
    conversion = getattr(raw_env, "denormalize_policy_action", None)
    if callable(conversion):
        action = conversion(action)
    else:
        active_ids, limits = runtime_joint_limits(raw_env)
        left_ids = list(raw_env.robot.get_joint_ids(name="left_eef", remove_mimic=True))
        right_ids = list(
            raw_env.robot.get_joint_ids(name="right_eef", remove_mimic=True)
        )
        mapped = map_normalized_grippers(
            as_numpy(action)[0], active_ids, left_ids, right_ids, limits
        )
        action = torch.as_tensor(
            mapped, dtype=torch.float32, device=raw_env.device
        ).reshape(1, 14)

    active_ids, limits = runtime_joint_limits(raw_env)
    if len(active_ids) != 14:
        raise ValueError(
            f"environment must expose 14 active joints, got {len(active_ids)}"
        )
    low = torch.as_tensor(
        limits[active_ids, 0], dtype=action.dtype, device=action.device
    )
    high = torch.as_tensor(
        limits[active_ids, 1], dtype=action.dtype, device=action.device
    )
    action = torch.maximum(torch.minimum(action, high), low)
    if not bool(torch.isfinite(action).all().item()):
        raise ValueError("physical environment action contains non-finite values")
    return action


