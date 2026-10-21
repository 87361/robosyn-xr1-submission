#!/usr/bin/env python3
# Copyright (C) 2026 Xiaomi Corporation.
"""Run one raw XR-1 checkpoint inference from a RoboSyn NPZ request."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

from mibot.models.VLA.XR1 import xr1
from mibot.server.runtime.client import Client
from mibot.utils.io import (
    ACTION_EPS,
    build_action_mask_for_representation,
    compose_state,
    denormalize_action,
    recover_action,
    recover_joint_action,
    resize_image,
)


IMAGE_KEYS = ("ego", "left_wrist", "right_wrist")
EEF_KEYS = ("left_ee_pos", "left_ee_rotm", "right_ee_pos", "right_ee_rotm")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Resolved Sim10 data YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Raw model_states.pt")
    parser.add_argument("--request", type=Path, required=True, help="Request NPZ or directory containing request.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: Path):
    text = path.read_text(encoding="utf-8")
    if "???" in text:
        raise ValueError(f"{path} is an unresolved template")
    config = yaml.safe_load(text)
    data = config["data"]["params"]["train_datasets"]
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    q01 = np.asarray(data["q01"], dtype=np.float32)
    q99 = np.asarray(data["q99"], dtype=np.float32)
    if mean.ndim != 2 or mean.shape[1] != 60 or std.shape != mean.shape:
        raise ValueError(f"action statistics must have shape (T, 60), got {mean.shape} and {std.shape}")
    action_length = int(mean.shape[0])
    if q01.shape != (1, 60) or q99.shape != (1, 60):
        raise ValueError(f"state quantiles must have shape (1, 60), got {q01.shape} and {q99.shape}")
    model_params = config.get("model", {}).get("params", {}).get("model", {})
    model_params = {key: value for key, value in model_params.items() if key != "type"}
    configured_action_length = int(model_params.get("action_length", action_length))
    if configured_action_length != action_length:
        raise ValueError(
            f"model/data action lengths disagree: {configured_action_length} != {action_length}"
        )
    model_params["action_length"] = action_length
    action_representation = data.get("action_representation", "eef_delta")
    if action_representation not in {"eef_delta", "joint_absolute"}:
        raise ValueError(f"unsupported action representation: {action_representation!r}")
    return mean, std, q01, q99, model_params, action_representation, action_length


def load_request(path: Path, action_representation: str = "eef_delta"):
    if path.is_dir():
        path = path / "request.npz"
    if not path.is_file():
        raise FileNotFoundError(f"request does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = (*IMAGE_KEYS, "state", "instruction")
        if action_representation == "eef_delta":
            required += EEF_KEYS
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f"request is missing required fields: {missing}")
        request = {key: payload[key].copy() for key in payload.files}

    state = np.asarray(request["state"], dtype=np.float32).reshape(-1)
    if state.shape != (14,):
        raise ValueError(f"state must use [left_joint6, left_gripper, right_joint6, right_gripper], got {state.shape}")
    instruction = np.asarray(request["instruction"])
    if instruction.ndim != 0:
        raise ValueError("instruction must be a scalar string")
    instruction = str(instruction.item())

    images = []
    for key in IMAGE_KEYS:
        array = np.asarray(request[key])
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError(f"{key} must be an HxWx3 uint8 RGB array, got {array.shape} {array.dtype}")
        images.append(resize_image(Image.fromarray(array, mode="RGB"), factor=32, max_pixels=160000))

    robot_state = {
        "left_arm_joint": state[:6],
        "left_gripper_pos": state[6:7],
        "right_arm_joint": state[7:13],
        "right_gripper_pos": state[13:14],
    }
    for key, shape in {
        "left_ee_pos": (3,),
        "left_ee_rotm": (3, 3),
        "right_ee_pos": (3,),
        "right_ee_rotm": (3, 3),
    }.items():
        if key not in request:
            continue
        value = np.asarray(request[key], dtype=np.float32).reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
        robot_state[key] = value
    return images, instruction, robot_state


def load_model(checkpoint: Path, model_params, device):
    model = xr1(**model_params)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    state_dict = payload.get("module", payload)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint must be a state dict or contain a 'module' state dict")
    prefix = "model."
    if not state_dict or not all(key.startswith(prefix) for key in state_dict):
        bad = next((key for key in state_dict if not key.startswith(prefix)), "<empty>")
        raise ValueError(f"raw checkpoint keys must all start with {prefix!r}; first mismatch: {bad}")
    model.load_state_dict({key[len(prefix) :]: value for key, value in state_dict.items()}, strict=True)
    return model.eval().to(device)


def prepare_batch(
    images, instruction, robot_state, q01, q99, device,
    action_representation="eef_delta", action_length=30,
):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    processor.tokenizer.padding_side = "right"
    batch = processor.apply_chat_template(
        [Client._messages(instruction, *images)],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        images_kwargs={"do_resize": False},
    )
    state = compose_state(
        left_gripper=robot_state["left_gripper_pos"],
        left_joint=robot_state["left_arm_joint"],
        right_gripper=robot_state["right_gripper_pos"],
        right_joint=robot_state["right_arm_joint"],
    )
    valid = q99 > q01
    normalized = np.zeros_like(state)
    normalized[valid] = 2.0 * (state[valid] - q01[valid]) / (q99[valid] - q01[valid] + ACTION_EPS) - 1.0
    batch["state"] = torch.from_numpy(np.clip(normalized, -1.0, 1.0))[None]
    batch["action"] = torch.zeros((1, action_length, 60), dtype=torch.bfloat16)
    batch["action_mask"] = torch.from_numpy(
        build_action_mask_for_representation(action_length, action_representation)
    )[None]
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    mean, std, q01, q99, model_params, action_representation, action_length = load_config(args.config)
    images, instruction, robot_state = load_request(args.request, action_representation)
    model = load_model(args.checkpoint, model_params, args.device)
    batch = prepare_batch(
        images, instruction, robot_state, q01, q99, args.device,
        action_representation, action_length,
    )
    mask = batch["action_mask"]
    mean_tensor = torch.from_numpy(mean).to(args.device)
    std_tensor = torch.from_numpy(std).to(args.device)
    with torch.inference_mode():
        normalized = model.generate(batch)
        action = denormalize_action(normalized * mask, mean_tensor, std_tensor) * mask
    action = np.asarray(action.detach().float().cpu(), dtype=np.float32)[0]
    if action.shape != (action_length, 60) or not np.isfinite(action).all():
        raise RuntimeError(f"invalid model output: shape={action.shape}, finite={np.isfinite(action).all()}")
    targets = (
        recover_action(action, robot_state)
        if action_representation == "eef_delta"
        else {"joint_action": recover_joint_action(action)}
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"raw_action": action}
    output.update({f"target_{key}": value for key, value in targets.items()})
    np.savez_compressed(args.output, **output)
    print(f"saved {args.output}: raw_action={action.shape}; targets={sorted(targets)}")


if __name__ == "__main__":
    main()
