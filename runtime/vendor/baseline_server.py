#!/usr/bin/env python3
"""Serve raw RoboSyn observations to an XR-1 checkpoint over framed NPZ/TCP."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import socket
import struct

import numpy as np
import torch
from PIL import Image

from infer_npz import IMAGE_KEYS, load_config, load_model, prepare_batch
from mibot.utils.io import recover_action, recover_joint_action, resize_image


def recv_all(conn: socket.socket, length: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = conn.recv(length - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def recv_npz(conn: socket.socket) -> dict[str, np.ndarray] | None:
    header = recv_all(conn, 4)
    if header is None:
        return None
    size = struct.unpack(">I", header)[0]
    body = recv_all(conn, size)
    if body is None:
        return None
    with np.load(BytesIO(body), allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def send_npz(conn: socket.socket, payload: dict[str, np.ndarray]) -> None:
    buffer = BytesIO()
    np.savez_compressed(buffer, **payload)
    data = buffer.getvalue()
    conn.sendall(struct.pack(">I", len(data)) + data)


def decode_request(request: dict[str, np.ndarray], action_representation="eef_delta"):
    required = (*IMAGE_KEYS, "state", "instruction")
    if action_representation == "eef_delta":
        required += ("left_ee_pos", "left_ee_rotm", "right_ee_pos", "right_ee_rotm")
    missing = [key for key in required if key not in request]
    if missing:
        raise ValueError(f"request is missing fields: {missing}")

    state = np.asarray(request["state"], dtype=np.float32).reshape(-1)
    if state.shape != (14,):
        raise ValueError(f"state must have shape (14,), got {state.shape}")
    instruction = str(np.asarray(request["instruction"]).item())
    images = []
    for key in IMAGE_KEYS:
        image = np.asarray(request[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{key} must be HxWx3 uint8, got {image.shape} {image.dtype}")
        images.append(
            resize_image(
                Image.fromarray(image, mode="RGB"), factor=32, max_pixels=160000
            )
        )

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
        if key in request:
            robot_state[key] = np.asarray(request[key], dtype=np.float32).reshape(shape)
    return images, instruction, robot_state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    mean, std, q01, q99, model_params, action_representation, action_length = load_config(args.config)
    model = load_model(args.checkpoint, model_params, args.device)
    mean_tensor = torch.from_numpy(mean).to(args.device)
    std_tensor = torch.from_numpy(std).to(args.device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"XR-1 raw policy server ready on {args.host}:{args.port}", flush=True)
        while True:
            conn, address = server.accept()
            print(f"client connected: {address}", flush=True)
            with conn:
                while (request := recv_npz(conn)) is not None:
                    images, instruction, robot_state = decode_request(request, action_representation)
                    batch = prepare_batch(
                        images, instruction, robot_state, q01, q99, args.device,
                        action_representation, action_length,
                    )
                    mask = batch["action_mask"]
                    with torch.inference_mode():
                        normalized = model.generate(batch)
                        action = normalized * mask * (std_tensor + 1e-6) + mean_tensor
                        action = action * mask
                    raw_action = np.asarray(action.detach().float().cpu(), dtype=np.float32)[0]
                    targets = (
                        recover_action(raw_action, robot_state)
                        if action_representation == "eef_delta"
                        else {"joint_action": recover_joint_action(raw_action)}
                    )
                    response = {"raw_action": raw_action}
                    response.update({f"target_{key}": value for key, value in targets.items()})
                    send_npz(conn, response)
            print("client disconnected", flush=True)


if __name__ == "__main__":
    main()
