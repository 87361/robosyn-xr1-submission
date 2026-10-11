"""RoboSynChallenge adapter; only RGB, proprioception and instruction enter the policy."""
from io import BytesIO
import json
import socket
import struct
import time
import uuid
import warnings
from pathlib import Path

import numpy as np

from .action import _direct_joint_action
from .robot_bridge import extract_observation, as_numpy
from .task_contract import load_contract


def scalar(value):
    return bool(as_numpy(value).reshape(-1)[0])


class Policy:
    def __init__(self, args):
        self.host = str(args.get('xr1_host', '127.0.0.1'))
        self.port = int(args['xr1_port'])
        self.task = str(args['task_name'])
        self.k = load_contract(self.task, args.get('xr1_execute_steps'))['execute_steps']
        self.seed = int(args.get('xr1_policy_seed', 20260908))
        self.session = uuid.uuid4().hex
        self.episode = -1
        self.sock = None
        self.trace = None
        self.trace_root = args.get('xr1_trace_root')
        self.chunk = 0
        self.steps = 0
        self.last_info = None

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=30)
        self.sock.settimeout(300)

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def recv_all(self, size):
        result = bytearray()
        while len(result) < size:
            block = self.sock.recv(size - len(result))
            if not block:
                raise ConnectionError('XR1 server disconnected')
            result.extend(block)
        return bytes(result)

    def infer(self, payload):
        buffer = BytesIO()
        np.savez_compressed(buffer, **payload)
        body = buffer.getvalue()
        # Same episode/chunk ID makes a retry idempotent on the server.
        for attempt in range(3):
            try:
                if self.sock is None:
                    self.connect()
                self.sock.sendall(struct.pack('>I', len(body)) + body)
                size = struct.unpack('>I', self.recv_all(4))[0]
                if not 0 < size < 64 * 1024 * 1024:
                    raise ValueError('Invalid XR1 response size')
                with np.load(BytesIO(self.recv_all(size)), allow_pickle=False) as data:
                    response = {k: data[k].copy() for k in data.files}
                if 'error' in response:
                    raise RuntimeError(str(response['error'].item()))
                return response
            except (OSError, ConnectionError):
                self.close()
                if attempt == 2:
                    raise
                time.sleep(1)

    def log(self, row):
        if self.trace is not None:
            try:
                self.trace.write(json.dumps(row) + '\n')
            except Exception as exc:
                warnings.warn(f'Optional XR1 trace unavailable: {exc}')
                self.trace = None


def get_model(usr_args):
    return Policy(usr_args)


def reset_model(model):
    if model.trace is not None:
        try:
            model.trace.close()
        except Exception:
            pass
    model.trace = None
    model.episode += 1
    model.chunk = 0
    model.steps = 0
    model.last_info = None
    if model.trace_root:
        try:
            root = Path(model.trace_root)
            root.mkdir(parents=True, exist_ok=True)
            model.trace = (root / f'episode_{model.episode:03d}.jsonl').open('w', buffering=1)
        except OSError as exc:
            warnings.warn(f'Optional XR1 trace unavailable: {exc}')


def eval(env, model, obs):
    # Include preprocessing, transport, model and action encoding in official timing.
    started = time.perf_counter()
    if model.episode < 0:
        reset_model(model)
    images, state = extract_observation(obs)
    instruction = getattr(env, '_current_instruction', None)
    if not instruction:
        raise ValueError('The evaluator must supply the task language instruction')
    response = model.infer(dict(images, state=state, instruction=np.asarray(instruction),
        episode_id=np.asarray(f'{model.session}:{model.episode}'),
        chunk_index=np.asarray(model.chunk), policy_seed=np.asarray(model.seed),
        prefix_joint_action=np.empty((0, 14), dtype=np.float32)))
    targets = np.asarray(response['target_joint_action'], dtype=np.float32)
    if targets.shape != (30, 14) or not np.isfinite(targets).all():
        raise ValueError(f'Invalid XR1 action shape or values: {targets.shape}')
    if int(response['policy_seed']) != model.seed or int(response['prefix_length']) != 0:
        raise ValueError('Server policy RNG/prefix contract mismatch')
    actions = [_direct_joint_action(env.unwrapped, row, model.task) for row in targets[:model.k]]
    # Action encoding may queue device operations even with a remote model.
    import torch
    if actions and actions[0].is_cuda:
        torch.cuda.synchronize(actions[0].device)
    elapsed = time.perf_counter() - started
    model.log({'kind': 'prediction', 'episode': model.episode, 'chunk': model.chunk,
        'policy_seed': model.seed, 'inference_seconds': elapsed, 'k': model.k,
        'state': state.tolist(), 'target_joint_action': targets.tolist()})
    final_obs, info, done = obs, model.last_info, False
    for action in actions:
        final_obs, _, terminated, truncated, info = env.step(action)
        model.steps += 1
        model.last_info = info
        success = scalar(env.get_wrapper_attr('is_task_success')())
        model.log({'kind': 'step', 'episode': model.episode, 'step': model.steps,
                   'success': success, 'physical_command': as_numpy(action).reshape(-1).tolist()})
        done = scalar(truncated) or (scalar(terminated) and not success)
        if done or success:
            break
    model.chunk += 1
    return final_obs, info, done, [elapsed]


def get_episode_diagnostics(model):
    return {'inference_calls': model.chunk, 'executed_action_steps': model.steps,
            'policy_seed': model.seed, 'execute_steps': model.k, 'prefix_steps': 0}
