"""RoboSynChallenge adapter; only RGB, proprioception and instruction enter the policy."""
import atexit
import hashlib
from io import BytesIO
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path

import numpy as np

from .runtime.policy_adapter.action import _direct_joint_action
from .runtime.policy_adapter.robot_bridge import extract_observation, as_numpy
from .runtime.policy_adapter.task_contract import load_contract


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


POLICY_ROOT = Path(__file__).resolve().parent
PR_SET_PDEATHSIG = 1


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_model_files(model_dir, artifact):
    """Check the task checkpoint against manifest.json before any GPU work."""
    models = Path(model_dir).expanduser().resolve()
    hint = 'download the complete model repository (manifest.json, processor/ and task folders)'
    checkpoint, config, manifest = models / artifact / 'model.pt', models / artifact / 'config.yaml', models / 'manifest.json'
    for path in (checkpoint, config, manifest):
        if not path.is_file():
            raise FileNotFoundError(f'{path} not found; {hint}')
    processor = models / 'processor'
    if not processor.is_dir():
        raise FileNotFoundError(f'{processor} not found; {hint}')
    files = json.loads(manifest.read_text())['files']
    for path in (checkpoint, config):
        name = path.relative_to(models).as_posix()
        record = files.get(name)
        if record is None or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'{name} does not match {manifest}')
    return checkpoint, config, processor


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def inference_devices(args):
    """Physical GPU for the inference process, matching the simulator's gpu_id."""
    if args.get('xr1_cuda_visible_devices') is not None:
        return str(args['xr1_cuda_visible_devices'])
    gpu = int(args.get('gpu_id') or 0)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    return visible.split(',')[gpu].strip() if visible else str(gpu)


def tail(path, size=4000):
    try:
        return path.read_bytes()[-size:].decode(errors='replace')
    except OSError:
        return ''


class InferenceServer:
    """XR1 inference process in its own Python, owned by the adapter like the official SmolVLA worker."""

    def __init__(self, args, artifact):
        python = args.get('xr1_python') or os.environ.get('XR1_PYTHON') or POLICY_ROOT / '.xr1_env/bin/python'
        if not os.access(python, os.X_OK):
            raise FileNotFoundError(f'XR1 inference Python not found: {python}; '
                                    'run install.sh or set XR1_PYTHON')
        model_dir = args.get('xr1_model_dir') or os.environ.get('XR1_MODEL_DIR')
        if not model_dir:
            raise ValueError('xr1_model_dir is required (eval.sh supplies its policy-local models/ by default)')
        checkpoint, config, processor = resolve_model_files(model_dir, artifact)
        self.log_root = Path(args.get('xr1_log_root') or tempfile.mkdtemp(prefix='xr1_server_'))
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.port = free_port()
        env = os.environ.copy()
        # Only the vendored runtime; no simulator packages leak into the model Python.
        env.update(PYTHONPATH=str(POLICY_ROOT / 'runtime'), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                   PYTHONFAULTHANDLER='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   CUDA_VISIBLE_DEVICES=inference_devices(args))
        env.setdefault('OMP_NUM_THREADS', '4')
        command = [str(python), str(POLICY_ROOT / 'runtime/server.py'), '--config', str(config),
                   '--checkpoint', str(checkpoint), '--processor', str(processor), '--port', str(self.port),
                   '--log-root', str(self.log_root),
                   '--attention-backend', str(args.get('xr1_attention_backend', 'sdpa'))]
        print(f'XR1 inference: {checkpoint} on GPU {env["CUDA_VISIBLE_DEVICES"]}, port {self.port}, '
              f'log {self.log_root}', flush=True)
        parent = os.getpid()

        def die_with_parent():
            # Stop the server even if the evaluator is killed without running atexit.
            try:
                import ctypes
                ctypes.CDLL('libc.so.6', use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
            except Exception:
                pass
            if os.getppid() != parent:
                os._exit(1)

        self.log = (self.log_root / 'server.log').open('w')
        self.proc = subprocess.Popen(command, cwd=POLICY_ROOT / 'runtime', env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT, preexec_fn=die_with_parent, start_new_session=True)
        atexit.register(self.close)
        try:
            self.wait_ready(float(args.get('xr1_startup_timeout') or 900))
        except BaseException:
            self.close()
            raise

    def wait_ready(self, timeout):
        ready, deadline = self.log_root / 'READY', time.monotonic() + timeout
        while not ready.exists():
            if self.proc.poll() is not None:
                raise RuntimeError(f'XR1 inference process exited with {self.proc.returncode}:\n'
                                   + tail(self.log_root / 'server.log'))
            if time.monotonic() > deadline:
                raise TimeoutError(f'XR1 inference process not ready after {timeout:.0f} s:\n'
                                   + tail(self.log_root / 'server.log'))
            time.sleep(1)
        if int(json.loads(ready.read_text())['port']) != self.port:
            raise RuntimeError(f'Unexpected XR1 READY file: {ready}')

    def close(self):
        try:
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=15)
        except ProcessLookupError:
            pass
        except Exception as exc:
            warnings.warn(f'XR1 inference cleanup: {exc}')
        try:
            self.log.close()
        except Exception:
            pass


_simulation_cleanup_registered = False


def _register_simulation_cleanup():
    """Drain official deferred cleanup after evaluator metrics and worker exit."""
    global _simulation_cleanup_registered
    # The evaluator creates its environment before get_model(). Standalone
    # adapter clients and CPU tests do not own a simulator to clean up.
    module = sys.modules.get('embodichain.lab.sim.sim_manager')
    if module is None or _simulation_cleanup_registered:
        return
    manager = module.SimulationManager
    cleanup = manager.flush_cleanup_queue

    def finish_simulation():
        # Native constraint handles must be released while their world is alive.
        # The stock deferred destroy otherwise quits the world before clearing
        # the constraint registry; late handle destructors can then segfault.
        if manager.is_instantiated():
            instance = manager.get_instance()
            names = instance.get_rigid_constraint_uid_list()
            for name in names:
                if not instance.remove_rigid_constraint(name):
                    raise RuntimeError(f'Failed to release simulation constraint: {name}')
            print(f'XR1_SIM_CONSTRAINTS_RELEASED {len(names)}', flush=True)
        cleanup()
        print('XR1_SIM_CLEANUP_DONE', flush=True)

    # atexit is LIFO: later InferenceServer hooks stop workers before this hook.
    atexit.register(finish_simulation)
    _simulation_cleanup_registered = True


def get_model(usr_args):
    _register_simulation_cleanup()
    if usr_args.get('xr1_port') is not None:
        return Policy(usr_args)  # An externally managed inference server.
    artifact = load_contract(str(usr_args['task_name']), usr_args.get('xr1_execute_steps'))['artifact']
    server = InferenceServer(usr_args, artifact)
    model = Policy({**usr_args, 'xr1_port': server.port})
    model.server = server
    return model


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
