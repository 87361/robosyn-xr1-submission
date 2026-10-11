"""Isolated XR1 server: per-episode RNG streams and native prefix inpainting."""
from __future__ import annotations
import argparse
import hashlib
from io import BytesIO
import os
from collections import OrderedDict
import json
from pathlib import Path
import random
import socketserver
import sys
import threading
import time

import numpy as np
import torch


def prepare_cached(processor, images, instruction, state, q01, q99, horizon, device):
    from mibot.utils.io import ACTION_EPS, compose_state, build_action_mask_for_representation
    from mibot.server.runtime.client import Client
    batch = processor.apply_chat_template([Client._messages(instruction, *images)],
        tokenize=True, return_dict=True, return_tensors='pt', padding=True,
        images_kwargs={'do_resize': False})
    packed = compose_state(state['left_gripper_pos'], state['left_arm_joint'],
                           state['right_gripper_pos'], state['right_arm_joint'])
    valid = q99 > q01
    normalized = np.zeros_like(packed)
    normalized[valid] = 2*(packed[valid]-q01[valid])/(q99[valid]-q01[valid]+ACTION_EPS)-1
    batch['state'] = torch.from_numpy(np.clip(normalized, -1, 1))[None]
    batch['action'] = torch.zeros((1,horizon,60), dtype=torch.bfloat16)
    batch['action_mask'] = torch.from_numpy(build_action_mask_for_representation(horizon,'joint_absolute'))[None]
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xr1-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--processor', type=Path, required=True)
    parser.add_argument('--attention-backend', choices=['sdpa', 'flash_attention_2'], default='sdpa')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--log-root', type=Path, required=True)
    args = parser.parse_args()
    print('XR1_STARTUP ' + json.dumps({'pid': os.getpid(), 'torch': torch.__version__,
        'torch_cpu_threads': torch.get_num_threads(), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')}), flush=True)
    os.environ["XR1_PROCESSOR_PATH"] = str(args.processor.resolve())
    os.environ["XR1_ATTENTION_BACKEND"] = args.attention_backend
    sys.path[:0] = [str(args.xr1_root), str(args.xr1_root/'tools')]
    from infer_npz import IMAGE_KEYS, load_config, load_model
    from vendor.baseline_server import recv_npz, send_npz, decode_request
    from mibot.utils.io import (ACTION_EPS, compose_joint_action, compose_state,
                               recover_joint_action, build_action_mask_for_representation)
    from mibot.server.runtime.client import Client
    from transformers import AutoProcessor
    mean, std, q01, q99, params, representation, horizon = load_config(args.config)
    assert representation == 'joint_absolute' and horizon == 30
    args.log_root.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    processor.tokenizer.padding_side = 'right'
    print('XR1_LOADING_MODEL', flush=True)
    model = load_model(args.checkpoint, params, 'cuda:0')
    mean_t, std_t = (torch.from_numpy(x).to('cuda:0') for x in (mean, std))
    lock = threading.Lock()
    contexts = OrderedDict()
    try: log = (args.log_root/'requests.jsonl').open('a', buffering=1)
    except OSError: log = None

    def rng_state():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())

    def restore_rng(state):
        random.setstate(state[0]); np.random.set_state(state[1])
        torch.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3])

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            while (request := recv_npz(self.request)) is not None:
                with lock:
                    try:
                        episode_id = str(request['episode_id'].item())
                        index = int(request['chunk_index'].item())
                        seed = int(request['policy_seed'].item())
                        prefix = np.asarray(request.get('prefix_joint_action',np.empty((0,14))),dtype=np.float32)
                        p = len(prefix)
                        assert prefix.shape == (p,14) and 0 <= p <= 6 and np.isfinite(prefix).all()
                        context = contexts.get(episode_id)
                        if context is not None and index == context['index']:
                            send_npz(self.request, context['response']); continue
                        if index == 0:
                            assert p == 0
                            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                            context = {'index':-1,'seed':seed}
                            contexts[episode_id] = context
                            while len(contexts) > 64: contexts.popitem(last=False)
                        else:
                            assert context is not None and context['seed'] == seed
                            assert index == context['index']+1, (index,context['index'])
                            restore_rng(context['rng'])
                        started = time.perf_counter()
                        images,instruction,state = decode_request(request, representation)
                        batch = prepare_cached(processor,images,instruction,state,q01,q99,horizon,'cuda:0')
                        mask = batch['action_mask'].clone()
                        if p:
                            packed = compose_joint_action(prefix[:,:6],prefix[:,6:7],prefix[:,7:13],prefix[:,13:14],p)
                            normalized_prefix = (packed-mean[:p])/(std[:p]+ACTION_EPS)
                            batch['action'][:,:p] = torch.from_numpy(normalized_prefix).to('cuda:0',dtype=torch.bfloat16)
                        batch['prefix_length'] = p
                        expected_prefix = batch['action'][:,:p].clone()
                        torch.cuda.synchronize()
                        inference_started = time.perf_counter()
                        with torch.inference_mode():
                            normalized = model.generate(batch)
                            prefix_error = float((normalized[:,:p]-expected_prefix).abs().max()) if p else 0.0
                            assert prefix_error <= 1e-6, prefix_error
                            action = (normalized*mask*(std_t+ACTION_EPS)+mean_t)*mask
                        torch.cuda.synchronize()
                        inference_seconds = time.perf_counter()-inference_started
                        raw = action.detach().float().cpu().numpy()[0]
                        assert raw.shape == (horizon,60) and np.isfinite(raw).all()
                        # Native inpainting fixed the normalized prefix. Restore its exact
                        # commanded values after bfloat16 normalization rounding only.
                        rounding_error = float(np.max(np.abs(recover_joint_action(raw[:p])-prefix))) if p else 0.0
                        if p: raw[:p] = packed
                        response = {'target_joint_action':recover_joint_action(raw),'raw_action':raw,
                            'native_prefix_max_error':np.asarray(prefix_error),
                            'prefix_rounding_error':np.asarray(rounding_error),
                            'generation_seconds':np.asarray(inference_seconds),
                            'server_seconds':np.asarray(time.perf_counter()-started),
                            'policy_seed':np.asarray(seed),'prefix_length':np.asarray(p)}
                        context.update(index=index,rng=rng_state(),response=response)
                        try:
                            input_hashes = {key: hashlib.sha256(np.ascontiguousarray(request[key]).tobytes()).hexdigest()
                                for key in (*IMAGE_KEYS, 'state', 'instruction')
                                if key in request}
                            if index == 0 and os.environ.get('XR1_CAPTURE_FIRST_INPUT') == '1':
                                # Local RGB/proprioception only; optional diagnostic, never privileged state.
                                capture_id = hashlib.sha256(episode_id.encode()).hexdigest()[:16]
                                np.savez_compressed(args.log_root / f'first_input_{capture_id}.npz', **request)
                            if log is not None: log.write(json.dumps({'episode_id':episode_id,'chunk_index':index,'policy_seed':seed,
                                'prefix_length':p,'native_prefix_max_error':prefix_error,
                                'prefix_rounding_error':rounding_error,'generation_seconds':inference_seconds,
                                'input_sha256':input_hashes,
                                'action_sha256':hashlib.sha256(response['target_joint_action'].tobytes()).hexdigest()})+'\n')
                        except Exception as exc: print(f'REQUEST_LOG_WARNING: {exc}',flush=True)
                        send_npz(self.request,response)
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        send_npz(self.request,{'error':np.asarray(str(exc))})
                        return

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    with Server(('127.0.0.1',args.port),Handler) as server:
        (args.log_root/'READY').write_text(json.dumps({'checkpoint':str(args.checkpoint),'horizon':horizon,'port':args.port}))
        print(f'XR1_REPLAN_SERVER_READY port={args.port} horizon={horizon}',flush=True)
        server.serve_forever()


if __name__ == '__main__': main()
