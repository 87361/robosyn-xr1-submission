"""Own the inference server lifecycle and invoke the organizer's unchanged evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from sim_runtime import prepare as prepare_sim_runtime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'policy/xr1_submission'))
from task_contract import load_contract
TASKS = 'click_bell handle_basket water_pouring table_rearrangement items_handover drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading'.split()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finish(output, processes, runtime, status):
    """Keep the primary result even if a child exits concurrently or cleanup fails."""
    errors = []
    for proc in processes:
        if proc is None:
            continue
        try:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=15)
        except Exception as exc:
            errors.append(f'process {proc.pid}: {exc}')
    retained = str(runtime) if runtime is not None and status != 0 else None
    if runtime is not None and status == 0:
        try:
            shutil.rmtree(runtime)
        except Exception as exc:
            retained = str(runtime)
            errors.append(f'runtime cleanup: {exc}')
    for name, content in {
        'exit.code': str(status) + '\n',
        'cleanup.json': json.dumps({'primary_exit_code': status,
            'retained_runtime': retained, 'warnings': errors}, indent=2) + '\n',
    }.items():
        try:
            (output / name).write_text(content)
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    for error in errors:
        try:
            print(f'CLEANUP_WARNING: {error}', file=sys.stderr, flush=True)
        except Exception:
            pass


def check_task_assets(challenge, task, setting):
    path = challenge / 'configs' / task / setting / 'gym_config.json'
    config = json.loads(path.read_text())
    missing = []
    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.startswith(('assets/', 'robosynchallenge/')):
            if not (challenge / value).exists():
                missing.append(value)
    visit(config)
    if missing:
        raise FileNotFoundError('Incomplete official task assets: ' + ', '.join(sorted(set(missing))))
    return {'config': str(path), 'sha256': sha(path), 'local_asset_references_present': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=TASKS)
    p.add_argument('--challenge-root', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--sim-python', required=True)
    p.add_argument('--model-python', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--setting', default='random')
    p.add_argument('--gpu', default='0')
    p.add_argument('--port', type=int, default=23609)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--video', action='store_true')
    p.add_argument('--attention-backend', choices=['sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--startup-timeout', type=int, default=900)
    p.add_argument('--no-isolate-simulator', action='store_true',
                   help='Use only with an independently validated short-path simulator installation')
    args, extra = p.parse_known_args()
    if extra and extra[0] == '--':
        extra = extra[1:]
    if args.episodes < 1:
        p.error('episodes must be positive')
    if any(value.split('=', 1)[0] == '--xr1_execute_steps' for value in extra):
        p.error('Execute steps are selected by task_mapping.json, not evaluator overrides')
    contract = load_contract(args.task)
    challenge = args.challenge_root.resolve()
    models = args.model_dir.resolve()
    mapping = json.loads((ROOT / 'task_mapping.json').read_text())
    if set(mapping['tasks']) != set(TASKS):
        raise ValueError('Submission must map all ten tasks')
    artifact = models / contract['artifact']
    manifest = json.loads((models / 'manifest.json').read_text())
    # Resolve every mapped checkpoint and hash all required inference assets before GPU startup.
    required = set()
    for entry in mapping['tasks'].values():
        required.update([entry['artifact'] + '/model.pt', entry['artifact'] + '/config.yaml'])
    if not required.issubset(manifest['files']):
        raise ValueError('Model manifest does not cover all task artifacts')
    for exe in [args.sim_python, args.model_python]:
        if not shutil.which(exe):
            raise FileNotFoundError(exe)
    evaluator = challenge / 'scripts/eval_policy.py'
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    assets = check_task_assets(challenge, args.task, args.setting)
    if not args.preflight_only:
        gpu_env = os.environ.copy()
        # The launcher may itself run through the model wrapper (Python 3.12).
        # Do not expose its site-packages to simulator Python (Python 3.11).
        # A requested wrapper can add its own runtime paths after this reset.
        gpu_env['PYTHONPATH'] = ''
        gpu_env.setdefault('OMP_NUM_THREADS', '4')
        gpu_env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE='1', PYTHONFAULTHANDLER='1')
        for exe in dict.fromkeys([args.sim_python, args.model_python]):
            subprocess.run([exe, str(ROOT / 'tools/gpu_preflight.py')], env=gpu_env,
                           timeout=120, check=True)
    for name, record in manifest['files'].items():
        path = models / name
        if not path.resolve().is_relative_to(models):
            raise ValueError('Unsafe artifact path')
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    dest = challenge / 'policy/xr1_submission'
    source = ROOT / 'policy/xr1_submission'
    if dest.exists() or dest.is_symlink():
        if dest.resolve() != source.resolve():
            raise FileExistsError(f'Refusing to replace another policy: {dest}')
    else:
        dest.symlink_to(source, target_is_directory=True)
    if args.preflight_only:
        print(json.dumps({'passed': True, 'task': args.task, 'artifact': str(artifact),
                          'execute_steps': contract['execute_steps'],
                          'all_tasks_mapped': True, 'manifest_verified': True}))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # Match the accepted workers; honor an explicit organizer setting.
    env.setdefault('OMP_NUM_THREADS', '4')
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONFAULTHANDLER='1')
    runtime = None
    runtime_record = {'isolated': False}
    if not args.no_isolate_simulator:
        runtime, runtime_env, runtime_record = prepare_sim_runtime(challenge,
            Path(env['EMBODICHAIN_ROOT']), Path(env['EMBODICHAIN_DATA_ROOT']))
        env.update(runtime_env)
        challenge = runtime / 'c'
        evaluator = challenge / 'scripts/eval_policy.py'
    # No dependency on a developer checkout or ambient PYTHONPATH in inference.
    model_env = env.copy()
    model_env['PYTHONPATH'] = str(ROOT / 'runtime')
    sim_env = env.copy()
    sim_paths = [str(challenge), env.get('EMBODICHAIN_ROOT', '')]
    if env.get('EMBODICHAIN_ROOT'):
        sim_paths.append(str(Path(env['EMBODICHAIN_ROOT']) / 'embodichain_tasks'))
    sim_env['PYTHONPATH'] = os.pathsep.join(x for x in sim_paths if x)
    server_command = [args.model_python, str(ROOT / 'runtime/server.py'),
        '--config', str(artifact / 'config.yaml'), '--checkpoint', str(artifact / 'model.pt'),
        '--processor', str(models / 'processor'), '--port', str(args.port),
        '--log-root', str(output / 'server'), '--attention-backend', args.attention_backend]
    eval_command = [args.sim_python, '-u', str(ROOT / 'tools/run_official_evaluator.py'), str(evaluator), '--config',
        str(source / 'deploy_policy.yml'), '--overrides', '--task_name', args.task,
        '--setting', args.setting, '--seed', str(args.seed), '--max_episodes', str(args.episodes),
        '--xr1_execute_steps', str(contract['execute_steps']),
        '--xr1_port', str(args.port), '--xr1_trace_root', str(output / 'traces'),
        '--eval_result_dir', str(output / 'raw'), '--eval_video_log', str(args.video),
        '--eval_video_obs_keys', "['cam_high','cam_left_wrist','cam_right_wrist']", *extra]
    record = {'task': args.task, 'artifact': str(artifact), 'model_manifest': manifest,
        'evaluator_sha256': sha(evaluator), 'server_command': server_command,
        'task_asset_preflight': assets, 'simulator_runtime': runtime_record,
        'eval_command': eval_command, 'evaluator_lifecycle': 'defer native close until official metrics are written',
        'evaluator_wrapper_sha256': sha(ROOT / 'tools/run_official_evaluator.py'),
        'attention_backend': args.attention_backend,
        'omp_num_threads': env.get('OMP_NUM_THREADS'),
        'task_mapping_sha256': sha(ROOT / 'task_mapping.json'),
        'horizon': contract['horizon'], 'execute_steps': contract['execute_steps'],
        'prefix_steps': contract['prefix_steps'], 'policy_seed': 20260908}
    (output / 'launch_contract.json').write_text(json.dumps(record, indent=2) + '\n')
    server = evaluator_process = None
    status = 1
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (output / 'server.log').open('w') as log:
            server = subprocess.Popen(server_command, env=model_env, cwd=ROOT / 'runtime',
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.startup_timeout
            while not (output / 'server/READY').exists():
                if server.poll() is not None:
                    raise RuntimeError('Inference server failed; see server.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('Inference server startup deadline exceeded')
                time.sleep(1)
            with (output / 'evaluation.log').open('w') as eval_log:
                evaluator_process = subprocess.Popen(eval_command, env=sim_env, cwd=challenge,
                    stdout=eval_log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 600 + args.episodes * 1800
                while evaluator_process.poll() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Evaluator exceeded its execution deadline')
                    with (output / 'evaluation.log').open('rb') as stream:
                        stream.seek(max(0, (output / 'evaluation.log').stat().st_size - 8192))
                        tail = stream.read()
                    if b'Assertion failed at /builds/Engine/dexsim/' in tail:
                        raise RuntimeError('Native DexSim assertion; stop and inspect evaluation.log')
                    time.sleep(1)
                status = evaluator_process.returncode
            if status:
                raise RuntimeError(f'Evaluator exited with {status}; see evaluation.log')
            results = list((output / 'raw').rglob('evaluation_metrics.json'))
            if len(results) != 1:
                raise RuntimeError(f'Expected exactly one metrics file, got {len(results)}')
            data = json.loads(results[0].read_text())
            if len(data['episodes']) != args.episodes:
                raise RuntimeError('Evaluator did not complete all requested episodes')
            shutil.copy2(results[0], output / 'metrics.json')
            status = 0
            print(json.dumps(data['summary']), flush=True)
    except BaseException:
        status = status if status else 1
        raise
    finally:
        finish(output, (evaluator_process, server), runtime, status)


if __name__ == '__main__':
    main()
