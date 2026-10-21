"""Official-structure contract: only policy/<name>/ is copied into the organizer checkout.

eval.sh must run the unchanged official scripts/eval_policy.py; get_model owns the
separate-Python inference process. A stub "XR1 Python" stands in for the model.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'policy/xr1_submission'

STUB_SERVER = '''#!{python}
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
root = Path(value('--log-root'))
root.mkdir(parents=True, exist_ok=True)
(root / 'stub.json').write_text(json.dumps({{'argv': args, 'pid': os.getpid(), 'cwd': os.getcwd(),
    'env': {{k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'HF_HUB_OFFLINE')}}}}))
mode = os.environ.get('XR1_STUB_MODE', 'ready')
if mode == 'exit':
    print('stub server failure', flush=True)
    sys.exit(3)
if mode == 'ready':
    (root / 'READY').write_text(json.dumps({{'port': int(value('--port'))}}))
while True:
    time.sleep(1)
'''

FAKE_EVALUATOR = '''import json, os, sys
from pathlib import Path
Path(os.environ['FAKE_EVALUATOR_RECORD']).write_text(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(),
    'env': {k: os.environ.get(k) for k in ('EMBODICHAIN_SIM_EXIT_PROCESS', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH')}}))
'''


def alive(pid):
    try:
        state = Path(f'/proc/{pid}/status').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return '\nState:\tZ' not in state


class Organizer:
    """Organizer checkout with only the copied policy directory, plus a model repository."""

    def __init__(self, root, name='xr1_submission'):
        self.root = root
        self.challenge = root / 'organizer'
        self.policy = self.challenge / 'policy' / name
        shutil.copytree(SOURCE, self.policy, ignore=shutil.ignore_patterns('__pycache__'))
        (self.challenge / 'scripts').mkdir()
        (self.challenge / 'scripts/eval_policy.py').write_text(FAKE_EVALUATOR)
        self.stub = root / 'xr1-python'
        self.stub.write_text(STUB_SERVER.format(python=sys.executable))
        self.stub.chmod(0o755)
        self.models = root / 'models'
        (self.models / 'processor').mkdir(parents=True)
        manifest = {'files': {}}
        mapping = json.loads((self.policy / 'task_mapping.json').read_text())
        for entry in mapping['tasks'].values():
            artifact = self.models / entry['artifact']
            artifact.mkdir(exist_ok=True)
            for name in ('model.pt', 'config.yaml'):
                path = artifact / name
                path.write_bytes(f'fixture {entry["artifact"]}/{name}\n'.encode())
                manifest['files'][path.relative_to(self.models).as_posix()] = {
                    'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        (self.models / 'manifest.json').write_text(json.dumps(manifest))

    def eval_sh(self, *args, **env):
        record = self.root / 'evaluator.json'
        record.unlink(missing_ok=True)
        environment = {**os.environ, 'PYTHON_BIN': sys.executable, 'XR1_PYTHON': str(self.stub),
                       'FAKE_EVALUATOR_RECORD': str(record), 'CUDA_VISIBLE_DEVICES': '7', **env}
        result = subprocess.run(['bash', str(self.policy / 'eval.sh'), *args], cwd=self.root,
                                env=environment, text=True, capture_output=True, timeout=30)
        return result, (json.loads(record.read_text()) if record.exists() else None)

    def adapter(self, body, mode='ready', **env):
        """Import the adapter the way the official evaluator does and run body."""
        code = textwrap.dedent(f'''
            import importlib, json, sys
            sys.path[:0] = [{str(self.challenge)!r}, {str(self.challenge / 'policy')!r}]
            policy = importlib.import_module('policy.{self.policy.name}')
            def config(task, **extra):
                return {{'task_name': task, 'xr1_model_dir': {str(self.models)!r},
                        'xr1_python': {str(self.stub)!r}, 'xr1_port': None, 'gpu_id': 1,
                        'xr1_startup_timeout': 5, **extra}}
        ''') + textwrap.dedent(body)
        environment = {**os.environ, 'XR1_STUB_MODE': mode, **env}
        environment.pop('CUDA_VISIBLE_DEVICES', None)
        return subprocess.Popen([sys.executable, '-c', code], cwd=self.root, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class PolicyDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='xr1_policy_')
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_eval_sh_runs_unchanged_official_evaluator(self):
        for name in ('xr1_submission', 'renamed_team_policy'):
            organizer = Organizer(self.root / name, name)
            result, record = organizer.eval_sh('items_handover', 'random', str(organizer.models), '1',
                                               '--max_episodes', '5', '--eval_video_log', 'true')
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = record['argv']
            self.assertEqual(record['cwd'], str(organizer.challenge))
            self.assertEqual(argv[:3], ['--config', f'policy/{name}/deploy_policy.yml', '--overrides'])
            overrides = dict(zip(argv[3::2], argv[4::2]))
            self.assertEqual(overrides['--policy_name'], name)
            self.assertEqual(overrides['--task_name'], 'items_handover')
            self.assertEqual(overrides['--setting'], 'random')
            self.assertEqual(overrides['--xr1_model_dir'], str(organizer.models))
            self.assertEqual(overrides['--xr1_python'], str(organizer.stub))
            self.assertEqual(overrides['--gpu_id'], '1')
            self.assertEqual(overrides['--max_episodes'], '5')
            self.assertEqual(overrides['--eval_video_log'], 'True')
            self.assertEqual(record['env']['EMBODICHAIN_SIM_EXIT_PROCESS'], '0')
            self.assertIsNone(record['env']['CUDA_VISIBLE_DEVICES'])
            self.assertTrue(record['env']['PYTHONPATH'].startswith(f'{organizer.challenge}:'))
        # Only policy/<name>/ was added to the organizer checkout.
        self.assertEqual(sorted(p.name for p in organizer.challenge.iterdir()), ['policy', 'scripts'])

    def test_eval_sh_without_gpu_or_extra_arguments(self):
        organizer = Organizer(self.root)
        result, record = organizer.eval_sh('click_bell', 'random', str(organizer.models))
        self.assertEqual(result.returncode, 0, result.stderr)
        overrides = dict(zip(record['argv'][3::2], record['argv'][4::2]))
        self.assertEqual(len(record['argv'][3:]), 2 * len(overrides))
        self.assertEqual(overrides['--gpu_id'], '0')
        self.assertEqual(overrides['--task_name'], 'click_bell')

    def test_eval_sh_rejects_missing_inputs_before_evaluation(self):
        organizer = Organizer(self.root)
        result, record = organizer.eval_sh('click_bell', 'random', str(self.root / 'missing'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('model directory does not exist', result.stderr)
        self.assertIsNone(record)
        result, record = organizer.eval_sh('click_bell', 'random', str(organizer.models),
                                           XR1_PYTHON=str(self.root / 'missing-python'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('install_inference.sh', result.stderr)
        self.assertIsNone(record)
        result, record = organizer.eval_sh('click_bell', 'random')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Usage', result.stderr)

    def test_get_model_owns_inference_process(self):
        organizer = Organizer(self.root)
        proc = organizer.adapter('''
            first = policy.get_model(config('items_handover'))
            second = policy.get_model(config('click_bell'))
            print(json.dumps([[m.port, m.k, str(m.server.log_root), m.server.proc.pid] for m in (first, second)]))
        ''')
        out, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, err)
        (port1, k1, log1, pid1), (port2, k2, log2, pid2) = json.loads(out.strip().splitlines()[-1])
        self.assertNotEqual(port1, port2)
        self.assertEqual((k1, k2), (20, 29))
        stub = json.loads((Path(log1) / 'stub.json').read_text())
        argv = dict(zip(stub['argv'][1::2], stub['argv'][2::2]))
        self.assertEqual(stub['argv'][0], str(organizer.policy / 'runtime/server.py'))
        self.assertEqual(argv['--checkpoint'], str(organizer.models / 'items_handover_180k/model.pt'))
        self.assertEqual(argv['--config'], str(organizer.models / 'items_handover_180k/config.yaml'))
        self.assertEqual(argv['--processor'], str(organizer.models / 'processor'))
        self.assertEqual(argv['--port'], str(port1))
        self.assertEqual(stub['cwd'], str(organizer.policy / 'runtime'))
        self.assertEqual(stub['env'], {'CUDA_VISIBLE_DEVICES': '1',
                                       'PYTHONPATH': str(organizer.policy / 'runtime'), 'HF_HUB_OFFLINE': '1'})
        # atexit stopped both inference processes when the evaluator exited normally.
        time.sleep(0.5)
        self.assertFalse(alive(pid1) or alive(pid2))

    def test_inference_process_stops_when_evaluator_is_killed(self):
        organizer = Organizer(self.root)
        proc = organizer.adapter('''
            model = policy.get_model(config('click_bell'))
            print('PID', model.server.proc.pid, flush=True)
            import time; time.sleep(60)
        ''')
        line = proc.stdout.readline()
        while not line.startswith('PID '):
            if not line:
                self.fail('Adapter exited before reporting its worker PID: ' + proc.stderr.read())
            line = proc.stdout.readline()
        pid = int(line.split()[1])
        self.assertTrue(alive(pid))
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        proc.stdout.close()
        proc.stderr.close()
        deadline = time.monotonic() + 10
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertFalse(alive(pid))

    def test_simulation_cleanup_runs_once_after_metrics_and_worker_shutdown(self):
        organizer = Organizer(self.root)
        record = self.root / 'shutdown.jsonl'
        proc = organizer.adapter(f'''
            import types
            from pathlib import Path
            record = Path({str(record)!r})
            workers = []
            constraints = ['fixture_weld']
            def remove(name):
                assert all(m.server.proc.poll() is not None for m in workers)
                constraints.remove(name)
                with record.open('a') as out: out.write('constraint_removed\\n')
                return True
            def flush():
                assert record.read_text().splitlines() == ['metrics_written', 'constraint_removed']
                assert not constraints
                assert all(m.server.proc.poll() is not None for m in workers)
                with record.open('a') as out: out.write('simulation_flushed\\n')
            instance = types.SimpleNamespace(get_rigid_constraint_uid_list=lambda: list(constraints),
                                             remove_rigid_constraint=remove)
            sys.modules['embodichain.lab.sim.sim_manager'] = types.SimpleNamespace(
                SimulationManager=types.SimpleNamespace(flush_cleanup_queue=flush,
                    is_instantiated=lambda: True, get_instance=lambda: instance))
            workers += [policy.get_model(config('click_bell')),
                        policy.get_model(config('items_handover'))]
            record.write_text('metrics_written\\n')
        ''')
        out, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, err)
        self.assertNotIn('Exception ignored in atexit', err)
        self.assertEqual(record.read_text().splitlines(), ['metrics_written', 'constraint_removed', 'simulation_flushed'])
        self.assertEqual(out.count('XR1_SIM_CONSTRAINTS_RELEASED 1'), 1)
        self.assertEqual(out.count('XR1_SIM_CLEANUP_DONE'), 1)

    def failure(self, organizer, mode='ready', **extra):
        proc = organizer.adapter(f'''
            try:
                policy.get_model(config('click_bell', **{extra!r}))
            except Exception as exc:
                print(json.dumps(str(exc)))
        ''', mode=mode)
        out, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, err)
        return json.loads(out.strip().splitlines()[-1])

    def test_startup_failures_are_explicit_and_leave_no_process(self):
        organizer = Organizer(self.root)
        for mode, message in [('exit', 'exited with 3'), ('hang', 'not ready after')]:
            log_root = self.root / f'log_{mode}'
            error = self.failure(organizer, mode, xr1_startup_timeout=2, xr1_log_root=str(log_root))
            self.assertIn(message, error)
            if mode == 'exit':
                self.assertIn('stub server failure', error)
            self.assertFalse(alive(json.loads((log_root / 'stub.json').read_text())['pid']))

    def test_incomplete_model_repository_fails_before_startup(self):
        organizer = Organizer(self.root)
        log_root = self.root / 'log'
        (organizer.models / 'processor').rename(organizer.models / 'processor.moved')
        self.assertIn('processor not found', self.failure(organizer, xr1_log_root=str(log_root)))
        (organizer.models / 'processor.moved').rename(organizer.models / 'processor')
        (organizer.models / 'click_bell_160k_ft3k/model.pt').write_bytes(b'tampered\n')
        self.assertIn('does not match', self.failure(organizer, xr1_log_root=str(log_root)))
        (organizer.models / 'manifest.json').unlink()
        self.assertIn('complete model repository', self.failure(organizer, xr1_log_root=str(log_root)))
        self.assertFalse(log_root.exists())


if __name__ == '__main__':
    unittest.main()
