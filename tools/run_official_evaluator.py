"""Run the unchanged official evaluator, then release native resources after its metrics."""
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

_fault_stream = None


def record_process(arguments):
    # Diagnostics are best-effort and must not prevent an official run.
    try:
        global _fault_stream
        import faulthandler
        import importlib.metadata as metadata
        import platform
        root = Path(arguments[arguments.index('--eval_result_dir') + 1]).parent
        # The official expert check redirects fd 2. Keep a separate descriptor
        # so native faults in that phase are still retained.
        _fault_stream = (root / 'native_fault.log').open('a')
        faulthandler.enable(file=_fault_stream, all_threads=True)
        versions = {}
        for name in ('torch', 'numpy', 'gymnasium', 'embodichain', 'dexsim_engine'):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        (root / 'sim_process.json').write_text(json.dumps({
            'pid': os.getpid(), 'proc_status': Path('/proc/self/status').read_text(),
            'python': sys.version, 'executable': sys.executable, 'hostname': platform.node(),
            'versions': versions, 'faulthandler': os.environ.get('PYTHONFAULTHANDLER'),
        }, indent=2) + '\n')
    except Exception as exc:
        try:
            print(f'PROCESS_DIAGNOSTIC_WARNING: {exc}', file=sys.stderr, flush=True)
        except Exception:
            pass


def fail_without_native_teardown():
    try:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    except BaseException:
        pass
    # Even a full log filesystem must not allow native close to replace failure.
    os._exit(1)

def main():
    script, *arguments = sys.argv[1:]
    record_process(arguments)
    import gymnasium as gym
    original_make = gym.make
    deferred = []
    environments = []

    def make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        environments.append(env)  # Keep native resources alive until metrics are written.
        close = env.close
        requested = False
        def close_after_metrics(*close_args, **close_kwargs):
            nonlocal requested
            if not requested:
                deferred.append((close, close_args, close_kwargs))
                requested = True
        env.close = close_after_metrics
        return env

    gym.make = make
    sys.argv = [script, *arguments]
    sys.path.insert(0, str(Path(script).resolve().parent))
    phase = None
    try:
        runpy.run_path(script, run_name="__main__")
        output = Path(arguments[arguments.index("--eval_result_dir")+1])
        expected = int(arguments[arguments.index("--max_episodes")+1])
        task = arguments[arguments.index("--task_name")+1]
        files = list(output.rglob("evaluation_metrics.json"))
        assert len(files) == 1, "Official evaluator did not write exactly one metrics file"
        data = json.loads(files[0].read_text())
        assert data["config"]["task"] == task
        assert len(data["episodes"]) == expected
        assert data["summary"]["episode_count"] == expected
        assert deferred, "Official evaluator did not request environment close"
        phase = output.parent / "evaluator_lifecycle.json"
        phase.write_text(json.dumps({"official_evaluator_returned":True,
            "official_metrics_verified_before_teardown":True,
            "deferred_close_count":len(deferred), "native_teardown":"started"},indent=2)+"\n")
        print("OFFICIAL_METRICS_VERIFIED_BEFORE_NATIVE_TEARDOWN", flush=True)
    except BaseException:
        # A native close must never overwrite an evaluator failure's exit code.
        fail_without_native_teardown()
    finally:
        gym.make = original_make
    try:
        for close, args, kwargs in deferred:
            close(*args, **kwargs)
        record = json.loads(phase.read_text())
        record["native_teardown"] = "returned"
        phase.write_text(json.dumps(record,indent=2)+"\n")
    except BaseException:
        fail_without_native_teardown()
    sys.stdout.flush()
    sys.stderr.flush()
    # The evaluator and explicit close completed; avoid a second native destruction.
    os._exit(0)

if __name__ == "__main__":
    main()

