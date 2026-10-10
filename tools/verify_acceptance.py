"""Verify a real multi-episode run, server resets, metrics, videos, and clean evaluator."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--challenge-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=2)
    a = p.parse_args()
    run = a.run
    assert (run / 'exit.code').read_text().strip() == '0'
    metrics = json.loads((run / 'metrics.json').read_text())
    assert len(metrics['episodes']) == a.episodes >= 2
    assert metrics['config']['task'] == 'click_bell'
    assert len({e['seed'] for e in metrics['episodes']}) == a.episodes
    assert metrics['summary']['success_count'] == sum(bool(e['success']) for e in metrics['episodes'])
    assert metrics['summary']['success_count'] == a.episodes, 'Acceptance requires successful bell presses'
    requests = [json.loads(line) for line in (run / 'server/requests.jsonl').read_text().splitlines()]
    groups = {}
    for row in requests:
        groups.setdefault(row['episode_id'], []).append(row)
        assert row['policy_seed'] == 20260908 and row['prefix_length'] == 0
    assert len(groups) == a.episodes
    for rows in groups.values():
        assert [r['chunk_index'] for r in rows] == list(range(len(rows)))
    videos = []
    for f in sorted((run / 'raw').rglob('*.mp4')):
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'stream=width,height,nb_frames:format=duration', '-of', 'json', str(f)], text=True))
        assert float(probe['format']['duration']) > 0
        videos.append({'path': str(f.relative_to(run)), 'bytes': f.stat().st_size, 'probe': probe})
    assert len(videos) == a.episodes
    contract = json.loads((run / 'launch_contract.json').read_text())
    port = int(contract['server_command'][contract['server_command'].index('--port') + 1])
    with socket.socket() as conn:
        conn.settimeout(2)
        assert conn.connect_ex(('127.0.0.1', port)) != 0, 'Inference listener was not cleaned up'
    challenge = a.challenge_root
    commit = subprocess.check_output(['git', '-C', str(challenge), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(challenge), 'diff', '--exit-code', 'HEAD', '--',
                    'scripts/eval_policy.py', 'configs', 'robosynchallenge'], check=True)
    packages = {}
    for name in ['torch', 'torchvision', 'transformers', 'numpy', 'liger-kernel', 'mmengine', 'pillow']:
        packages[name] = importlib.metadata.version(name)
    record = {'passed': True, 'task': 'click_bell', 'episodes': a.episodes,
        'successes': metrics['summary']['success_count'], 'summary': metrics['summary'],
        'challenge_commit': commit, 'evaluator_unmodified': True,
        'evaluator_sha256': hashlib.sha256((challenge / 'scripts/eval_policy.py').read_bytes()).hexdigest(),
        'attention_backend': contract['attention_backend'],
        'inference_listener_cleaned_up': True,
        'source_code_sha256': {str(f.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(f.read_bytes()).hexdigest()
            for directory in ('policy', 'runtime', 'tools')
            for f in (Path(__file__).resolve().parents[1] / directory).rglob('*')
            if f.is_file() and f.suffix in ('.py', '.sh', '.yml')},
        'model_versions': packages, 'gpu': subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
        'episode_rng_reset_verified': True, 'videos': videos,
        'metrics_sha256': hashlib.sha256((run / 'metrics.json').read_bytes()).hexdigest(),
        'boundaries': ['real bell end-to-end only; not a ten-task release reevaluation',
                       'not verified on RTX 5090', 'not uploaded or submitted to the organizers']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
