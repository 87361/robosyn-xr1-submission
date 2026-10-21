"""Build the official-structure release; bind it to the audited source and the tested policy bytes."""
import hashlib
import json
from pathlib import Path
import zipfile

TASKS = ('click_bell handle_basket water_pouring table_rearrangement items_handover '
         'drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading').split()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    for name in ('README.md', 'SUBMISSION_TEXT.md'):
        if 'TEN_TASK_ACCEPTANCE_PENDING' in (root / name).read_text():
            raise RuntimeError(f'Update the accepted results before packaging: {name}')
    evidence = root / 'verification/policy_structure_20261021'
    audit = json.loads((evidence / 'source_audit.json').read_text())
    validation = json.loads((evidence / 'validation.json').read_text())
    if not validation['passed'] or sorted(validation['tasks']) != sorted(TASKS) \
            or validation['episodes_completed'] != len(TASKS):
        raise RuntimeError('Missing ten-task official-structure end-to-end acceptance')
    if not validation.get('simulation_cleanup_required') or validation.get('reused_tasks'):
        raise RuntimeError('Require fresh ten-task acceptance of the simulation cleanup hook')
    for task, result in validation['tasks'].items():
        if not result['passed'] or result['exit_code'] != 0 or result['native_fault_bytes'] != 0 \
                or not result['server_stopped_and_gpu_released'] or not result['port_released']:
            raise RuntimeError(f'Incomplete runtime/cleanup acceptance: {task}')
        if result.get('simulation_cleanup_markers') != 1:
            raise RuntimeError(f'Official simulation cleanup did not finish: {task}')
        metrics = evidence / task / 'evaluation_metrics.json'
        if digest(metrics) != result['metrics_sha256']:
            raise RuntimeError(f'Official metrics changed: {task}')
    files = sorted(p for p in root.rglob('*') if p.is_file()
        and not set(p.relative_to(root).parts) & {'dist', '__pycache__', '.git', '.venv', '.xr1_env'}
        and p.relative_to(root).parts[0] not in {'models', 'outputs'}
        and p.name not in {'SOURCE_SHA256.json', 'PUBLICATION_VALIDATION.json'})
    manifest = {p.relative_to(root).as_posix(): digest(p) for p in files}
    policy_files = {k: v for k, v in manifest.items() if k.startswith('policy/xr1_submission/')}
    audited = {k: v['sha256'] for k, v in {**audit['unchanged'], **audit['modified']}.items()}
    if policy_files != audited:
        raise RuntimeError('Policy files differ from the source audit')
    if policy_files != json.loads((evidence / 'policy_source_sha256.json').read_text()):
        raise RuntimeError('Release differs from the policy directory tested on Baidu')
    publication = {
        'scope': 'Official-structure adapter; one public-setting episode per task',
        'base_code_commit': audit['base_code_commit'], 'hf_revision': audit['hf_revision'],
        'inference_runtime_unchanged': sorted(audit['unchanged']),
        'validation': validation, 'formal_ten_task_rerun': False,
        'historical_public_score': '184/200; original separate N20 batches only',
    }
    (root / 'PUBLICATION_VALIDATION.json').write_text(json.dumps(publication, indent=2) + '\n')
    manifest['PUBLICATION_VALIDATION.json'] = digest(root / 'PUBLICATION_VALIDATION.json')
    (root / 'SOURCE_SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    dist = root / 'dist'
    dist.mkdir(exist_ok=True)
    target = dist / 'robosyn-xr1-submission.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted([*manifest, 'SOURCE_SHA256.json']):
            archive.write(root / name, 'robosyn-xr1-submission/' + name)
    target.with_suffix('.zip.sha256').write_text(f'{digest(target)}  {target.name}\n')
    print(json.dumps({'archive': str(target), 'files': len(manifest), 'passed': True}))


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1])
