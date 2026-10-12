"""Validate new runtime evidence while retaining the historical acceptance chain."""
import hashlib
import json


CHANGED_RUNTIME = {
    'runtime/server.py', 'tools/launch.py', 'tools/run_official_evaluator.py',
    'tools/gpu_preflight.py',
}
BUILD_TOOLS = {'tools/build_release.py', 'tools/verify_runtime_diagnostics.py', 'tools/verify_original3.py'}


def verify(root, manifest):
    # This evidence describes the exact previous mapping, before original3 selection.
    manifest = dict(manifest)
    prior = root / 'verification/pre_original3_task_mapping.json'
    manifest['task_mapping.json'] = hashlib.sha256(prior.read_bytes()).hexdigest()
    folder = root / 'verification/runtime_diagnostics_20261012'
    proof = json.loads((folder / 'proof.json').read_text())
    if proof['baseline_github_revision'] != '3935166d34285543e2c4af236e31710821dd939a':
        raise RuntimeError('Unexpected diagnostic baseline')
    required = {'baseline_source_sha256.json', 'candidate_source_sha256.json', 'replay_result.json', 'replay_server_source.py', 'final_resource_check.json'}
    for task in ('handle_basket', 'sample_loading', 'mixer_operating'):
        required.update(f'{task}/{name}' for name in ('audit.json', 'metrics.json', 'launch_contract.json', 'cleanup.json'))
    if not required.issubset(proof['evidence_sha256']):
        raise RuntimeError('Incomplete diagnostic evidence manifest')
    for name, expected in proof['evidence_sha256'].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()):
            raise RuntimeError('Unsafe evidence path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Diagnostic evidence changed: {name}')
    baseline = json.loads((folder / 'baseline_source_sha256.json').read_text())
    tested = json.loads((folder / 'candidate_source_sha256.json').read_text())
    resources = json.loads((folder / 'final_resource_check.json').read_text())
    if not resources['passed'] or resources['compute_processes'] or not resources['port_23872_closed'] or resources['gpu_used_MiB'] >= 1024:
        raise RuntimeError('Diagnostic worker resources were not released')
    # Every executable runtime file must be exactly the file used in real episodes.
    for name, digest in manifest.items():
        runtime_file = name.startswith(('runtime/', 'policy/', 'tools/')) and name not in BUILD_TOOLS
        if runtime_file or name == 'task_mapping.json':
            if tested.get(name) != digest:
                raise RuntimeError(f'Runtime was not accepted end to end: {name}')
            if name not in CHANGED_RUNTIME and baseline.get(name) != digest:
                raise RuntimeError(f'Unexpected runtime change: {name}')
    for name in tested:
        if name.startswith(('runtime/', 'policy/', 'tools/')) and name not in BUILD_TOOLS and name not in manifest:
            raise RuntimeError(f'Accepted runtime file removed: {name}')
    accepted = {}
    mapping = json.loads(prior.read_text())
    expected_artifacts = json.loads((root / 'verification/artifact_manifest.json').read_text())['files'].copy()
    for task, prefix in (('item_assembly', 'assembly'), ('water_pouring', 'water_pouring'),
                         ('handle_basket', 'handle_basket'), ('items_handover', 'items_handover'),
                         ('table_rearrangement', 'table_rearrangement'), ('drawer_open_place', 'drawer_open_place'),
                         ('click_bell', 'click_bell')):
        export = json.loads((root / f'verification/{prefix}_export.json').read_text())
        artifact = mapping['tasks'][task]['artifact']
        for filename, key in (('model.pt', 'model_sha256'), ('config.yaml', 'config_sha256')):
            expected_artifacts[f'{artifact}/{filename}'] = {'sha256': export[key]}
    for task in ('handle_basket', 'sample_loading', 'mixer_operating'):
        audit = json.loads((folder / task / 'audit.json').read_text())
        metrics_path = folder / task / 'metrics.json'
        contract = json.loads((folder / task / 'launch_contract.json').read_text())
        cleanup = json.loads((folder / task / 'cleanup.json').read_text())
        if not audit['passed'] or audit['task'] != task or audit['episodes'] != 1:
            raise RuntimeError(f'Missing real diagnostic acceptance: {task}')
        if audit['official_revision'] != '8818c82df5fa2522809b8105e7018950dcc9a9a8' or audit['eval_reset_sync_steps'] != 0:
            raise RuntimeError(f'Wrong official evaluation contract: {task}')
        actual_artifacts = contract['model_manifest']['files']
        if set(actual_artifacts) != set(expected_artifacts):
            raise RuntimeError(f'Model artifact inventory changed: {task}')
        for name, record in expected_artifacts.items():
            if actual_artifacts[name]['sha256'] != record['sha256']:
                raise RuntimeError(f'Wrong model or processor: {task}: {name}')
        if hashlib.sha256(metrics_path.read_bytes()).hexdigest() != audit['metrics_sha256']:
            raise RuntimeError(f'Diagnostic metrics changed: {task}')
        if (contract['horizon'], contract['execute_steps'], contract['prefix_steps'], contract['policy_seed']) != (30, 29, 0, 20260908):
            raise RuntimeError(f'Diagnostic policy settings changed: {task}')
        if contract['evaluator_wrapper_sha256'] != manifest['tools/run_official_evaluator.py']:
            raise RuntimeError(f'Wrong evaluated wrapper: {task}')
        if contract['task_mapping_sha256'] != manifest['task_mapping.json']:
            raise RuntimeError(f'Wrong evaluated mapping: {task}')
        if contract['attention_backend'] != 'sdpa' or contract['omp_num_threads'] != '4':
            raise RuntimeError(f'Unexpected execution environment: {task}')
        if cleanup['primary_exit_code'] != 0 or cleanup['warnings']:
            raise RuntimeError(f'Diagnostic cleanup failed: {task}')
        accepted[task] = {'episodes': 1, 'successes': audit['summary']['success_count'],
                          'gpu_peak_MiB': audit['gpu_peak_MiB'], 'metrics_sha256': audit['metrics_sha256']}
    replay = json.loads((folder / 'replay_result.json').read_text())
    if replay['published_candidate_max_abs'] != [0.0] * 7:
        raise RuntimeError('Real-input action replay differs')
    formal = root / 'verification/selected_official_n50'
    evidence = json.loads((formal / 'evidence_sha256.json').read_text())
    for name, digest in evidence.items():
        if hashlib.sha256((formal / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Historical formal evidence changed: {name}')
    formal_results = []
    for record in json.loads((formal / 'summary.json').read_text()):
        task = record['task']
        report = json.loads((formal / task / 'result.json').read_text())
        metrics_path = formal / task / 'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        if not report['passed'] or report['episodes'] != 50 or len(metrics['episodes']) != 50:
            raise RuntimeError(f'Incomplete historical formal run: {task}')
        if hashlib.sha256(metrics_path.read_bytes()).hexdigest() != report['metrics_sha256']:
            raise RuntimeError(f'Historical formal metrics changed: {task}')
        if (report['execute_steps'], report['prefix_steps'], report['eval_reset_sync_steps']) != (20 if task == 'items_handover' else 29, 0, 0):
            raise RuntimeError(f'Wrong historical formal policy contract: {task}')
        artifact = mapping['tasks'][task]['artifact']
        if report['model_artifact'].rsplit('/', 1)[-1] != artifact:
            raise RuntimeError(f'Historical checkpoint selection changed: {task}')
        for name, expected in expected_artifacts.items():
            if name.startswith(('processor/', artifact + '/')):
                if report['model_manifest']['files'][name]['sha256'] != expected['sha256']:
                    raise RuntimeError(f'Historical model/processor changed: {task}: {name}')
        successes = sum(bool(row['success']) for row in metrics['episodes'])
        if successes != record['successes'] or successes != metrics['summary']['success_count']:
            raise RuntimeError(f'Historical success total mismatch: {task}')
        formal_results.append(record)
    expected_tasks = {'click_bell', 'handle_basket', 'water_pouring', 'table_rearrangement',
                      'items_handover', 'drawer_open_place', 'item_assembly'}
    if len(formal_results) != 7 or {r['task'] for r in formal_results} != expected_tasks:
        raise RuntimeError('Historical formal evidence must cover the seven selected tasks')
    return baseline, {'baseline_github_revision': proof['baseline_github_revision'],
                      'historical_runtime_diagnostic_acceptance': accepted,
                      'historical_selected_official_n50': formal_results,
                      'native_initialization_crash_fixed': False,
                      'fresh_install_verified': False,
                      'replay_scope': 'Seven real basket inputs; intermediate diagnostic server source retained with evidence'}
