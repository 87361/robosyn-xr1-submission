"""Create a code-only archive with a checksum manifest; never include model/cache files."""
import hashlib
import json
from pathlib import Path
import zipfile
import yaml

root = Path(__file__).resolve().parents[1]
selected = sorted(p for p in root.rglob('*') if p.is_file()
                  and not set(p.relative_to(root).parts) & {'dist', '__pycache__', '.git', '.venv'}
                  and p.relative_to(root).parts[0] not in {'models', 'outputs'}
                  and p.relative_to(root).as_posix() not in {'SOURCE_SHA256.json', 'PUBLICATION_VALIDATION.json'}
                  and (p.suffix in {'.py', '.sh', '.yml', '.json', '.md', '.txt'}
                       or p.name in {'LICENSE', '.gitignore'}))
dist = root / 'dist'
dist.mkdir(exist_ok=True)
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
accepted = json.loads((root / 'verification/acceptance.json').read_text())
if not accepted['passed']:
    raise RuntimeError('No successful end-to-end acceptance report')
for name, digest in accepted['source_code_sha256'].items():
    if name in {'tools/build_release.py', 'tools/launch.py', 'policy/xr1_submission/deploy_policy.yml'}:
        continue
    if manifest.get(name) != digest:
        raise RuntimeError(f'Accepted source missing or changed: {name}')
audit = json.loads((root / 'verification/bell50_audit.json').read_text())
preflight = json.loads((root / 'verification/bell50_preflight.json').read_text())
if not audit['evaluation_integrity_passed'] or audit['episodes'] != 50:
    raise RuntimeError('Missing complete official-default 50-episode audit')
if hashlib.sha256((root / 'verification/bell50_metrics.json').read_bytes()).hexdigest() != audit['metrics_sha256']:
    raise RuntimeError('50-episode metrics were changed')
name = 'policy/xr1_submission/deploy_policy.yml'
published = yaml.safe_load((root / name).read_text())
if published['eval_reset_sync_steps'] != audit['evaluation']['eval_reset_sync_steps'] or published['eval_reset_sync_steps'] != 0:
    raise RuntimeError('Publication must use official reset default 0')
historical = (root / name).read_text().replace('eval_reset_sync_steps: 0', 'eval_reset_sync_steps: 10')
if hashlib.sha256(historical.encode()).hexdigest() != preflight['release_files_sha256'][name]:
    raise RuntimeError('Unexpected configuration change beyond the verified reset override')
mapping = json.loads((root / 'task_mapping.json').read_text())
expected_tasks = set('click_bell handle_basket water_pouring table_rearrangement items_handover drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading'.split())
if set(mapping['tasks']) != expected_tasks:
    raise RuntimeError('Task mapping must cover exactly the ten official tasks')
if (mapping['horizon'], mapping['execute_steps'], mapping['prefix_steps']) != (30, 29, 0):
    raise RuntimeError('Unexpected inference contract change')
for task, entry in mapping['tasks'].items():
    expected = {'item_assembly': 'item_assembly_160k_ft10k', 'water_pouring': 'water_pouring_160k_ft5k'}.get(task, 'shared_160k')
    if entry['artifact'] != expected:
        raise RuntimeError(f'Unexpected checkpoint mapping for {task}')
assembly = json.loads((root / 'verification/assembly_export.json').read_text())
assembly_acceptance = json.loads((root / 'verification/assembly_acceptance.json').read_text())
if not assembly['passed'] or not assembly['exported_tensors_exactly_equal_source']:
    raise RuntimeError('Assembly export has not passed exact tensor verification')
if not assembly_acceptance['passed'] or assembly_acceptance['model_sha256'] != assembly['model_sha256']:
    raise RuntimeError('Assembly checkpoint has not passed packaged runtime acceptance')
if assembly_acceptance['config_sha256'] != assembly['config_sha256']:
    raise RuntimeError('Assembly config differs from runtime acceptance')
if hashlib.sha256((root / 'verification/assembly_metrics.json').read_bytes()).hexdigest() != assembly_acceptance['metrics_sha256']:
    raise RuntimeError('Assembly acceptance metrics were changed')
for name, digest in assembly_acceptance['runtime_source_sha256'].items():
    if manifest.get(name) != digest:
        raise RuntimeError(f'Assembly accepted runtime changed: {name}')
water_pouring = json.loads((root / 'verification/water_pouring_export.json').read_text())
water_pouring_acceptance = json.loads((root / 'verification/water_pouring_acceptance.json').read_text())
if not water_pouring['passed'] or not water_pouring['exported_tensors_exactly_equal_source']:
    raise RuntimeError('Water export has not passed exact tensor verification')
if not water_pouring_acceptance['passed'] or water_pouring_acceptance['model_sha256'] != water_pouring['model_sha256']:
    raise RuntimeError('Water checkpoint has not passed packaged runtime acceptance')
if water_pouring_acceptance['config_sha256'] != water_pouring['config_sha256']:
    raise RuntimeError('Water config differs from runtime acceptance')
if hashlib.sha256((root / 'verification/water_pouring_metrics.json').read_bytes()).hexdigest() != water_pouring_acceptance['metrics_sha256']:
    raise RuntimeError('Water acceptance metrics were changed')
for name, digest in water_pouring_acceptance['runtime_source_sha256'].items():
    if manifest.get(name) != digest:
        raise RuntimeError(f'Water accepted runtime changed: {name}')
water50_root = root / 'verification/water50'
water50 = json.loads((water50_root / 'audit.json').read_text())
if not water50['passed'] or water50['episodes'] != 50 or water50['successes'] != 50:
    raise RuntimeError('Missing selected water checkpoint 50-episode evidence')
if water50['checkpoint_sha256'] != water_pouring['source_checkpoint_sha256']:
    raise RuntimeError('Water checkpoint differs from the selected 50/50 checkpoint')
for name, digest in water50['source_files_sha256'].items():
    if hashlib.sha256((water50_root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'Historical water evidence changed: {name}')
validation = {'passed': True, 'historical_release_zip_sha256': audit['release_zip_sha256'],
    'bell50_metrics_sha256': audit['metrics_sha256'],
    'policy_and_model_runtime_unchanged': True, 'shared_checkpoint_unchanged': True,
    'effective_config_equivalent': True, 'assembly_model_sha256': assembly['model_sha256'],
    'assembly_packaged_runtime_episodes': assembly_acceptance['episodes'],
    'water_model_sha256': water_pouring['model_sha256'],
    'water_packaged_runtime_episodes': water_pouring_acceptance['episodes'],
    'publication_changes': ['reset default 10 -> 0, identical to the audited 50-episode override',
                            'item_assembly alone uses the user-selected shared160k + 10k fine-tune',
                            'water_pouring alone uses the user-selected shared160k + 5k fine-tune',
                            'checkpoint-specific export and packaged-runtime acceptance',
                            'defer final native environment close until unchanged official evaluator writes metrics',
                            'publication documentation and this build-time validation']}
validation_bytes = (json.dumps(validation, indent=2) + '\n').encode()
manifest['PUBLICATION_VALIDATION.json'] = hashlib.sha256(validation_bytes).hexdigest()
with zipfile.ZipFile(dist / 'robosyn-xr1-submission.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in selected:
        archive.write(p, 'robosyn-xr1-submission/' + str(p.relative_to(root)))
    archive.writestr('robosyn-xr1-submission/PUBLICATION_VALIDATION.json', validation_bytes)
    archive.writestr('robosyn-xr1-submission/SOURCE_SHA256.json', json.dumps(manifest, indent=2) + '\n')
path = dist / 'robosyn-xr1-submission.zip'
(dist / 'robosyn-xr1-submission.zip.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n')
print(path)
