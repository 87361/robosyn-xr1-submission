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
                  and (p.suffix in {'.py', '.sh', '.yml', '.json', '.md', '.txt'}
                       or p.name in {'LICENSE', '.gitignore'}))
dist = root / 'dist'
dist.mkdir(exist_ok=True)
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
accepted = json.loads((root / 'verification/acceptance.json').read_text())
if not accepted['passed']:
    raise RuntimeError('No successful end-to-end acceptance report')
for name, digest in accepted['source_code_sha256'].items():
    if name in {'tools/build_release.py', 'policy/xr1_submission/deploy_policy.yml'}:
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
validation = {'passed': True, 'historical_release_zip_sha256': audit['release_zip_sha256'],
    'bell50_metrics_sha256': audit['metrics_sha256'],
    'runtime_and_model_unchanged': True, 'effective_config_equivalent': True,
    'publication_changes': ['reset default 10 -> 0, identical to the audited 50-episode override',
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
