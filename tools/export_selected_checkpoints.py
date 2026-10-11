"""Cloud-only: add the selected basket/handover checkpoints to an immutable bundle."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
import yaml


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('base-models', 'evidence', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = json.loads((args.base_models / 'manifest.json').read_text())
    shared_cfg = yaml.safe_load((args.base_models / 'shared_160k/config.yaml').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for name, record in base['files'].items():
        source = args.base_models / name
        assert source.resolve().is_relative_to(args.base_models.resolve())
        assert source.stat().st_size == record['bytes'] and sha(source) == record['sha256'], name
        dest = args.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.pt':
            os.link(source, dest)
        else:
            shutil.copy2(source, dest)
    manifest = {**base, 'artifact': 'xr1_sim10_selected_h30_basket140k_handover180k',
                'files': dict(base['files']), 'artifacts': dict(base['artifacts'])}
    baseline = torch.load(args.base_models / 'shared_160k/model.pt', map_location='cpu',
                          mmap=True, weights_only=True)
    for task, step, k in (('handle_basket', 140000, 29), ('items_handover', 180000, 20)):
        evidence_root = args.evidence / task
        contract = json.loads((evidence_root / 'contract.json').read_text())
        audit = json.loads((evidence_root / 'checkpoint_audit.json').read_text())
        asset = contract['asset']
        assert (contract['task'], asset['step'], asset['horizon'], contract['k'], contract['p']) == (task, step, 30, k, 0)
        source = Path(asset['source_checkpoint'])
        config_path = Path(asset['source_config'])
        source_sha = sha(source)
        assert asset['passed'] and audit['passed']
        assert source_sha == asset['sha256'] == audit['sha256_from_verified_transfer']
        assert source.stat().st_size == asset['bytes'] == audit['bytes']
        assert sha(config_path) == asset['config_sha256']
        cfg = yaml.safe_load(config_path.read_text())
        data = cfg['data']['params']['train_datasets']
        model = cfg['model']['params']['model']
        assert data['action_representation'] == 'joint_absolute'
        assert model.get('action_length', 30) == 30 and data['action_length'] == 30
        for key in ('mean', 'std', 'q01', 'q99', 'action_representation'):
            assert data[key] == shared_cfg['data']['params']['train_datasets'][key], key
        assert model == shared_cfg['model']['params']['model']
        raw = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
        state = raw.get('module', raw)
        assert set(state) == set(baseline)
        for name, tensor in state.items():
            assert name.startswith('model.') and isinstance(tensor, torch.Tensor)
            assert tensor.shape == baseline[name].shape and tensor.dtype == baseline[name].dtype, name
            assert torch.isfinite(tensor).all(), name
        artifact = f'{task}_{step // 1000}k'
        target = args.output / artifact
        target.mkdir()
        torch.save(state, target / 'model.pt')
        exported = torch.load(target / 'model.pt', map_location='cpu', mmap=True, weights_only=True)
        assert set(exported) == set(state)
        for name in state:
            assert torch.equal(state[name], exported[name]), name
        clean = {'data': {'params': {'train_datasets': {key: data[key] for key in
                 ('mean', 'std', 'q01', 'q99', 'action_representation')}}},
                 'model': {'params': {'model': model}}}
        (target / 'config.yaml').write_text(yaml.safe_dump(clean, sort_keys=False))
        report = {'passed': True, 'task': task, 'artifact': artifact, 'checkpoint_step': step,
                  'source_run': config_path.parent.name, 'source_checkpoint_sha256': source_sha,
                  'source_config_sha256': sha(config_path), 'selection_contract_sha256': sha(evidence_root / 'contract.json'),
                  'tensor_count': len(state), 'all_tensors_finite': True,
                  'tensor_keys_shapes_dtypes_match_shared': True, 'exported_tensors_exactly_equal_source': True,
                  'normalization_equal_to_shared': True, 'horizon': 30, 'execute_steps': k, 'prefix_steps': 0,
                  'training_data': 'official ten-task Sim release, 1000 episodes per task',
                  'model_sha256': sha(target / 'model.pt'), 'config_sha256': sha(target / 'config.yaml')}
        manifest['artifacts'][artifact] = report
        for f in sorted(target.iterdir()):
            manifest['files'][str(f.relative_to(args.output))] = {'bytes': f.stat().st_size, 'sha256': sha(f)}
        (args.output.parent / f'{task}_export.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        del raw, state, exported, tensor
        gc.collect()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
