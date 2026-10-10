"""Cloud-only export: model tensors, inference config, and offline processor assets."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor, Qwen3VLConfig


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--processor-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    model = a.output / 'shared_160k'
    model.mkdir()
    raw = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = raw.get('module', raw)
    if not state or not all(k.startswith('model.') and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Expected complete model tensor dictionary')
    torch.save(state, model / 'model.pt')
    # Retain exact inference fields, omit private training paths and credentials.
    config = yaml.safe_load(a.config.read_text())
    data = config['data']['params']['train_datasets']
    clean = {'data': {'params': {'train_datasets': {k: data[k] for k in
        ('mean', 'std', 'q01', 'q99', 'action_representation') if k in data}}},
        'model': {'params': {'model': config['model']['params']['model']}}}
    (model / 'config.yaml').write_text(yaml.safe_dump(clean, sort_keys=False))
    processor = AutoProcessor.from_pretrained(a.processor_source, local_files_only=True)
    processor.save_pretrained(a.output / 'processor')
    Qwen3VLConfig.from_pretrained(a.processor_source, local_files_only=True).save_pretrained(a.output / 'processor')
    manifest = {'schema': 1, 'artifact': 'xr1_sim10_160k_h30_k29', 'checkpoint_step': 160000,
        'source_checkpoint_sha256': sha(a.checkpoint), 'source_config_sha256': sha(a.config),
        'processor_revision': a.processor_source.name, 'files': {}}
    for f in sorted(a.output.rglob('*')):
        if f.is_file():
            manifest['files'][str(f.relative_to(a.output))] = {'bytes': f.stat().st_size, 'sha256': sha(f)}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'exported': True, 'tensor_count': len(state), 'files': len(manifest['files'])}))


if __name__ == '__main__':
    main()
