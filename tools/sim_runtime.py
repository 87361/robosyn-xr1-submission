"""Isolate native caches and shorten paths without changing task/evaluator semantics."""
import hashlib
from pathlib import Path
import shutil
import tempfile


def prepare(challenge, embodi, data):
    root = Path(tempfile.mkdtemp(prefix='xrs_', dir='/tmp'))
    ignore = shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'eval_result', 'eval_videos')
    shutil.copytree(challenge, root / 'c', symlinks=True, ignore=ignore)
    shutil.copytree(embodi, root / 'e', symlinks=True, ignore=ignore)
    target_data = root / 'd'
    (target_data / 'assembled').mkdir(parents=True)
    for source in data.iterdir():
        if source.name != 'assembled':
            (target_data / source.name).symlink_to(source, target_is_directory=source.is_dir())
    if (data / 'extract').is_dir():
        for source in (data / 'extract').iterdir():
            dest = target_data / source.name
            if not dest.exists() and not dest.is_symlink():
                dest.symlink_to(source, target_is_directory=source.is_dir())
    # Equivalent to the established cloud runtime's single cache-location patch.
    path = root / 'e/embodichain/lab/sim/sim_manager.py'
    before = path.read_bytes()
    text = before.decode()
    old = 'SIM_CACHE_DIR = Path.home() / ".cache" / "embodichain_cache"'
    if text.count(old) != 1:
        raise RuntimeError('EmbodiChain cache API differs; review compatibility before running')
    text = text.replace(old, 'SIM_CACHE_DIR = Path(os.environ["XR1_SIM_CACHE_DIR"])')
    compile(text, str(path), 'exec')
    path.write_text(text)
    cache = root / 'k'
    cache.mkdir()
    env = {'EMBODICHAIN_ROOT': str(root / 'e'), 'EMBODICHAIN_DATA_ROOT': str(target_data),
           'XR1_SIM_CACHE_DIR': str(cache / 'sim'), 'XDG_CACHE_HOME': str(cache)}
    record = {'runtime_root': str(root), 'source_challenge': str(challenge),
        'source_embodichain': str(embodi), 'source_data': str(data),
        'patch': 'isolated simulator cache path only; no HOME override',
        'sim_manager_original_sha256': hashlib.sha256(before).hexdigest(),
        'sim_manager_runtime_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return root, env, record
