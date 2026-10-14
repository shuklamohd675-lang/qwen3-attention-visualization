"""Download and validate the exact public Qwen model used by experiment 2-2."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT.parent / '.cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
FILES = ['config.json', 'generation_config.json', 'merges.txt', 'model.safetensors',
         'tokenizer.json', 'tokenizer_config.json', 'vocab.json']


def main() -> None:
    import requests
    from huggingface_hub import snapshot_download

    response = requests.get(
        f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true',
        timeout=60,
    )
    response.raise_for_status()
    metadata = response.json()
    if metadata['sha'] != REVISION:
        raise RuntimeError('Unexpected model revision')
    entries = {item['rfilename']: item for item in metadata['siblings']}
    cache = Path(os.environ['HF_HOME']) / 'hub'
    snapshot = cache / 'models--Qwen--Qwen3-0.6B' / 'snapshots' / REVISION
    complete = all((snapshot / name).is_file() and
                   (snapshot / name).stat().st_size == entries[name]['size'] for name in FILES)
    if not complete:
        snapshot = Path(snapshot_download(
            repo_id=MODEL, revision=REVISION, allow_patterns=FILES, cache_dir=cache,
        ))
    for name in FILES:
        path = snapshot / name
        entry = entries[name]
        if path.stat().st_size != entry['size']:
            raise RuntimeError(f'Invalid size: {name}')
        lfs_hash = (entry.get('lfs') or {}).get('sha256')
        with path.open('rb') as handle:
            if lfs_hash:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
                expected = lfs_hash
            else:
                digest = hashlib.sha1(f'blob {entry["size"]}\0'.encode('ascii'))
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                actual, expected = digest.hexdigest(), entry['blobId']
        if actual != expected:
            raise RuntimeError(f'Official file hash mismatch: {name}')
        print(f'Verified: {name}', flush=True)
    print(f'Model ready: {snapshot}', flush=True)


if __name__ == '__main__':
    main()
