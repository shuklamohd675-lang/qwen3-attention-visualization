"""Independently check published experiment artifacts without downloading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory: Path) -> dict:
    read = lambda name: json.loads((directory / name).read_text(encoding='utf-8'))
    manifest, evidence, summary = (read(name) for name in ('manifest.json', 'evidence.json', 'summary.json'))
    for document in (manifest, evidence, summary):
        require(document['status'] == 'passed', 'Experiment did not fully pass')
        require(all(document['gates'].values()), 'At least one acceptance gate failed')
    require(manifest['gates'] == evidence['gates'] == summary['gates'], 'Gate records disagree')
    require(evidence['model_revision'] == summary['model_revision'] == REVISION, 'Model revision mismatch')
    for filename, expected in manifest['artifacts'].items():
        require(Path(filename).name == filename, 'Unexpected manifest path')
        require(digest(directory / filename) == expected, f'Artifact hash mismatch: {filename}')
    sources = {
        'local_runner': ROOT / 'run-chapter2-attention.py',
        'upstream_runner': ROOT / 'chapter2/attention_visualization/run_attention_experiment.py',
        'upstream_visualization': ROOT / 'chapter2/attention_visualization/visualization.py',
    }
    for name, path in sources.items():
        require(digest(path) == evidence['source_hashes'][name], f'Source hash mismatch: {name}')
    expected_keys = {f'{section}_layer_{layer}' for section in ('simple', 'generated') for layer in (0, 13, 27)}
    max_error = 0.0
    with np.load(directory / 'attention_matrices.npz', allow_pickle=False) as arrays:
        require(set(arrays.files) == expected_keys, 'Unexpected matrix collection')
        for key in sorted(expected_keys):
            section, layer = key.split('_', 1)
            record = evidence[section]
            matrix = arrays[key]
            length = len(record['token_ids'])
            require(matrix.shape == (length, length), f'Shape mismatch: {key}')
            require(np.isfinite(matrix).all() and (matrix >= 0).all(), f'Invalid values: {key}')
            require(np.max(np.abs(np.triu(matrix, 1))) <= 1e-7, f'Causality check failed: {key}')
            error = float(np.max(np.abs(matrix.sum(axis=1) - 1)))
            max_error = max(max_error, error)
            require(error <= evidence['parameters']['row_sum_absolute_tolerance'], f'Row sums failed: {key}')
            expected_sink = record['metrics'][layer]['attention_sink_mean_excluding_first_row']
            require(np.isclose(matrix[1:, 0].mean(), expected_sink, rtol=1e-6, atol=1e-8), f'Statistic mismatch: {key}')
    generated = evidence['generated']
    require(generated['termination'] == 'eos', 'Generation was truncated')
    require(generated['tokens'][-1] == '<|im_end|>', 'Missing final EOS token')
    require(generated['context_length'] + generated['new_tokens'] == len(generated['token_ids']), 'Token counts disagree')
    thinking, answer = (generated['regions'][name] for name in ('thinking', 'answer'))
    require(bool(thinking and answer), 'Missing generated regions')
    require(set(thinking).isdisjoint(answer), 'Generated regions overlap')
    require(min(thinking) >= generated['context_length'] and max(answer) < len(generated['token_ids']), 'Region out of range')
    require(generated['tokens'][thinking[-1]] == '</think>', 'Thinking region did not close')
    require('391' in generated['final_answer'], 'Expected arithmetic result not present')
    return {'status': 'passed', 'artifact_hashes_checked': len(manifest['artifacts']),
            'source_hashes_checked': len(sources), 'matrices_checked': len(expected_keys),
            'max_row_sum_error': max_error, 'final_answer': generated['final_answer']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=ROOT / 'results/20261013T121659009239Z')
    args = parser.parse_args()
    print(json.dumps(verify(args.directory.resolve()), ensure_ascii=False, indent=2))
