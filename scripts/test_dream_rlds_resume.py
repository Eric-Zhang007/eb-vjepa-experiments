"""Real RLDS iterator checkpoint equivalence in separate processes, including augmentation."""
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = Path('/data/users/jiachangzhang/eb_vjepa')
sys.path.insert(0, str(ROOT / 'repos/Dream-VLX/vla'))

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['save', 'restore', 'rank'])
parser.add_argument('--buffer', type=int, default=100000)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--dataset', choices=['libero_mix4'], required=True)
args = parser.parse_args()
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
assert NUM_ACTIONS_CHUNK == 8 and PROPRIO_DIM == 8
assert ACTION_PROPRIO_NORMALIZATION_TYPE.value == 'bounds_q99'
from prismatic.vla.datasets.datasets import RLDSDataset
args.output.mkdir(parents=True, exist_ok=True)
rank = 1 if args.mode == 'rank' else 0
data = RLDSDataset(
    Path('/data/users/jiachangzhang/DNFT4RP/upstream/vla_adapter/datasets/rlds'),
    'libero_mix4', lambda x: x, (224, 224), shuffle_buffer_size=args.buffer,
    image_aug=True, seed=239, rank=rank,
)
assert list(data.dataset.sample_weights) == [0.25] * 4
def digest(sample):
    # Hash values by field, independent of dictionary order and pickle's object
    # identity/alias memoization across independently started Python processes.
    if isinstance(sample, dict):
        return {key: digest(sample[key]) for key in sorted(sample)}
    value = np.asarray(sample)
    payload = json.dumps(value.tolist(), default=lambda x: x.decode('utf-8'), sort_keys=True).encode() if value.dtype.kind in 'OSU' else value.tobytes()
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + payload).hexdigest()
start = time.monotonic()
if args.mode == 'save':
    iterator = iter(data)
    first_sample = next(iterator)
    assert first_sample['action'].shape[-2:] == (8, 7), first_sample['action'].shape
    first = digest(first_sample)
    ready = time.monotonic()
    data.save_iterator_state(args.output / 'iterator')
    saved = time.monotonic()
    expected = [digest(next(iterator)) for _ in range(128)]
    result = dict(first=first, expected=expected, buffer=args.buffer, action_chunk=NUM_ACTIONS_CHUNK, normalization=ACTION_PROPRIO_NORMALIZATION_TYPE.value,
                  startup_seconds=ready-start, save_seconds=saved-ready,
                  checkpoint_bytes=sum(p.stat().st_size for p in args.output.glob('iterator.*')))
    (args.output / 'expected.json').write_text(json.dumps(result, indent=2))
else:
    result_file = args.output / 'result.json'
    result = json.loads((result_file if result_file.exists() else args.output / 'expected.json').read_text())
    assert result['buffer'] == args.buffer
    if args.mode == 'restore':
        data.restore_iterator_state(args.output / 'iterator')
        iterator = iter(data)
        actual = [digest(next(iterator)) for _ in range(128)]
        assert actual == result['expected'], (actual, result['expected'])
        result['restored_128_samples_exact'] = True
    else:
        assert digest(next(iter(data))) != result['first']
        result['rank_distinct'] = True
    result[args.mode + '_seconds'] = time.monotonic() - start
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
