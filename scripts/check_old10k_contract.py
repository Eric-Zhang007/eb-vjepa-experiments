import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[1]
out = root / 'experiments/action_only/old10k_extend40k'
saved = json.loads((out / 'source_trainer_state.json').read_text())['resume_contract']
cfg = OmegaConf.load(out / 'train/config.yaml')
def diff(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            diff(a.get(key), b.get(key), prefix + '.' + key)
    elif a != b:
        print(prefix, repr(a), '=>', repr(b))
for key, value in saved.items():
    if key in ('model', 'data'):
        current = cfg.model if key == 'model' else cfg.data.train
        diff(value, OmegaConf.to_container(current, resolve=True), key)
    elif key in cfg:
        diff(value, cfg[key], key)
print('dataset stats:', saved['dataset_stats_sha256'], hashlib.sha256((out / 'train/dataset_stats.json').read_bytes()).hexdigest())
original = root / 'experiments/action_only/ctp_full_010000_handoff/train_action_only_010000/dataset_stats.json'
diff(json.loads(original.read_text()), json.loads((out / 'train/dataset_stats.json').read_text()), 'stats')
print('original stats hash:', hashlib.sha256(original.read_bytes()).hexdigest())
