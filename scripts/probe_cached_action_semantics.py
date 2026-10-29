"""Use cached fixed-scene action velocities for a small paraphrase probe."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/data/users/jiachangzhang/eb_vjepa')
OUT = ROOT / 'experiments/instruction_shift_140k_20261026'
CASES = {
    'pick up the milk and place it in the basket',
    'put the black bowl in the bottom drawer of the cabinet and close it',
    'push the plate to the front of the stove',
    'put the cream cheese in the bowl',
    'pick up the black bowl on the wooden cabinet and place it on the plate',
}

def main():
    pairs = json.loads((OUT/'pairs.json').read_text())
    groups = {}
    for x in pairs:
        groups.setdefault(x['canonical'], []).append(x['paraphrase'])
    groups = {k:v for k,v in groups.items() if k in CASES and len(v) >= 2}
    obj = torch.load(OUT/'dream_features.pt', map_location='cpu', weights_only=True)
    velocities = obj['velocities']
    result = {'protocol': {'type':'cached_fixed_observation_action_velocity_paraphrase_probe', 'semantic_switch_probe':False, 'scenes':['scene40','scene80'], 'flow_times':[0.0,0.5]}, 'cases':{}}
    rel=[]; cos=[]
    for scene in ('scene40_time0.0','scene40_time0.5','scene80_time0.0','scene80_time0.5'):
        result['cases'][scene] = {}
        for canonical, paras in groups.items():
            x = velocities[scene][canonical].double().flatten()
            rows=[]
            for text in paras:
                y = velocities[scene][text].double().flatten()
                d = float((y-x).norm()/(x.norm()+1e-8))
                c = float((x@y)/((x.norm()+1e-8)*(y.norm()+1e-8)))
                rows.append({'text':text,'relative_l2_to_canonical':d,'cosine_to_canonical':c})
                rel.append(d); cos.append(c)
            result['cases'][scene][canonical] = rows
    result['summary'] = {'n_cases':len(groups),'n_comparisons':len(rel),'mean_relative_l2':float(np.mean(rel)), 'median_relative_l2':float(np.median(rel)), 'mean_cosine':float(np.mean(cos)), 'min_cosine':float(np.min(cos))}
    (OUT/'behavior_paraphrase_probe.json').write_text(json.dumps(result,indent=2))
    (OUT/'behavior_paraphrase_probe.md').write_text('\n'.join([
        '# Fixed-observation paraphrase action probe','',
        'This uses cached action velocity outputs with identical scene, proprio, noise, and flow state. It tests paraphrase stability only; it is not a semantic-switch or success-rate test.','',
        f"- Cases: {len(groups)}; comparisons: {len(rel)}",
        f"- Mean relative velocity change: {np.mean(rel):.4f}",
        f"- Median relative velocity change: {np.median(rel):.4f}",
        f"- Mean velocity cosine: {np.mean(cos):.4f}",
        f"- Minimum velocity cosine: {np.min(cos):.4f}",
        '', 'A real task-discrimination probe still needs matched scenes with two valid competing targets and closed-loop goal predicates.'
    ]))
    print(json.dumps(result['summary'], indent=2))

if __name__ == '__main__': main()
