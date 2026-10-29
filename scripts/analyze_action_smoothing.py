"""Validate and summarize the paired smoothing pilot without treating selection as a benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np


def key(x):
    return x['suite'], x['task_id'], x['trial']


def diagnostics(path):
    z = np.load(path)
    a, pos = z['executed'], z['eef_pos']
    result = {}
    for name, values in [('translation', a[:, :3]), ('rotation', a[:, 3:6]), ('eef_position', pos)]:
        d = np.linalg.norm(np.diff(values, axis=0), axis=1)
        boundary = np.arange(1, len(values)) % 10 == 0
        result[name] = dict(mean_delta=float(d.mean()), p95_delta=float(np.percentile(d, 95)),
                            boundary_delta=float(d[boundary].mean()) if boundary.any() else None,
                            within_chunk_delta=float(d[~boundary].mean()),
                            second_difference=float(np.linalg.norm(np.diff(values,n=2,axis=0),axis=1).mean()))
    result['eef_path_length'] = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    result['gripper_switches'] = int(np.count_nonzero(np.diff(a[:,6])))
    assert np.array_equal(z['raw'][:,6],a[:,6])
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); args=p.parse_args()
    out=args.output
    manifest=json.loads((out/'manifest.json').read_text())
    reports=[json.loads(p.read_text()) for p in sorted(out.glob('shard_*/report.json'))]
    assert len(reports)==8 and all(r['phase']=='complete' for r in reports)
    hashes={r['checkpoint_sha256'] for r in reports}
    assert len(hashes)==1 and len({r['manifest_sha256'] for r in reports})==1
    items=[]
    for path in sorted(out.glob('shard_*/report.json')):
        for x in json.loads(path.read_text())['items']:
            x['diagnostics']=diagnostics(path.parent/x['trace'])
            x['trace']=str(path.parent/x['trace']); x['contact_sheet']=str(path.parent/x['contact_sheet'])
            items.append(x)
    assert len(items)==len(manifest['entries'])*3
    assert len({(key(x),x['alpha']) for x in items})==len(items)
    base={key(x):x for x in items if x['alpha']==1}
    assert set(base)=={key(x) for x in manifest['entries']}
    comparisons=[]
    for alpha in (1.,.8,.5):
        selected=[x for x in items if x['alpha']==alpha]
        wins=[list(key(x)) for x in selected if x['success'] and not base[key(x)]['success']]
        losses=[list(key(x)) for x in selected if not x['success'] and base[key(x)]['success']]
        comparisons.append(dict(alpha=alpha, successes=sum(x['success'] for x in selected),episodes=len(selected),
                                rescued=wins, broken=losses,
                                mean_action_delta=float(np.mean([x['executed_metrics']['mean_action_delta'] for x in selected])),
                                mean_second_difference=float(np.mean([x['executed_metrics']['mean_second_difference'] for x in selected]))))
    result=dict(selection=manifest['selection'], checkpoint_sha256=hashes.pop(),
                baseline_history_mismatches=[list(key(x)) for x in base.values() if x['success']!=x['historical_success']],
                comparisons=comparisons,items=items,
                limitations=['Outcome-selected diagnostic subset is not an unbiased success-rate estimate.',
                             'Different outcomes have different trajectory lengths; full-rollout motion metrics are descriptive, not causal.',
                             'Same initial states and RNG seeds; intervention changes subsequent observations and trajectories.',
                             'Clean only; no Plus smoothing evaluation in this pilot.'])
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='items'},indent=2))


if __name__=='__main__': main()
