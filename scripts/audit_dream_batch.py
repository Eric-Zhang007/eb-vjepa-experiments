import json,os,hashlib
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from elaf.evaluation.dream_jepa import DreamJEPAPolicy
r=Path('/data/users/jiachangzhang/eb_vjepa')
o=r/'experiments/dream_jepa_batch_audit_20261025';o.mkdir(exist_ok=True)
c=json.loads((r/'experiments/dream_jepa_eval_100k_20261025/policy.json').read_text());c.update(language_subgoals=False,device='cuda:0')
p=DreamJEPAPolicy(c)
obs=[]
for f in sorted((r/'experiments/dream_jepa_failures_100k_20261025/videos').glob('*.jsonl')):
    trace=[json.loads(l) for l in f.read_text().splitlines()];frames=imageio.mimread(f.with_suffix('.mp4'),memtest=False)
    idx=min(40,len(frames)-1);spec=trace[0]['spec']
    history=[dict(image=np.ascontiguousarray(frames[k][:,:256]),wrist_image=np.ascontiguousarray(frames[k][:,256:])) for k in range(idx-3,idx+1)]
    obs.append(dict(**history[-1],history=history,proprio=np.asarray(trace[idx]['proprio'],dtype=np.float32),instruction=trace[0]['instruction'],suite=spec['suite'],noise_seed=spec['seed'],episode_key=f.stem))
def metric(a,b):
    d=np.abs(a-b)
    return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),rms=float(np.sqrt(np.mean((a-b)**2))),gripper_sign_flips=int(np.sum((a[:,:,-1]>=.5)!=(b[:,:,-1]>=.5))))
ref=np.concatenate([p.predict([x]) for x in obs]);results={}
results['repeat_single']=metric(ref[:1],p.predict(obs[:1]))
for n in [2,4,8]:
    output=np.concatenate([p.predict(obs[k:k+n]) for k in range(0,len(obs),n)])
    results[f'batch{n}']=metric(ref,output)
    np.save(o/f'batch{n}.npy',output)
rev=p.predict(list(reversed(obs)))[::-1];results['reverse_batch8_vs_single']=metric(ref,rev);results['reverse_batch8_vs_batch8']=metric(np.load(o/'batch8.npy'),rev)
dup=p.predict([obs[0]]*8);results['duplicate_batch8_vs_single']=metric(np.repeat(ref[:1],8,0),dup);results['duplicate_within_batch']=metric(np.repeat(dup[:1],8,0),dup)
np.save(o/'single.npy',ref)
(o/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2),flush=True)
