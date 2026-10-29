"""Wait for final checkpoint; paired current/delta slot-specific-content ablation.

No optimization. Calibration uses fixed monitor features. Evaluation uses 32
deterministic training samples, not a held-out control benchmark. Noise is never
averaged across slots; targets remain the original full representation targets.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from elaf.v0_train import build_backbone, model_config
from elaf.v0_model import ELAFV0Policy
from elaf.v0_dataset import CachedV0Dataset, collate_v0
from elaf.v0_training import seed_all, load_trainable_state, atomic_json


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
    p=args.run_dir;out=p/'slot_ablation';out.mkdir(exist_ok=True)
    while True:
        status=json.loads((p/'status.json').read_text())
        if status['status']=='completed':break
        if status['status'] in ['failed','paused']:raise RuntimeError('Trainer stopped before completion')
        time.sleep(15)
    config=json.loads((p/'resolved_config.json').read_text())['config']
    checkpoint=(p/'checkpoints/latest').resolve()
    seed_all(239);torch.set_num_threads(2)
    model=ELAFV0Policy(build_backbone(config),model_config(config)).cuda().eval()
    state=torch.load(checkpoint/'training.pt',map_location='cpu',weights_only=False)
    load_trainable_state(model,state['model']);del state
    identity=json.loads((p/'run_identity.json').read_text())
    cache=torch.load(identity['monitor_path'],map_location='cpu',weights_only=True)
    zsum=dsum=None;n=0
    for start in range(0,len(cache['current_features']),8):
        z=model._resample(cache['current_features'][start:start+8].cuda())
        f=model._resample(cache['future_features'][start:start+8].cuda());d=(f-z)/model.delta_scale
        a=(z-z.mean(1,keepdim=True)).sum(0);b=(d-d.mean(1,keepdim=True)).sum(0)
        zsum=a if zsum is None else zsum+a;dsum=b if dsum is None else dsum+b;n+=len(z)
    zb,db=zsum/n,dsum/n
    del cache
    dataset=CachedV0Dataset(config['manifest'],config['checkpoint'],image_aug=False)
    indices=[]
    for suite in sorted(dataset.suite_indices):
        members=dataset.suite_indices[suite]
        for j in range(8):
            index=members[j*len(members)//8]
            while not bool(dataset[index]['future_valid']):index+=1
            indices.append(index)
    records=[]
    for start in range(0,len(indices),4):
        batch=collate_v0([dataset[i] for i in indices[start:start+4]],pad_token_id=dataset.pad_token_id)
        batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        size=len(batch['actions']);noise_d=torch.randn(size,32,1024,device='cuda');noise_a=torch.randn(size,8,7,device='cuda',dtype=torch.bfloat16)
        for tau in [.25,.5,.75]:
            t=torch.full((size,),tau,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                base=model(batch,u=t,s=t,delta_noise=noise_d,action_noise=noise_a,return_debug=True)
                condition,mask=model._condition(batch)
            z,d=base['current_state'],base['delta_target']
            z_shared=z.mean(1,keepdim=True)+zb
            d_shared=d.mean(1,keepdim=True)+db
            av=batch['action_valid'][...,None];dv=batch['future_valid'][:,None,None]
            def mse(x,y,valid):
                valid=valid.expand_as(x)
                return (x.float()-y.float()).square()[valid].mean().item()
            for mode in ['full','current_shared','delta_shared','both_shared']:
                cz=z_shared if mode in ['current_shared','both_shared'] else z
                cd=d_shared if mode in ['delta_shared','both_shared'] else d
                du=torch.lerp(noise_d,cd,t[:,None,None])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    va,vd=model.velocity(condition,mask,cz,du,base['noisy_actions'],t,t,batch['action_valid'])
                records.append(dict(batch=start//4,tau=tau,mode=mode,sample_ids=batch['sample_id'],
                    action_mse=mse(va,base['action_velocity_target'],av),
                    delta_mse=mse(vd,base['delta_velocity_target'],dv),
                    action_output_change_rms=mse(va,base['action_velocity'],av)**.5,
                    delta_output_change_rms=mse(vd,base['delta_velocity'],dv)**.5))
        atomic_json(out/'results.json',dict(checkpoint=str(checkpoint),calibration_samples=n,evaluation_samples=len(indices),
            protocol='fixed training samples; u=s in .25,.5,.75; original targets, same Gaussian noise; no closed-loop claim',
            records=records))
        print('completed samples',start+size,flush=True)
    atomic_json(out/'status.json',{'status':'completed'})


if __name__=='__main__':main()
