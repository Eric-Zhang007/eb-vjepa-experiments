"""Read-only fixed-cache slot energy and query attention diagnostics."""
import argparse
import json
import time
from pathlib import Path
import torch
from elaf.v0_model import ELAFV0Config, SharedResampler
from elaf.v0_training import atomic_json


def unique_energy(z):
    x = z.double() - z.double().mean(0, keepdim=True)
    energy = x.square().sum()
    if energy <= 1e-20:
        return None
    return ((x-x.mean(1, keepdim=True)).square().sum()/energy).item()


def attention_metrics(a):
    """a [B,heads,queries,patches]; pairwise JSD in natural-log units."""
    def summarize(p):
        entropy = -(p*p.clamp_min(1e-30).log()).sum(-1)
        pairs = torch.triu_indices(p.shape[-2],p.shape[-2],1,device=p.device)
        values=[]
        for ids in pairs.split(32,dim=1):
            x,y=p[...,ids[0],:],p[...,ids[1],:]
            m=(x+y)/2
            js=(x*(x.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())+
                y*(y.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1)/2
            values.append(js)
        return entropy,torch.cat(values,dim=-1).mean(-1)
    entropy,js=summarize(a.mean(1))
    head_entropy,head_js=summarize(a)
    return entropy,js,head_entropy,head_js


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--steps',type=int,nargs='+',default=[0,1000,1500,2000]);args=ap.parse_args()
    torch.set_num_threads(2)
    identity=json.loads((args.run_dir/'run_identity.json').read_text())
    cache=torch.load(identity['monitor_path'],map_location='cpu',weights_only=True)
    out=args.run_dir/'slot_diagnostics';out.mkdir(exist_ok=True)
    import wandb
    run=wandb.init(project='eb_vjepa',entity='zjc010100000-xi-an-jiaotong-university-',
        name='slot-diagnostics-'+identity['parent_run_id'],group='elaf-v0-'+identity['parent_run_id'],
        job_type='slot-diagnostics',config={'parent_run_id':identity['parent_run_id'],'steps':args.steps,
        'monitor_manifest_sha256':identity['monitor_manifest_sha256'],'attention_jsd_units':'nats'})
    atomic_json(out/'run_identity.json',{'wandb_run_id':run.id,'parent_run_id':identity['parent_run_id']})
    try:
        for step in args.steps:
            snap=Path(identity['snapshot_dir'])/f'resampler-step-{step:09d}.pt'
            while not snap.exists():
                status=json.loads((args.run_dir/'status.json').read_text())
                if status['status'] in ['completed','paused','failed']:
                    raise RuntimeError(f'Trainer stopped without requested snapshot {step}')
                time.sleep(15)
            payload=torch.load(snap,map_location='cpu',weights_only=True)
            assert payload['parent_run_id']==identity['parent_run_id']
            model=SharedResampler(ELAFV0Config(**payload['config'])).cuda().eval()
            model.load_state_dict(payload['state_dict'])
            representations=[];metrics={};maps={}
            with torch.no_grad():
                for label,key in [('current','current_features'),('future','future_features')]:
                    zs=[];es=[];js=[];hes=[];hjs=[];meanmap=None
                    for start in range(0,len(cache[key]),8):
                        features=cache[key][start:start+8].cuda().float()
                        h=model.input_norm(features);q=model.queries.unsqueeze(0).expand(len(h),-1,-1)
                        attended,a=model.attention(q,h,h,need_weights=True,average_attn_weights=False)
                        z=model.output_projection(model.output_norm(attended+q))
                        zs.append(z.cpu());e,j,he,hj=attention_metrics(a)
                        es.append(e.cpu());js.append(j.cpu());hes.append(he.cpu());hjs.append(hj.cpu())
                        contribution=a.mean(1).sum(0).cpu()
                        meanmap=contribution if meanmap is None else meanmap+contribution
                    z=torch.cat(zs);representations.append(z)
                    entropy=torch.cat(es);head_entropy=torch.cat(hes)
                    prefix='monitor/attention/'+label
                    metrics[prefix+'/entropy_nats']=entropy.mean().item()
                    metrics[prefix+'/entropy_normalized']=(entropy.mean()/torch.log(torch.tensor(cache[key].shape[1]))).item()
                    metrics[prefix+'/pairwise_jsd_nats']=torch.cat(js).mean().item()
                    metrics[prefix+'/per_head_pairwise_jsd_nats']=torch.cat(hjs).mean().item()
                    maps[label]={'query_entropy':entropy.mean(0),'head_query_entropy':head_entropy.mean(0),
                                 'mean_attention':meanmap/len(z)}
            metrics['monitor/z/unique_slot_energy']=unique_energy(representations[0])
            metrics['monitor/delta/unique_slot_energy']=unique_energy(representations[1]-representations[0])
            report={'training_step':step,'samples':len(representations[0]),'metrics':metrics,
                    'parent_run_id':identity['parent_run_id'],'snapshot':str(snap),
                    'definition':'sample-centered energy; JSD over query pairs per sample, then averaged; FP32 attention'}
            torch.save(maps,out/f'attention-{step:06d}.pt')
            atomic_json(out/f'metrics-{step:06d}.json',report)
            run.log({'training_step':step,**metrics},step=step)
            print(json.dumps(report),flush=True)
            del model,representations
        atomic_json(out/'status.json',{'status':'completed','steps':args.steps})
    finally:run.finish()


if __name__=='__main__':main()
