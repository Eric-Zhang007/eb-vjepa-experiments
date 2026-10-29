"""Read-only E-LAF v2 fixed-set diagnostics; isolated from the optimizer process."""
import argparse
import dataclasses
import json
import math
import re
import time
from pathlib import Path

import torch

from elaf.diagnostics import representation_diagnostics
from elaf.slot_diagnostics import attention_metrics, unique_energy
from elaf.v0_training import atomic_json
from elaf.monitor import monitor_manifest_sha256


def v2_model_config(config):
    from elaf.v0_train import model_config
    from elaf.v2_model import ELAFV2Config
    values = dataclasses.asdict(model_config(config))
    spatial = config.get('spatial_resampler', config.get('resampler', {}))
    values.update(grid_shape=tuple(spatial.get('grid_shape', config.get('grid_shape', (2,24,24)))),
                  residual_alpha=spatial.get('residual_alpha', config.get('residual_alpha', .1)),
                  resampler_fp32=True)
    return ELAFV2Config(**values)


def _attention_forward(model, features):
    if hasattr(model, 'forward_with_attention'):
        return model.forward_with_attention(features)
    h = model.input_norm(features)
    q = model.queries.unsqueeze(0).expand(len(h), -1, -1)
    r, a = model.attention(q, h, h, need_weights=True, average_attn_weights=False)
    return model.output_projection(model.output_norm(r + q)), a


@torch.no_grad()
def diagnose_resampler(model, cache, *, initial_delta_rms=None, batch_size=8):
    """Exact historical metric definitions on the same ordered fixed cache."""
    model.float().eval()
    device = next(model.parameters()).device
    states, metrics, maps = [], {}, {}
    for label, key in [('current', 'current_features'), ('future', 'future_features')]:
        zs, entropies, jsds, head_entropies, head_jsds = [], [], [], [], []
        meanmap = None
        for start in range(0, len(cache[key]), batch_size):
            h = cache[key][start:start + batch_size].to(device=device, dtype=torch.float32)
            z, a = _attention_forward(model, h)
            e, j, he, hj = attention_metrics(a)
            zs.append(z.detach())
            entropies.append(e); jsds.append(j); head_entropies.append(he); head_jsds.append(hj)
            contribution = a.mean(1).sum(0)
            meanmap = contribution if meanmap is None else meanmap + contribution
        z = torch.cat(zs); states.append(z)
        prefix = 'monitor/attention/' + label
        metrics[prefix + '/entropy_nats'] = torch.cat(entropies).mean().item()
        metrics[prefix + '/entropy_normalized'] = metrics[prefix + '/entropy_nats'] / math.log(cache[key].shape[1])
        metrics[prefix + '/pairwise_jsd_nats'] = torch.cat(jsds).mean().item()
        metrics[prefix + '/per_head_pairwise_jsd_nats'] = torch.cat(head_jsds).mean().item()
        maps[label] = {'mean_attention': (meanmap/len(z)).cpu(),
                       'head_query_entropy': torch.cat(head_entropies).mean(0).cpu()}
    # The spatial skip path is full-rank, so the former random-resampler cap of
    # 256 is no longer applicable. Historical raw/effective rank stays identical.
    result = representation_diagnostics(*states, initial_delta_rms=initial_delta_rms,
                                        representation_rank_cap=None)
    result['metrics'].update(metrics)
    for label, z in [('z', states[0]), ('delta', states[1]-states[0])]:
        result['metrics']['monitor/'+label+'/unique_slot_energy'] = unique_energy(z)
        result['metrics']['monitor/'+label+'/rms'] = z.square().mean().sqrt().item()
    result['attention_maps'] = maps
    return result


def _stats(x):
    x = x.detach().float()
    return {'rms': x.square().mean().sqrt().item(), 'max_abs': x.abs().max().item()}


def diagnose_policy_batch(policy, batch, *, seed=7331, times=None):
    """No optimizer steps; restore flags/modes/RNG. Use a separate model worker.

    Same noise and s=.5 across ten u-bin midpoints. Per-loss unscaled LoRA
    gradients are measured at u=.5 in one additional forward/backward.
    """
    times = times or [i/10+.05 for i in range(10)]
    device = next(policy.parameters()).device
    modes = {m: m.training for m in policy.modules()}
    lora = [(n, p) for n, p in policy.named_parameters() if 'lora_' in n]
    flags = [p.requires_grad for _, p in lora]
    handles = []
    record = {'active': False, 'adapters': {}, 'activations': {}}
    def layer(name):
        match = re.search(r'\.layers\.(\d+)(?:\.|$)', name)
        return int(match.group(1)) if match else -1
    layers = sorted({layer(n) for n, _ in lora if layer(n) >= 0})
    chosen = set(layers[:1] + layers[len(layers)//2:len(layers)//2+1] + layers[-4:])
    def adapter_hook(name):
        def hook(module, args, output):
            if not record['active']: return
            with torch.no_grad():
                x = args[0].detach(); base = module.base_layer(x)
                delta = torch.zeros_like(base, dtype=torch.float32)
                for adapter in module.active_adapters:
                    if adapter in module.lora_A:
                        z = x.to(module.lora_A[adapter].weight.dtype)
                        delta += (module.lora_B[adapter](module.lora_A[adapter](z))*module.scaling[adapter]).float()
                b, d = _stats(base), _stats(delta)
                record['adapters'][name] = {'base': b, 'lora': d, 'ratio': d['rms']/max(b['rms'],1e-30),
                                             'absolute_output': _stats(output)}
        return hook
    def block_hook(name):
        def hook(module, args, output):
            if not record['active']: return
            h = output[0] if isinstance(output, tuple) else output
            record['activations'][name] = {k: _stats(v) for k,v in
                [('all',h),('current',h[:,-72:-40]),('delta',h[:,-40:-8]),('action',h[:,-8:])]}
        return hook
    for name, module in policy.named_modules():
        if layer(name) not in chosen: continue
        if hasattr(module,'lora_A') and name.endswith(('q_proj','k_proj')):
            handles.append(module.register_forward_hook(adapter_hook(name)))
        if re.search(r'\.layers\.\d+$',name): handles.append(module.register_forward_hook(block_hook(name)))
    def masked_stats(pred,target,valid):
        mask = valid[...,None].expand_as(pred)
        p,t = pred.float()[mask],target.float()[mask]
        if not p.numel(): return None
        return {'mse':(p-t).square().mean().item(),'prediction_rms':p.square().mean().sqrt().item(),
                'target_rms':t.square().mean().sqrt().item(),'zero_mse':t.square().mean().item(),
                'cosine':(torch.dot(p,t)/(p.norm()*t.norm()).clamp_min(1e-30)).item()}
    try:
        policy.eval()
        for _,p in lora: p.requires_grad_(True)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            b = len(batch['actions']); c = policy.config
            nd = torch.randn((b,c.latent_tokens,c.latent_dim),device=device)
            na = torch.randn((b,c.action_horizon,c.action_dim),device=device,dtype=torch.bfloat16)
            def forward(t):
                u=torch.full((b,),t,device=device); s=torch.full((b,),.5,device=device)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    return policy.loss(batch,u=u,s=s,delta_noise=nd,action_noise=na,return_debug=True)
            flows=[]
            with torch.no_grad():
                for t in times:
                    output=forward(t)
                    action=masked_stats(output['action_velocity'],output['action_velocity_target'],batch['action_valid'])
                    delta=masked_stats(output['delta_velocity'],output['delta_velocity_target'],batch['future_valid'][:,None])
                    analytic=masked_stats(-output['noisy_delta'].float()/(1-t),output['delta_velocity_target'],batch['future_valid'][:,None])
                    flows.append({'u':t,'s':.5,'action':action,'delta':delta,'delta_analytic':analytic})
            record['active']=True; output=forward(.5); record['active']=False
            params=[p for _,p in lora]
            ga=torch.autograd.grad(output['action_loss'],params,retain_graph=True,allow_unused=True)
            ga=[g.detach().float().cpu() if g is not None else None for g in ga]
            gd=torch.autograd.grad(output['delta_loss'],params,allow_unused=True)
            sums={}
            for (name,_),a,d in zip(lora,ga,gd):
                row=sums.setdefault(layer(name),[0.,0.,0.])
                if a is not None: row[0]+=a.double().square().sum().item()
                if d is not None:
                    d=d.detach().double().cpu();row[1]+=d.square().sum().item()
                    if a is not None:row[2]+=(a.double()*d).sum().item()
            def summary(v):
                a,d,dot=v
                return {'action_grad_norm':a**.5,'delta_grad_norm':d**.5,'cosine':dot/(a*d)**.5 if a*d else None}
            return {'seed':seed,'sample_ids':batch.get('sample_id'),'flow_bins':flows,
                    'lora_gradients':{'global':summary([sum(v[i] for v in sums.values()) for i in range(3)]),
                                      'layers':{str(k):summary(v) for k,v in sums.items()}},
                    'adapters':record['adapters'],'activations':record['activations'],
                    'attention_logits':'not recorded: native FlashAttention does not materialize logits; output ratios alone do not prove attention explosion'}
    finally:
        for h in handles:h.remove()
        for (_,p),flag in zip(lora,flags):p.requires_grad_(flag)
        for module,mode in modes.items():module.training=mode


def full_checkpoint_diagnostic(config, checkpoint, device):
    from elaf.v0_train import build_backbone
    from elaf.v0_dataset import CachedV0Dataset,collate_v0
    from elaf.v0_training import load_trainable_state
    from elaf.v2_model import ELAFV2Policy
    policy=ELAFV2Policy(build_backbone(config),v2_model_config(config)).to(device).eval()
    state=torch.load(Path(checkpoint)/'training.pt',map_location='cpu',weights_only=False)
    saved=state['model']
    # A frozen Stage-2 LoRA is absent from optimizer groups but remains part of
    # the model state. Refuse to silently diagnose base-initialized adapters.
    required={n for n,_ in policy.named_parameters()
              if 'lora_' in n or n.split('.',1)[0] in
              {'resampler','current_in','delta_in','action_in','action_out','delta_out',
               'current_position','delta_position','action_position','delta_time','action_time','proprio_in'}}
    missing=required-set(saved)
    if missing: raise ValueError(f'Checkpoint omits learned v2/LoRA tensors: {sorted(missing)[:8]}')
    for name in required:
        if saved[name].shape != dict(policy.named_parameters())[name].shape:
            raise ValueError(f'Checkpoint tensor shape mismatch: {name}')
    policy.load_state_dict(saved,strict=False);del state,saved,required
    dataset=CachedV0Dataset(config['manifest'],config['checkpoint'],image_aug=False)
    indices=[next(i for i in dataset.suite_indices[suite] if bool(dataset[i]['future_valid'])) for suite in sorted(dataset.suite_indices)]
    records=[]
    for index in indices:
        batch=collate_v0([dataset[index]],pad_token_id=dataset.pad_token_id)
        batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        records.append(diagnose_policy_batch(policy,batch,seed=81000+index))
    del policy,dataset
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return {'checkpoint':str(checkpoint),'records':records,'protocol':'four fixed training samples, one per suite; no augmentation; same noise; s=.5; u bin midpoints; per-loss gradients at u=.5; no updates'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--watch',action='store_true')
    parser.add_argument('--device',default='cuda');parser.add_argument('--poll-seconds',type=float,default=15)
    parser.add_argument('--no-wandb',action='store_true');args=parser.parse_args()
    import fcntl
    from elaf.v2_model import ELAFV2Config,SpatialAnchoredResampler
    torch.set_num_threads(4)
    out=args.run_dir/'v2_diagnostics';out.mkdir(exist_ok=True,parents=True)
    lock=(out/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (args.run_dir/'run_identity.json').exists():time.sleep(args.poll_seconds)
    identity=json.loads((args.run_dir/'run_identity.json').read_text())
    config=json.loads((args.run_dir/'resolved_config.json').read_text())['config']
    cache=torch.load(identity['monitor_path'],map_location='cpu',weights_only=True)
    cache_sha=monitor_manifest_sha256(cache)
    if identity.get('monitor_manifest_sha256') != cache_sha:
        raise ValueError('Run identity does not match the fixed monitor cache')
    run=None
    if not args.no_wandb:
        import wandb
        saved=out/'wandb_run.json';old=json.loads(saved.read_text()) if saved.exists() else {}
        run=wandb.init(project=config.get('wandb_project','eb_vjepa'),entity=config.get('wandb_entity'),
            id=old.get('id'),resume='must' if old else None,name='elaf-v2-diagnostics-'+identity['parent_run_id'],
            job_type='v2-diagnostics',config={'parent_run_id':identity['parent_run_id']})
        run.define_metric('training_step');run.define_metric('monitor/*',step_metric='training_step')
        atomic_json(saved,{'id':run.id})
    try:
        while True:
            for snap in sorted(Path(identity['snapshot_dir']).glob('resampler-step-*.pt')):
                step=int(snap.stem.rsplit('-',1)[-1]);dest=out/f'representation-{step:09d}.json'
                if dest.exists():continue
                payload=torch.load(snap,map_location='cpu',weights_only=True)
                if payload['parent_run_id']!=identity['parent_run_id']:raise ValueError('snapshot parent mismatch')
                if payload.get('monitor_manifest_sha256')!=cache_sha:raise ValueError('snapshot monitor cache mismatch')
                model=SpatialAnchoredResampler(ELAFV2Config(**payload['config'])).to(args.device)
                model.load_state_dict(payload['state_dict'])
                baseline=out/'representation-000000000.json'
                initial=json.loads(baseline.read_text())['initial_delta_rms'] if baseline.exists() else None
                result=diagnose_resampler(model,cache,initial_delta_rms=initial)
                torch.save(result,out/f'representation-{step:09d}.pt')
                report={'training_step':step,'num_samples':result['num_samples'],'initial_delta_rms':result['initial_delta_rms'],'metrics':result['metrics']}
                atomic_json(dest,report)
                if run:run.log({'training_step':step,**result['metrics']})
                del model,result
                print('representation complete',step,flush=True)
            for request in sorted((args.run_dir/'full_diagnostic_queue').glob('step-*.json')):
                receipt=out/(request.stem+'.done.json')
                if receipt.exists():continue
                job=json.loads(request.read_text());step=job.get('step',job.get('training_step'))
                checkpoint=job.get('checkpoint',job.get('checkpoint_path'))
                result=full_checkpoint_diagnostic(config,checkpoint,args.device)
                result['training_step']=step
                atomic_json(out/f'full-{step:09d}.json',result)
                if run:
                    rows=result['records'];metrics={}
                    for field in ['action_grad_norm','delta_grad_norm','cosine']:
                        vals=[r['lora_gradients']['global'][field] for r in rows]
                        metrics['monitor/lora/'+field]=sum(vals)/len(vals)
                    for stream in ['action','delta']:
                        for field in ['mse','prediction_rms','target_rms','zero_mse','cosine']:
                            vals=[r['flow_bins'][4][stream][field] for r in rows]
                            metrics[f'monitor/{stream}/{field}']=sum(vals)/len(vals)
                    for i in range(10):
                        for kind in ['delta','delta_analytic']:
                            for field in ['mse','prediction_rms','target_rms','zero_mse','cosine']:
                                vals=[r['flow_bins'][i][kind][field] for r in rows]
                                metrics[f'monitor/delta_bins/{i}/{kind}/{field}']=sum(vals)/len(vals)
                    for name in rows[0]['adapters']:
                        key=name.split('.layers.')[-1]
                        metrics['monitor/lora_ratio/'+key]=sum(r['adapters'][name]['ratio'] for r in rows)/len(rows)
                    run.log({'training_step':step,**metrics})
                atomic_json(receipt,{'status':'complete','training_step':step})
                print('full diagnostic complete',step,flush=True)
            # Emit concise transition artifacts only after both queues have
            # delivered diagnostics for that same checkpoint.
            for step in (0,1000,2000,3000,5000,10000,11000,12000,15000):
                report_path=out/f'report-{step:09d}.md'
                rep_path=out/f'representation-{step:09d}.json'
                full_path=out/f'full-{step:09d}.json'
                if report_path.exists() or not (rep_path.exists() and full_path.exists()):continue
                rep=json.loads(rep_path.read_text());full=json.loads(full_path.read_text())
                metric=rep['metrics'];rows=full['records']
                gl=[r['lora_gradients']['global'] for r in rows]
                lines=[f'# E-LAF v2 diagnostics: step {step}','',
                    f"- Current/Delta unique slot energy: {metric.get('monitor/z/unique_slot_energy'):.6g} / {metric.get('monitor/delta/unique_slot_energy'):.6g}",
                    f"- Delta RMS / Current effective rank / Delta effective rank: {metric.get('monitor/delta/rms_raw'):.6g} / {metric.get('monitor/z/global_erank'):.4g} / {metric.get('monitor/delta/global_erank'):.4g}",
                    f"- Slot CKA mean / per-head query JSD (current): {metric.get('monitor/z/slot_cka_mean'):.6g} / {metric.get('monitor/attention/current/per_head_pairwise_jsd_nats'):.6g} nats",
                    f"- Mean Action/Delta LoRA gradient norm: {sum(x['action_grad_norm'] for x in gl)/len(gl):.6g} / {sum(x['delta_grad_norm'] for x in gl)/len(gl):.6g}",
                    f"- Mean Action/Delta gradient cosine: {sum(x['cosine'] for x in gl)/len(gl):.6g}",
                    '', 'Delta time-bin MSE and analytic denoising baselines are in `full-'+f'{step:09d}.json`.',
                    'These are fixed-batch diagnostics, not validation or closed-loop performance.']
                report_path.write_text('\n'.join(lines)+'\n')
            status_path=args.run_dir/'status.json'
            status=json.loads(status_path.read_text()).get('status') if status_path.exists() else None
            if not args.watch or status in ['completed','paused','failed','decision_required']:break
            time.sleep(args.poll_seconds)
    finally:
        if run:run.finish()


if __name__=='__main__':main()
