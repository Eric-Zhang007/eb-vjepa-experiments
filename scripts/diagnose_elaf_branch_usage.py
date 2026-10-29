"""Paired branch-content and attention-edge interventions; no training."""
import argparse
import json
from pathlib import Path
import torch
from elaf.v0_train import build_backbone, model_config
from elaf.v0_model import ELAFV0Policy
from elaf.v0_dataset import CachedV0Dataset, collate_v0
from elaf.v0_training import seed_all, load_trainable_state, atomic_json


def block_mask(mask, mode, current_tokens=32, delta_tokens=32, action_tokens=8):
    if mask.dtype != torch.bool or mask.ndim != 4:
        raise ValueError('Expected SDPA bool [B,1,L,L] allowed mask')
    length=mask.shape[-1]
    c=length-current_tokens-delta_tokens-action_tokens
    d=c+current_tokens;a=d+delta_tokens
    result=mask.clone()
    if mode=='block_action_delta':result[:,:,a:,d:a]=False
    elif mode=='block_action_current':result[:,:,a:,c:d]=False
    elif mode=='isolate_delta':
        result[:,:,:d,d:a]=False
        result[:,:,a:,d:a]=False
    elif mode!='full':raise ValueError(mode)
    return result


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
    p=args.run_dir;out=p/'branch_usage_native';out.mkdir(exist_ok=False)
    contract=json.loads((p/'resolved_config.json').read_text());config=contract['config']
    checkpoint=(p/'checkpoints/latest').resolve()
    metadata=json.loads((checkpoint/'progress.json').read_text())
    if metadata['progress']['step']!=2000:raise ValueError('Expected fixed step2000 checkpoint')
    seed_all(731);torch.set_num_threads(2)
    policy=ELAFV0Policy(build_backbone(config),model_config(config)).cuda().eval()
    state=torch.load(checkpoint/'training.pt',map_location='cpu',weights_only=False)
    load_trainable_state(policy,state['model']);del state
    dataset=CachedV0Dataset(config['manifest'],config['checkpoint'],image_aug=False)
    indices=[]
    for j in range(8):
        for suite in sorted(dataset.suite_indices):
            members=dataset.suite_indices[suite];offset=j*len(members)//8
            while not bool(dataset[members[offset]]['future_valid']):offset+=1
            indices.append(members[offset])
    from dreamvla.modeling_dreamvl import DreamVLFlashAttention2
    import dreamvla.modeling_dreamvl as dream
    layers=[m for m in policy.backbone.model.modules() if isinstance(m,DreamVLFlashAttention2)]
    if not layers:raise RuntimeError('No decoder FlashAttention layers found')
    intervention={'mode':'full','checked_layers':0}
    original_flash=dream._flash_attention_forward
    def native_intervention(q,k,v,mask,query_length,**kwargs):
        intervention['checked_layers']+=1
        result=original_flash(q,k,v,mask,query_length,**kwargs)
        mode=intervention['mode'];length=q.shape[1]
        if mode=='full':return result
        if mask.ndim!=2 or query_length!=length or k.shape[1]!=length:
            raise RuntimeError('Expected full-sequence 2D-mask attention, without KV cache')
        if mode in ['block_action_delta','block_action_current']:
            changed=mask.clone()
            key_slice=slice(length-40,length-8) if mode=='block_action_delta' else slice(length-72,length-40)
            changed[:,key_slice]=False
            result[:,-8:]=original_flash(q[:,-8:],k,v,changed,8,**kwargs)
        elif mode=='isolate_delta':
            idx=torch.cat([torch.arange(length-40,device=q.device),torch.arange(length-8,length,device=q.device)])
            result[:,idx]=original_flash(q[:,idx],k[:,idx],v[:,idx],mask[:,idx],length-32,**kwargs)
        else:raise ValueError(mode)
        return result
    records=[];validation={};hook=None
    def mse(x,y,valid):
        return (x.float()-y.float()).square()[valid.expand_as(x)].mean().item()
    def velocity(condition,mask,z,du,at,u,s,valid,mode='full'):
        intervention['mode']=mode;before=intervention['checked_layers']
        with torch.autocast('cuda',dtype=torch.bfloat16):v=policy.velocity(condition,mask,z,du,at,u,s,valid)
        if hook is not None and intervention['checked_layers']-before!=len(layers):
            raise RuntimeError('Not every layer received intervention')
        return v
    for start in range(0,32,4):
        batch=collate_v0([dataset[i] for i in indices[start:start+4]],pad_token_id=dataset.pad_token_id)
        batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):condition,mask=policy._condition(batch)
        z=policy._resample(batch['current_features']);f=policy._resample(batch['future_features'])
        d=(f-z)/policy.delta_scale
        valid_a=batch['action_valid'];valid_d=batch['future_valid'];d=d.masked_fill(~valid_d[:,None,None],0)
        action=batch['actions'].to(condition.dtype).masked_fill(~valid_a[:,:,None],0)
        nd=torch.randn_like(d);na=torch.randn_like(action);s=torch.full((4,),.5,device='cuda')
        at=torch.lerp(na,action,s.to(action.dtype)[:,None,None]);ta=action.float()-na.float();td=d.float()-nd.float()
        if start==0:
            u=s;du=torch.lerp(nd,d,u[:,None,None])
            fa=velocity(condition,mask,z,du,at,u,s,valid_a)
            dream._flash_attention_forward=native_intervention
            hook=True
            sd=velocity(condition,mask,z,du,at,u,s,valid_a)
            validation['decoder_layers']=len(layers)
            for name,x,y in zip(['action','delta'],fa,sd):
                change=(x.float()-y.float()).square().mean().sqrt()
                relative=(change/x.float().square().mean().sqrt().clamp_min(1e-8)).item()
                validation[name+'_native_wrapper_relative_rms']=relative
                if relative!=0:raise RuntimeError('Unblocked native wrapper changed output')
            isolated=velocity(condition,mask,z,du,at,u,s,valid_a,'isolate_delta')[0]
            isolated_flipped=velocity(condition,mask,z,-du,at,u,s,valid_a,'isolate_delta')[0]
            change=(isolated.float()-isolated_flipped.float()).abs().max().item()
            validation['isolated_delta_flip_action_max_abs']=change
            if change>1e-5:raise RuntimeError('Delta still influences Action after global key isolation')
            atomic_json(out/'validation.json',validation)
        for bin_id in range(10):
            tau=(bin_id+.5)/10;u=torch.full((4,),tau,device='cuda')
            du=torch.lerp(nd,d,u[:,None,None]);full=velocity(condition,mask,z,du,at,u,s,valid_a)
            analytic=-du/(1-tau)
            for mode in ['full','shuffle_delta_content','block_action_delta','block_action_current','isolate_delta']:
                if mode=='full':va,vd=full
                else:
                    alt_du=torch.lerp(nd,d.roll(1,0),u[:,None,None]) if mode=='shuffle_delta_content' else du
                    va,vd=velocity(condition,mask,z,alt_du,at,u,s,valid_a,'full' if mode=='shuffle_delta_content' else mode)
                records.append(dict(batch=start//4,u=tau,s=.5,mode=mode,sample_ids=batch['sample_id'],
                    donor_sample_ids=batch['sample_id'][-1:]+batch['sample_id'][:-1] if mode=='shuffle_delta_content' else None,
                    action_mse=mse(va,ta,valid_a[...,None]),delta_mse=mse(vd,td,valid_d[:,None,None]),
                    delta_zero_mse=mse(torch.zeros_like(td),td,valid_d[:,None,None]),
                    delta_analytic_mse=mse(analytic,td,valid_d[:,None,None]),
                    delta_analytic_expected=mse(d/(1-tau),torch.zeros_like(d),valid_d[:,None,None]),
                    action_change_rms=mse(va,full[0],valid_a[...,None])**.5,
                    delta_change_rms=mse(vd,full[1],valid_d[:,None,None])**.5))
        atomic_json(out/'results.json',dict(checkpoint=str(checkpoint),sample_count=32,indices=indices,validation=validation,
            protocol='32 fixed training samples, no augmentation; u bin midpoints .05 through .95; s=.5; same noise, original target; native FlashAttention for all interventions; shuffle only clean delta content across suites',
            warning='Direct-edge masks retain indirect routes. isolate_delta prevents all non-Delta tokens reading Delta keys at every layer. Inference distribution shift; no closed-loop or auxiliary-training benefit claim.',records=records))
        print('completed',start+4,'samples',flush=True)
    atomic_json(out/'status.json',{'status':'completed'})


if __name__=='__main__':main()
