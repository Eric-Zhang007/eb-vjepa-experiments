"""Same-batch analytic baseline, token sensitivity and fresh fixed-batch optimization."""
import argparse
import json
from pathlib import Path
import torch
from elaf.v0_train import build_backbone, model_config
from elaf.v0_model import ELAFV0Policy
from elaf.v0_dataset import CachedV0Dataset, collate_v0
from elaf.v0_training import seed_all, load_trainable_state, parameter_groups, clip_grad_groups, atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--updates', type=int, default=60)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    torch.set_num_threads(2)
    seed_all(config['seed'])
    dataset = CachedV0Dataset(config['manifest'], config['checkpoint'], image_aug=False)
    indices = []
    for suite in sorted(dataset.suite_indices):
        for i in dataset.suite_indices[suite]:
            if dataset[i]['future_valid']:
                indices.append(i)
                break
    batch = collate_v0([dataset[i] for i in indices], pad_token_id=dataset.pad_token_id)
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    policy = ELAFV0Policy(build_backbone(config), model_config(config)).cuda().eval()
    # Keep fresh adapter/module state only; the frozen backbone is shared.
    fresh = {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}
    state = torch.load(args.checkpoint/'training.pt', map_location='cpu', weights_only=False)
    load_trainable_state(policy, state['model'])
    del state
    b = len(indices)
    noise_d = torch.randn(b,32,1024,device='cuda',dtype=torch.float32)
    noise_a = torch.randn(b,8,7,device='cuda',dtype=torch.bfloat16)
    t = torch.full((b,), .5, device='cuda')
    def forward(u=t, nd=noise_d):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return policy(batch,u=u,s=t,delta_noise=nd,action_noise=noise_a,return_debug=True)
    def numbers(result):
        valid=batch['future_valid'][:,None,None].expand_as(result['delta_velocity'])
        target=result['delta_velocity_target'].float()[valid]
        pred=result['delta_velocity'].float()[valid]
        analytic=(-2*result['noisy_delta'].float())[valid]
        raw=result['delta_target'].float()[valid]
        return dict(delta_model_mse=(pred-target).square().mean().item(),
            delta_zero_mse=target.square().mean().item(),delta_analytic_mse=(analytic-target).square().mean().item(),
            analytic_expected_mse=(4*raw.square().mean()).item(),delta_rms=raw.square().mean().sqrt().item(),
            action_model_mse=result['action_loss'].item(),delta_prediction_rms=pred.square().mean().sqrt().item())
    hooks=[];activations={}
    for name in ['delta_in','delta_out','action_out']:
        module=getattr(policy,name)
        def hook(m,inp,out,name=name):
            activations[name+'/input_rms']=inp[0].detach().float().square().mean().sqrt().item()
            activations[name+'/output_rms']=out.detach().float().square().mean().sqrt().item()
        hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        ref=forward();report=numbers(ref);report['activations']=dict(activations)
        flipped=forward(nd=-noise_d)
        changed=forward(u=torch.full_like(t,.25))
        report['delta_output_change_noise_flip_rms']=(ref['delta_velocity'].float()-flipped['delta_velocity'].float()).square().mean().sqrt().item()
        report['delta_output_change_u025_rms']=(ref['delta_velocity'].float()-changed['delta_velocity'].float()).square().mean().sqrt().item()
        # Separate time embedding sensitivity: hold all input tokens fixed.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            condition,mask=policy._condition(batch)
            _,time_only=policy.velocity(condition,mask,ref['current_state'],ref['noisy_delta'],ref['noisy_actions'],torch.full_like(t,.25),t,batch['action_valid'])
        report['delta_output_change_time_only_rms']=(ref['delta_velocity'].float()-time_only.float()).square().mean().sqrt().item()
    for h in hooks:h.remove()
    report.update(sample_ids=batch.get('sample_id'),indices=indices,checkpoint=str(args.checkpoint),u=.5,s=.5,augmentation=False)
    atomic_json(args.output/'analytic_baseline.json',report)
    print(json.dumps(report),flush=True)
    del ref,flipped,changed,time_only
    with torch.no_grad():
        for n,p in policy.named_parameters():
            if n in fresh:p.copy_(fresh[n])
    del fresh
    optimizer=torch.optim.AdamW(parameter_groups(policy,config['optimizer']))
    policy.train()
    for step in range(args.updates+1):
        optimizer.zero_grad(set_to_none=True)
        result=forward()
        metrics=dict(step=step,**numbers(result))
        with (args.output/'fresh_fixed_batch.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        if step%10==0:print(json.dumps(metrics),flush=True)
        if step==args.updates:break
        result['loss'].backward()
        clip_grad_groups(optimizer,1.)
        optimizer.step()
    atomic_json(args.output/'status.json',{'status':'completed','updates':args.updates})


if __name__=='__main__':main()
