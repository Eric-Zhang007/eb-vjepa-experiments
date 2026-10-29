"""Real-cache checks; disposable updates do not seed the fresh formal run."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from Roboactualizer.utils import misc


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    cfg=OmegaConf.load(out/'train_config.yaml');torch.set_num_threads(2);torch.manual_seed(cfg.seed)
    assert cfg.resume is None and 'warm_start_source' not in cfg.model
    misc.register_work_dir(out/'preflight')
    ds=instantiate(cfg.data.train);sample=ds[20]
    torch.save(sample,out/'preflight/real_sample.pt')
    def batch(n):
        s={k:v.unsqueeze(0).repeat(n,*([1]*v.ndim)).cuda() for k,v in sample.items() if torch.is_tensor(v)}
        return dict(z0=s['video_latent'][:,0].float(),z_future=s['video_latent'][:,1:].float(),
                    action_chunk=s['action'][:,:32].float(),proprio=s['proprio'][:,0].float(),
                    text_context=s['context'],text_mask=s['context_mask'].bool(),
                    action_is_pad=s['action_is_pad'][:,:32].bool(),image_is_pad=s['image_is_pad'].bool())
    model=instantiate(cfg.model,model_dtype=torch.float32,device='cuda')
    model.save_checkpoint(str(out/'initialization_audit.pt'),step=0)
    b=batch(1);noise=torch.randn_like(b['action_chunk']);sigma=torch.full((1,1,1),.5,device='cuda');t=torch.full((1,),500.,device='cuda')
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        original=model.prediction_and_targets(b,noise,sigma,t)
        changed=model.prediction_and_targets(dict(b,z_future=b['z_future']*3+5),noise,sigma,t)
        assert torch.equal(original[0],changed[0]) and torch.equal(original[2],changed[2])
    # Raw pixel and cached feature paths use the same camera isolation/normalization.
    rawcfg=OmegaConf.create(OmegaConf.to_container(cfg.data.train,resolve=True));rawcfg.rgb_latent_cache_dir=None
    raw=instantiate(rawcfg)[20]
    with torch.no_grad():
        z=model.backbone.encode_clip_last_group(raw['video'][:1].cuda()).half().float().cpu()
        torch.testing.assert_close(z[0],sample['video_latent'][0].float(),rtol=.01,atol=.01)
    model.train();optim=torch.optim.AdamW(model.trainable_parameters(),lr=1e-4)
    report=dict(status='running',parameters=sum(p.numel() for p in model.trainable_parameters()),
                shapes={k:list(v.shape) for k,v in b.items()},initialization='random',future_target_leakage=False,
                cache_online_match=True,updates=[])
    # Verify actual configured microbatch fits, including AdamW moments.
    big=batch(int(cfg.batch_size));torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        optim.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,logs=model(big)
        loss.backward();g=torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True)
        optim.step();report['updates'].append(dict(loss=float(loss.detach()),grad_norm=float(g)))
    report['peak_memory_gib']=torch.cuda.max_memory_allocated()/2**30
    del big;optim.zero_grad(set_to_none=True);model.eval()
    # Compare inference implementations after nonzero updates, not zero-init outputs.
    generated=[]
    with patch.object(model.backbone,'encode_clip_last_group',return_value=b['z0']),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for cache,graph in [(False,False),(True,False),(True,True)]:
            torch.manual_seed(239);torch.cuda.manual_seed_all(239)
            generated.append(model.generate(b['z0'],b['text_context'],b['text_mask'],b['proprio'],num_inference_steps=4,use_kv_cache=cache,use_cuda_graph=graph).clone())
    for x in generated[1:]:torch.testing.assert_close(generated[0],x,rtol=.02,atol=.02)
    report.update(status='passed',inference_path_max_abs_diffs=[float((generated[0]-x).abs().max()) for x in generated[1:]])
    (out/'preflight/results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
