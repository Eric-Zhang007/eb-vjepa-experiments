"""Measure real-shape forward/backward throughput before fixing micro-batch."""
import argparse
import json
from pathlib import Path
import time
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-checkpointing',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    cfg=OmegaConf.load(args.output/'train_config.yaml')
    model=instantiate(cfg.model,model_dtype=torch.float32,device='cuda:0')
    model.load_checkpoint(str(args.output/'initialization.pt'));model.train()
    if args.no_checkpointing:model.predictor_checkpointing=False
    s=torch.load(args.output/'preflight/real_sample.pt',map_location='cpu',weights_only=False)
    base=dict(z0=s['video_latent'][0].float(),z_future=s['video_latent'][1:].float(),
        action_chunk=s['action'][:16].float(),proprio=s['proprio'][0].float(),
        text_context=s['context'],text_mask=s['context_mask'].bool(),
        action_is_pad=s['action_is_pad'][:16].bool(),image_is_pad=s['image_is_pad'].bool())
    results=[]
    for n in ([40] if args.no_checkpointing else [8,20,40]):
        b={k:v.unsqueeze(0).repeat(n,*([1]*v.ndim)).cuda() for k,v in base.items()}
        torch.cuda.reset_peak_memory_stats();times=[]
        for i in range(3):
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=model(b)
            loss.backward();torch.cuda.synchronize()
            if i:times.append(time.perf_counter()-start)
        row=dict(batch=n,seconds=sum(times)/len(times),peak_gib=torch.cuda.max_memory_allocated()/2**30)
        row['samples_per_second']=n/row['seconds'];results.append(row)
        print(json.dumps(row),flush=True)
        del b,loss;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    name='no_checkpointing_benchmark.json' if args.no_checkpointing else 'microbatch_benchmark.json'
    (args.output/name).write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
