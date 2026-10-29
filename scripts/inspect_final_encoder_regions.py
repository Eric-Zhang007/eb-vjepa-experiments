"""Add the final 103233-step encoder to the unchanged task ROI protocol."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();r=args.root
    source=r/'experiments/libero_encoder_longrun_latents_20261021'
    out=r/'experiments/libero_encoder_final103233_regions_20261021'
    out.mkdir(exist_ok=True)
    names=['official','grid11000','full10000','full40000','full100000']
    for name in ['samples.json','images.npy']+[n+'.npy' for n in names]:
        dest=out/name
        if not dest.exists():os.link(source/name,dest)
    identities=json.loads((source/'identities.json').read_text())
    identities={n:identities[n] for n in names}
    from elaf.v0_cache import encode_windows,sha256
    checkpoint=r/'experiments/ctp_full_72h/embodied_vjepa_encoder_103233.pt'
    identities['full103233']=dict(path=str(checkpoint),sha256=sha256(checkpoint))
    (out/'identities.json').write_text(json.dumps(identities,indent=2))
    torch.set_num_threads(4)
    factory=r/'repos/RoboActualizer/src/Roboactualizer/models/roboactualizer/vjepa.py'
    spec=importlib.util.spec_from_file_location('factory',factory)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    teacher=module.build_vjepa_encoder(ckpt_path=str(checkpoint),model_size='large',
        image_size=(384,384),torch_hub_dir=str(r/'repos/vjepa2')).eval().requires_grad_(False).to('cuda:0')
    samples=json.loads((out/'samples.json').read_text());values=[]
    for i,s in enumerate(samples):
        image=np.load(r/'data/elaf_v0_grid11k_raw'/s['path']/'images.npy',mmap_mode='r')
        bits=encode_windows(teacher,image,[s['t'],s['future']],'cuda:0')
        values.append(torch.from_numpy(bits).view(torch.bfloat16).float().numpy())
        if i%8==0:print('encoded',i,flush=True)
    np.save(out/'full103233.npy',np.stack(values).astype(np.float16))
    del teacher,values;torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(r/'scripts/inspect_task_region_latents.py'),'--input',str(out)],check=True)
    (out/'status.json').write_text(json.dumps(dict(state='completed',encoder_step=103233)))


if __name__=='__main__':main()
