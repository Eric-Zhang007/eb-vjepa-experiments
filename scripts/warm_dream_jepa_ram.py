"""Bounded background warm-up of immutable wrist features; no training state."""
import argparse
import os
parser=argparse.ArgumentParser()
parser.add_argument('--gpu',required=True)
parser.add_argument('--shard',type=int,required=True)
args=parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
os.environ['OMP_NUM_THREADS']='2'
import sys
from pathlib import Path
import json
import time
import random
from collections import OrderedDict
import numpy as np
import torch
ROOT=Path('/data/users/jiachangzhang/eb_vjepa')
EXP=ROOT/'experiments/dream_jepa_dualview_20261023'
sys.path.insert(0,str(EXP/'infra_source/vla'))
os.environ['DREAM_JEPA_RAM_SOCKET']=str(EXP/'wrist_ram.sock')
from dreamvla.jepa_runtime import FrozenGrid11k
cfg=json.loads((EXP/'resolved_config.json').read_text())
encoder=FrozenGrid11k(ROOT,cfg['jepa_checkpoint'],'cuda:0')
frames=[(key,t,ep['path']) for key,ep in encoder.index.items() for t in range(ep['length'])]
random.Random(987123).shuffle(frames)
frames=frames[args.shard::2]
maps=OrderedDict();count=0;started=time.monotonic()
for start in range(0,len(frames),8):
    stats=encoder.cache.request('stats')
    if stats['bytes'] >= .9*stats['capacity']:
        print('Cache warm capacity reached',flush=True);break
    selected=frames[start:start+8]
    keys=[encoder.cache_namespace+k+':'+str(t) for k,t,p in selected]
    values=encoder.cache.get(keys)
    missing=[i for i,v in enumerate(values) if v is None]
    if not missing:continue
    clips=[]
    for i in missing:
        key,t,p=selected[i]
        if p not in maps:maps[p]=np.load(encoder.cache_root/p/'wrist_images.npy',mmap_mode='r')
        maps.move_to_end(p)
        while len(maps)>64:maps.popitem(last=False)
        clips.append(np.asarray(maps[p][np.maximum(np.arange(t-3,t+1),0)]))
    videos=torch.from_numpy(np.stack(clips)).pin_memory()
    raw=encoder.encode_videos(videos,pad_batch=True).cpu().numpy()
    encoder.cache.put([(keys[i],raw[j].tobytes()) for j,i in enumerate(missing)])
    count+=len(missing)
    if start%256==0:
        result=dict(frames=count,seconds=time.monotonic()-started,cache=encoder.cache.request('stats'))
        (EXP/f'wrist_warm_{args.shard}.json').write_text(json.dumps(result));print(json.dumps(result),flush=True)
print(json.dumps(dict(status='done',frames=count,seconds=time.monotonic()-started)),flush=True)
