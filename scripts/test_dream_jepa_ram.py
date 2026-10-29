"""Compare online, cold-cache, warm-cache and partial-hit encoder outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['OMP_NUM_THREADS'] = '2'
import json
from pathlib import Path
import sys
import time
import torch
import numpy as np

ROOT = Path('/data/users/jiachangzhang/eb_vjepa')
EXP = ROOT/'experiments/dream_jepa_dualview_20261023'
sys.path.insert(0,str(EXP/'infra_source/vla'))
from dreamvla.jepa_runtime import FrozenGrid11k
config=json.loads((EXP/'resolved_config.json').read_text())
os.environ['DREAM_JEPA_RAM_SOCKET']=str(EXP/'wrist_ram.sock')
encoder=FrozenGrid11k(ROOT,config['jepa_checkpoint'],'cuda:0')
encoder.cache_namespace += 'test-' + str(time.time_ns()) + ':'
entries=list(encoder.index.items())
def batch(indices):
    selected=[entries[i] for i in indices]
    return dict(jepa_episode_key=[torch.tensor(list(bytes.fromhex(k.split(':')[1])),dtype=torch.uint8) for k,e in selected],
                jepa_frame=[min(20,e['length']-1) for k,e in selected],dataset_names=[k.split(':')[0] for k,e in selected])
test=batch([0,2,4,6,8,10,12,14])
cache=encoder.cache
encoder.cache=None
torch.cuda.synchronize(); start=time.monotonic(); ref=encoder(test);torch.cuda.synchronize();online=time.monotonic()-start
encoder.cache=cache
cold=encoder(test)
if encoder.cache_write is not None: encoder.cache_write.result()
torch.cuda.synchronize();start=time.monotonic();warm=encoder(test);torch.cuda.synchronize();cached=time.monotonic()-start
for values in (cold,warm):
    for a,b in zip(ref,values):assert torch.equal(a,b),(a-b).abs().max().item()
partial=batch([0,3,4,7,8,11,12,15])
encoder.cache=None; ref2=encoder(partial);encoder.cache=cache;partial_out=encoder(partial)
if encoder.cache_write is not None: encoder.cache_write.result()
for a,b in zip(ref2,partial_out):assert torch.equal(a,b),(a-b).abs().max().item()
result=dict(exact_online_cold_warm=True,exact_partial_hit_repacking=True,online_seconds=online,warm_seconds=cached,
            cache_stats=cache.request('stats'),partial_metrics=encoder.last_metrics)
(EXP/'infra_encoder_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
