"""Build deterministic links from native RLDS episodes to existing raw features."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT=Path('/data/users/jiachangzhang/eb_vjepa')
cache=ROOT/'data/elaf_v0_grid11k_raw'
manifest=json.loads((cache/'manifest.json').read_text())
assert manifest['complete']
assert manifest['encoder_sha256']=='468d7af3a5bdae9fb2205718ab0f86a13f5395afcc53700500439d993baf987a'
index={}
for ep in manifest['episodes']:
    directory=cache/ep['path']
    image=np.load(directory/'images.npy',mmap_mode='r')[0]
    key=tf.raw_ops.Fingerprint(data=tf.reshape(tf.io.serialize_tensor(tf.convert_to_tensor(image)),[1]),method='farmhash64').numpy()[0].tobytes().hex()
    identity=f"libero_{ep['suite']}_no_noops:{key}"
    if identity in index:
        raise ValueError('Ambiguous episode identity: '+identity)
    features=np.load(directory/'features.npy',mmap_mode='r')
    assert features.shape==(ep['length'],1152,1024) and features.dtype==np.uint16
    record=dict(path=ep['path'],length=ep['length'])
    for field,filename in [('primary','images.npy'),('wrist','wrist_images.npy')]:
        path=directory/filename
        arr=np.load(path,mmap_mode='r')
        assert arr.dtype==np.uint8 and arr.shape[-1]==3 and arr.shape[0]==ep['length']
        record[field+'_prefix']=f'npyframe|{path}|{arr.offset}|{arr.shape[1]}|{arr.shape[2]}|'
    index[identity]=record
output=ROOT/'experiments/dream_jepa_dualview_20261023/episode_index.json'
output.write_text(json.dumps(index))
print(f'Indexed {len(index)} unique episodes: {output}')
