"""Check immutable frame references against native TFDS pixels and resize."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

ROOT=Path('/data/users/jiachangzhang/eb_vjepa')
EXPERIMENT=ROOT/'experiments/dream_jepa_dualview_20261023'
sys.path.insert(0,str(EXPERIMENT/'source/vla'))
from prismatic.vla.datasets.rlds.obs_transforms import decode_and_resize

index=json.loads((EXPERIMENT/'episode_index.json').read_text())
checks=[]
for suite in ('spatial','object','goal','10'):
    path=Path('/data/users/jiachangzhang/DNFT4RP/upstream/vla_adapter/datasets/rlds')/f'libero_{suite}_no_noops/1.0.0'
    builder=tfds.builder_from_directory(str(path))
    episode=next(iter(builder.as_dataset(split='train',shuffle_files=False)))
    rows=list(tfds.as_numpy(episode['steps']))
    first=tf.convert_to_tensor(rows[0]['observation']['image'])
    fingerprint=tf.raw_ops.Fingerprint(data=tf.reshape(tf.io.serialize_tensor(first),[1]),method='farmhash64').numpy()[0].tobytes().hex()
    ep=index[f'libero_{suite}_no_noops:{fingerprint}']
    for field,filename,key in [('primary','images.npy','image'),('wrist','wrist_images.npy','wrist_image')]:
        cached=np.load(ROOT/'data/elaf_v0_grid11k_raw'/ep['path']/filename,mmap_mode='r')
        assert np.array_equal(cached,np.stack([r['observation'][key] for r in rows]))
        for t in (0,len(rows)//2,len(rows)-1):
            referenced=decode_and_resize({'image_primary':tf.constant(ep[field+'_prefix']+str(t))},resize_size=(224,224),depth_resize_size={})['image_primary']
            native=decode_and_resize({'image_primary':tf.convert_to_tensor(rows[t]['observation'][key])},resize_size=(224,224),depth_resize_size={})['image_primary']
            assert np.array_equal(referenced.numpy(),native.numpy())
        checks.append(dict(suite=suite,camera=field,all_episode_pixels_equal=True,three_resized_frames_equal=True))
(EXPERIMENT/'pixel_equivalence.json').write_text(json.dumps(checks,indent=2))
print('PASS native TFDS/cache equality and frame-reference resize equality for both cameras in all four suites')
