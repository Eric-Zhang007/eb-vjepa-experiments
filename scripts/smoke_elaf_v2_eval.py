"""Actual-weight inference and encoder/cache equivalence before simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from elaf.evaluation.elaf_v2 import ELAFV2InferencePolicy
from elaf.v0_cache import encode_windows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    cfg=json.loads(args.config.read_text())
    policy=ELAFV2InferencePolicy(cfg)
    meta=json.loads((Path(cfg['checkpoint'])/'progress.json').read_text())
    manifest_path=Path(meta['contract']['config']['manifest'])
    manifest=json.loads(manifest_path.read_text()); ep=manifest['episodes'][0]
    root=manifest_path.parent/ep['path']
    images=np.load(root/'images.npy',mmap_mode='r')
    wrist=np.load(root/'wrist_images.npy',mmap_mode='r')
    # Proprio values are only needed for a finite-output smoke, not a score.
    st=policy.stats['libero_'+ep['suite']+'_no_noops']['proprio']
    proprio=(np.asarray(st['q01'])+np.asarray(st['q99']))/2
    observations=[]
    for t in [0,10]:
        history=[dict(image=np.array(images[i]), wrist_image=np.array(wrist[i]))
                 for i in range(max(0,t-3),t+1)]
        observations.append(dict(**history[-1],history=history,proprio=proprio,
                                 instruction=ep['instruction'],suite='libero_'+ep['suite'],noise_seed=239+t))
    online=policy.encode(observations)
    reference=encode_windows(policy.teacher,images,[0,10],policy.device)
    reference=torch.from_numpy(reference).view(torch.bfloat16).to(policy.device)
    torch.testing.assert_close(online,reference,atol=0,rtol=0)
    batched=policy.predict(observations)
    singles=np.concatenate([policy.predict([o]) for o in observations])
    difference=float(np.max(np.abs(batched-singles)))
    assert np.isfinite(batched).all() and batched.shape==(2,8,7)
    assert difference<.1, difference
    result=dict(encoder_matches_training_pipeline=True,actions_shape=list(batched.shape),
                batch_single_max_abs=difference,peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30,
                identity=policy.fingerprint())
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
