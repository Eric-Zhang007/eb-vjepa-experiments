"""Real checkpoint native equivalence check; run with a reserved GPU."""
import ast,json,argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from elaf.evaluation.official_dream import OfficialDreamPolicy
from elaf.evaluation.policy import initial_noise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
    cfg=json.loads(Path(a.config).read_text());p=OfficialDreamPolicy(cfg)
    rng=np.random.default_rng(16)
    obs=[dict(image=rng.integers(0,256,(256,256,3),dtype=np.uint8),wrist_image=rng.integers(0,256,(256,256,3),dtype=np.uint8),proprio=np.zeros(8,np.float32),instruction=text,noise_seed=17+i,suite='libero_10') for i,text in enumerate(['pick up the book','put the yellow and white mug in the microwave and close it'])]
    source=Path(cfg['repo'])/'vla/experiments/robot/libero/run_libero_eval_dream.py'
    tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_action')
    from experiments.robot.openvla_utils import center_crop_image,normalize_proprio,resize_image_for_policy
    namespace=dict(Image=Image,torch=torch,DEVICE=p.device,center_crop_image=center_crop_image,normalize_proprio=normalize_proprio)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    x=obs[0];nativeobs=dict(full_image=resize_image_for_policy(x['image'],224),wrist_image=resize_image_for_policy(x['wrist_image'],224),state=x['proprio'].copy())
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),patch.object(p.head,'sample_noise',return_value=initial_noise([x],(8,7),p.device,torch.bfloat16)):
        native=namespace['get_action'](p.model,p.processor,nativeobs,x['instruction'],'libero_10_no_noops',center_crop=True,action_head=p.head,proprio_projector=p.proprio,noisy_action_projector=p.noisy,num_images_in_input=2,use_flow_matching=True,action_chunk=8,steps=p.steps)
    singles=np.concatenate([p.predict([x]) for x in obs]);batch=p.predict(obs)
    np.testing.assert_allclose(singles[0],native,rtol=0,atol=1e-6)
    np.testing.assert_allclose(batch,singles,rtol=.03,atol=.015)
    np.testing.assert_array_equal(p.predict(obs),batch)
    report=dict(native_batch1_max_error=float(np.abs(singles[0]-native).max()),batch2_max_error=float(np.abs(batch-singles).max()),shape=list(batch.shape),fingerprint=p.fingerprint())
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
