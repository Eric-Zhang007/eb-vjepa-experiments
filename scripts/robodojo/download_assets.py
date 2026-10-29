"""Resume official RoboDojo Assets-only download, trying mirror before proxy."""
import os, subprocess, sys
from pathlib import Path
ROOT=Path('/data/users/jiachangzhang/eb_vjepa')
OUT=ROOT/'experiments/robodojo_setup'
code = "from huggingface_hub import snapshot_download; snapshot_download('RoboDojo-Benchmark/RoboDojo', repo_type='dataset', allow_patterns=['Assets/**'], local_dir='/data/users/jiachangzhang/eb_vjepa/repos/RoboDojo', max_workers=4)"
rc=1
for mirror in [True,False]:
    env=os.environ.copy()
    env['HF_HUB_DISABLE_XET']='1'
    if mirror:
        env['HF_ENDPOINT']='https://hf-mirror.com'
        for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']: env.pop(k,None)
    else:
        env.pop('HF_ENDPOINT',None)
        env['HTTP_PROXY']=env['HTTPS_PROXY']='http://127.0.0.1:7890'
    rc=subprocess.call([sys.executable,'-c',code],env=env)
    if rc==0: break
if rc==0:
    for name in ['Robots','Object','Material','Eval_Layout']:
        if not (ROOT/'repos/RoboDojo/Assets'/name).is_dir(): rc=2
(OUT/'assets.exitcode').write_text(str(rc))
sys.exit(rc)
