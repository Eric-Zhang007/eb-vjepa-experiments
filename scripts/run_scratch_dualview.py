"""Supervise prepared inputs, real smoke/resume, fresh training and local evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from omegaconf import OmegaConf


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root,out=a.root,a.output
    lock=(out/'experiment.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),started=time.time(),gpus=[0,1,2,3],global_batch=128,target_step=21360)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
         'TOKENIZERS_PARALLELISM':'false','TMPDIR':str(root/'cache/tmp'),'HF_HOME':str(root/'cache/huggingface'),
         'PYTHONPATH':str(root/'repos/RoboActualizer/src')+':'+str(root/'envs/eb_vjepa_pretrain/lib/python3.11/site-packages')}
    def status(**kw):
        state.update(kw);f=out/'status.tmp';f.write_text(json.dumps(state,indent=2));f.replace(out/'status.json')
    def run(cmd,phase,extra_env=None):
        with (out/(phase+'.log')).open('a') as f:
            child=subprocess.Popen(cmd,cwd=root/'repos/RoboActualizer',env={**env,**(extra_env or {})},
                                   stdout=f,stderr=f,stdin=subprocess.DEVNULL)
            status(phase=phase,child_pid=child.pid,command=cmd)
            rc=child.wait()
        if rc:raise RuntimeError(f'{phase} exited {rc}')
    def train(name):
        return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                str(root/'repos/RoboActualizer/scripts/train.py'),'--config-path',str(out),'--config-name',name,
                'hydra.run.dir='+str(out/'hydra'/name)]
    try:
        status(phase='preparing_inputs')
        while True:
            cache=json.loads((out/'cache_status.json').read_text())
            if cache['phase']=='failed':raise RuntimeError(cache)
            f=out/'text_download_status.json'
            # Text preparation reports its own encoder-specific failures.
            text=out/'text_cache_status.json'
            if text.exists() and json.loads(text.read_text())['status']=='failed':raise RuntimeError(text.read_text())
            if cache['phase']=='complete' and text.exists() and json.loads(text.read_text())['status']=='complete':break
            time.sleep(10)
        run([sys.executable,str(root/'scripts/preflight_scratch_dualview.py'),'--output',str(out)],'preflight',{'CUDA_VISIBLE_DEVICES':'0'})
        if json.loads((out/'preflight/results.json').read_text())['status']!='passed':raise RuntimeError('Preflight failed')
        cfg=OmegaConf.load(out/'train_config.yaml');smoke=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        smoke.output_dir=str(out/'smoke');smoke.max_steps=3;smoke.log_every=1;smoke.wandb.enabled=False;smoke.num_workers=2
        d=Path(smoke.output_dir);d.mkdir(exist_ok=False);(d/'PAUSE_REQUEST').touch()
        OmegaConf.save(smoke,out/'smoke.yaml');run(train('smoke'),'smoke_pause')
        first=json.loads((d/'checkpoints/state/latest/trainer_state.json').read_text());assert first['global_step']==1
        (d/'PAUSE_REQUEST').unlink();smoke.resume=str(d/'checkpoints/state/latest')
        OmegaConf.save(smoke,out/'smoke_resume.yaml');run(train('smoke_resume'),'smoke_resume')
        last=json.loads((d/'checkpoints/state/latest/trainer_state.json').read_text());assert last['global_step']==3 and last['batch_in_epoch']==3
        import torch
        opt=torch.load(d/'checkpoints/state/latest/optimizer.bin',map_location='cpu',weights_only=False)
        steps=sorted({int(v['step']) for v in opt['state'].values() if 'step' in v});assert steps==[3]
        (out/'ddp_resume_test.json').write_text(json.dumps(dict(status='passed',world_size=4,pause_step=1,resume_step=3,optimizer_steps=steps)))
        del opt
        # Fresh formal model/optimizer: no smoke or old action-head weights imported.
        assert cfg.resume is None
        run(train('train_config'),'train')
        last=json.loads((out/'train/checkpoints/state/latest/trainer_state.json').read_text())
        if last['global_step']!=21360:status(phase='paused',step=last['global_step']);return
        run([sys.executable,str(out/'evaluation_021360/supervisor.py')],'evaluation')
        results={kind:{k:v for k,v in json.loads((out/'evaluation_021360'/kind/'summary.json').read_text()).items() if k!='records'} for kind in ['clean','plus']}
        status(phase='completed',results=results,finished=time.time())
    except Exception as e:
        status(phase='failed',error=repr(e));raise


if __name__=='__main__':main()
