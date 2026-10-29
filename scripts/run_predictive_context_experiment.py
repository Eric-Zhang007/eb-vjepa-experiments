"""Coordinate cache completion, six-GPU resume smoke, training, and local evaluation.

All learning is performed by RoboActualizer/scripts/train.py and its existing
trainer. This file only supervises processes and records their commands/status.
"""
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
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); r, out = args.root, args.output
    lock = (out/'experiment.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(supervisor_pid=os.getpid(), training_gpus=list(range(6)), started=time.time())
    def status(**kw):
        state.update(kw); tmp=out/'status.tmp'; tmp.write_text(json.dumps(state, indent=2)); tmp.replace(out/'status.json')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5', 'OMP_NUM_THREADS':'2',
        'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'TOKENIZERS_PARALLELISM':'false',
        'TMPDIR':str(r/'cache/tmp'), 'HF_HOME':str(r/'cache/huggingface'),
        'HF_DATASETS_CACHE':str(r/'cache/huggingface/datasets'),
        'PYTHONPATH':f'{r}/repos/RoboActualizer/src:{r}/envs/eb_vjepa_pretrain/lib/python3.11/site-packages'}
    def execute(cmd, phase, filename):
        with (out/filename).open('a') as log:
            child = subprocess.Popen(cmd, env=env, cwd=r/'repos/RoboActualizer',
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            status(phase=phase, child_pid=child.pid, command=cmd, log=str(out/filename))
            code=child.wait()
        if code: raise RuntimeError(f'{phase} exited {code}; see {filename}')
    def train_command(config):
        return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=6',
                str(r/'repos/RoboActualizer/scripts/train.py'),'--config-path',str(out),
                '--config-name',config,f'hydra.run.dir={out}/hydra/{config}']
    try:
        status(phase='waiting_for_cache_and_preflight')
        while True:
            cache=json.loads((out/'cache_build/full_cache_status.json').read_text())
            if cache['phase']=='failed':raise RuntimeError(f'Cache failed: {cache.get("error")}')
            preflight=out/'preflight/results.json'
            launch=json.loads((out/'preflight_launch.json').read_text())
            proc=Path(f'/proc/{launch["pid"]}/stat')
            if not preflight.exists() and (not proc.exists() or proc.read_text().split(') ',1)[1][0]=='Z'):
                raise RuntimeError('Real-batch preflight failed; inspect preflight.log')
            if cache['phase']=='complete' and preflight.exists():break
            time.sleep(10)
        assert json.loads(preflight.read_text())['status']=='passed'
        cfg=OmegaConf.load(out/'train_config.yaml')
        smoke=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        smoke.output_dir=str(out/'smoke_train'); smoke.max_steps=3; smoke.log_every=1
        smoke.wandb.enabled=False; smoke.num_workers=2
        smoke_dir=Path(smoke.output_dir)
        smoke_dir.mkdir(exist_ok=False); (smoke_dir/'PAUSE_REQUEST').touch()
        OmegaConf.save(smoke,out/'smoke.yaml')
        execute(train_command('smoke'),'six_gpu_pause_smoke','smoke.log')
        first=json.loads((smoke_dir/'checkpoints/state/latest/trainer_state.json').read_text())
        if first['global_step']!=1:raise RuntimeError(f'Expected pause at step 1: {first["global_step"]}')
        (smoke_dir/'PAUSE_REQUEST').unlink()
        smoke.resume=str(smoke_dir/'checkpoints/state/latest')
        OmegaConf.save(smoke,out/'smoke_resume.yaml')
        execute(train_command('smoke_resume'),'six_gpu_resume_smoke','smoke_resume.log')
        final=json.loads((smoke_dir/'checkpoints/state/latest/trainer_state.json').read_text())
        expected_cursor=3*int(smoke.gradient_accumulation_steps)
        if final['global_step']!=3 or final['batch_in_epoch']!=expected_cursor:
            raise RuntimeError(f'Smoke cursor/step not restored: {final}')
        import torch
        optimizer=torch.load(smoke_dir/'checkpoints/state/latest/optimizer.bin',map_location='cpu',weights_only=False)
        optimizer_steps=sorted({int(v['step']) for v in optimizer['state'].values() if 'step' in v})
        if optimizer_steps!=[3]:raise RuntimeError(f'AdamW state did not continue: {optimizer_steps}')
        del optimizer
        (out/'ddp_resume_test.json').write_text(json.dumps(dict(status='passed',
            world_size=6,pause_step=1,resume_step=3,batch_cursor=expected_cursor,optimizer_steps=optimizer_steps),indent=2))
        execute(train_command('train_config'),'training','train.log')
        trained=json.loads((out/'train/checkpoints/state/latest/trainer_state.json').read_text())
        if trained['global_step']<20000:
            status(phase='paused',step=trained['global_step']);return
        # Reuse corrected, matched evaluation workers. No W&B monitor is started.
        evaluation=out/'evaluation_020000';evaluation.mkdir(exist_ok=False)
        source=r/'experiments/action_only/full103233_action20k_instruction_retest_20261021/supervisor.py'
        code=source.read_text().replace("RUN=ROOT/'experiments/action_only/ctp_full_handoff/train_action_only_103233'",
                                      f'RUN=Path({str(out/"train")!r})')
        code=code.replace('encoder_step=103233','encoder_step=11000').replace('gpus=[6,7]','gpus=[0,1]').replace('gpu=6+shard%2','gpu=shard%2')
        code=code.replace('action_head_step=20000','action_head_step=20000,source_action_head_step=20000')
        script=evaluation/'supervisor.py';script.write_text(code)
        execute([sys.executable,str(script)],'evaluating','evaluation.log')
        results={name:json.loads((evaluation/name/'summary.json').read_text()) for name in ['clean','plus']}
        status(phase='completed',results={name:{k:v for k,v in d.items() if k!='records'} for name,d in results.items()},finished=time.time())
    except BaseException as error:
        status(phase='failed',error=repr(error));raise


if __name__=='__main__':main()
