"""Wait for the authorized 15K decision stop, then evaluate its pinned weights."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--smoke-result',type=Path)
    parser.add_argument('--name',default=None)
    args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=True)
    def status(**values):
        f=out/'status.tmp';f.write_text(json.dumps(values,indent=2));f.replace(out/'status.json')
    gpus=[int(g) for g in args.gpus.split(',')]
    if args.checkpoint is None:
        status(state='waiting_for_15000_decision_stop',training_run=str(args.run))
        while True:
            state=json.loads((args.run/'status.json').read_text())
            if state['status'] in ('decision_required','paused','completed'):
                break
            if state['status']=='failed':raise RuntimeError('Training failed before evaluation')
            time.sleep(10)
    source=(args.checkpoint or args.run/'checkpoints/latest').resolve()
    metadata=json.loads((source/'progress.json').read_text())
    step=metadata['progress']['step']
    pinned=args.run/'preserved'/f'eval-step-{step:06d}'
    if not pinned.exists():shutil.copytree(source,pinned,copy_function=os.link)
    cfg=json.loads((out/'smoke_policy.json').read_text());cfg['checkpoint']=str(pinned)
    (out/'policy.json').write_text(json.dumps(cfg,indent=2))
    # Do not displace any existing GPU process, including another user's jobs.
    while args.checkpoint is None:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(a):int(b) for a,b in (line.split(',') for line in lines.strip().splitlines())}
        if all(memory[g]<512 for g in gpus):break
        status(state='waiting_for_free_gpus',gpus=gpus,checkpoint=str(pinned),step=step)
        time.sleep(10)
    if not (args.smoke_result or out/'smoke_result.json').exists():
        raise RuntimeError('Actual-weight adapter smoke has not passed')
    import wandb
    run=wandb.init(mode='disabled', project='eb_vjepa',entity='zjc010100000-xi-an-jiaotong-university-',
        name=args.name or f'elaf-v2-step{step}-libero200-plus280',job_type='evaluation',
        dir=str(out),config=dict(policy=cfg,training_step=step,gpus=gpus,episodes=dict(clean=200,plus=280),
        execute_steps=8,settle_steps=10,protocol='same exploratory manifests and canonical-language protocol as corrected Dream baseline'))
    (out/'run_identity.json').write_text(json.dumps(dict(wandb_run_id=run.id,checkpoint=str(pinned),step=step),indent=2))
    env=dict(os.environ,PYTHONPATH=str(args.root),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    results={}
    try:
        clean=json.loads((args.root/'experiments/elaf/evaluation_manifests/clean.json').read_text())
        smoke=[clean[0],clean[50]]
        (out/'sim_smoke_manifest.json').write_text(json.dumps(smoke,indent=2))
        for name,benchmark,manifest,gpus,nenv,batch in [
            ('sim_smoke','clean',out/'sim_smoke_manifest.json',str(gpus[0]),2,2),
            ('clean','clean',args.root/'experiments/elaf/evaluation_manifests/clean.json',args.gpus,24,8),
            ('plus','plus',args.root/'experiments/elaf/evaluation_manifests/plus.json',args.gpus,24,8)]:
            command=[sys.executable,'-m','elaf.evaluation.launch','--root',str(args.root),
                '--policy-config',str(out/'policy.json'),'--manifest',str(manifest),
                '--benchmark',benchmark,'--gpus',gpus,'--environments',str(nenv),
                '--batch-size',str(batch),'--batch-wait-ms','5','--execute-steps','8','--output',str(out/name)]
            (out/f'{name}_command.json').write_text(json.dumps(command,indent=2))
            with (out/f'{name}.log').open('w') as stream:
                child=subprocess.Popen(command,cwd=args.root,env=env,stdout=stream,stderr=subprocess.STDOUT)
                status(state='running',phase=name,child_pid=child.pid,step=step,wandb_run_id=run.id)
                if child.wait()!=0:raise RuntimeError(f'{name} failed; inspect evaluator logs')
            summary=json.loads((out/name/'summary.json').read_text());results[name]=summary
            if name!='sim_smoke':
                run.log({f'eval/{name}/success_rate':summary['successes']/summary['episodes'],
                         f'eval/{name}/episodes':summary['episodes'],f'eval/{name}/successes':summary['successes']})
            (out/'summary.json').write_text(json.dumps(results,indent=2))
        status(state='completed',step=step,wandb_run_id=run.id)
    except BaseException as error:
        status(state='failed',error=str(error),step=step,wandb_run_id=run.id)
        run.finish(exit_code=1);raise
    run.finish()


if __name__=='__main__':main()
