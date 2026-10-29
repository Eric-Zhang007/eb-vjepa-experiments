"""Sequential clean200/Plus280 using two batched GPU policy services; no W&B."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--environments',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    out=args.output
    config=json.loads((out/'policy.json').read_text())
    if config['type']!='dream_jepa_dualview':raise ValueError('Wrong policy family')
    environment=dict(os.environ,PYTHONPATH=str(args.root),OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        WANDB_MODE='disabled',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2')
    started=time.time();summaries={}
    def status(**kw):
        tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(pid=os.getpid(),started=started,**kw),indent=2));tmp.replace(out/'status.json')
    try:
        for benchmark,count in [('clean',200),('plus',280)]:
            manifest=args.root/'experiments/elaf/evaluation_manifests'/f'{benchmark}.json'
            if len(json.loads(manifest.read_text()))!=count:raise ValueError('Unexpected manifest size')
            cmd=[sys.executable,'-m','elaf.evaluation.launch','--root',str(args.root),
                '--policy-config',str(out/'policy.json'),'--manifest',str(manifest),
                '--benchmark',benchmark,'--gpus','6,7','--environments',str(args.environments),
                '--batch-size',str(args.batch_size),'--batch-wait-ms','5','--execute-steps','8',
                '--output',str(out/benchmark)]
            (out/f'{benchmark}_command.json').write_text(json.dumps(cmd,indent=2))
            with (out/f'{benchmark}.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd=args.root,env=environment,stdout=log,stderr=subprocess.STDOUT)
                status(state='running',benchmark=benchmark,child_pid=child.pid)
                if child.wait()!=0:raise RuntimeError(f'{benchmark} evaluation failed; inspect GPU logs')
            summaries[benchmark]=json.loads((out/benchmark/'summary.json').read_text())
        (out/'summary.json').write_text(json.dumps(summaries,indent=2))
        status(state='complete',seconds=time.time()-started)
    except BaseException as exc:
        status(state='failed',error=repr(exc),seconds=time.time()-started)
        raise


if __name__=='__main__':main()
