#!/usr/bin/env python3
"""Portable Linux handoff CLI. Setup never starts GPU work implicitly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent
OLD = '/data/users/jiachangzhang/eb_vjepa'
DESTINATIONS = {
    'dream_vla_baseline': 'repos/Dream-VLX',
    'dream_jepa_final': 'experiments/dream_jepa_dualview_20261023/infra_source',
    'roboactualizer': 'repos/RoboActualizer', 'vjepa2': 'repos/vjepa2',
    'libero': 'repos/LIBERO', 'libero_plus': 'repos/LIBERO-plus-official',
    'libero_pro': 'repos/LIBERO-PRO-official', 'vjepa_policy': 'repos/VJEPA-Policy',
}
MODELS = {
    'grid11k': ('Isla8899/eb-vjepa-grid11k', 'experiments/ctp_grid20k_20261008'),
    'baseline': ('Isla8899/dream-vla-libero-mix4-200k', 'checkpoints/baseline200k'),
    'dream-jepa': ('Isla8899/dream-jepa-dualview-156500', 'checkpoints/dream_jepa156500'),
    'base': ('Dream-org/Dream-VLA-7B', 'checkpoints/dream_vla/Dream-VLA-7B'),
}
TEXT = {'.py','.sh','.json','.yaml','.yml','.txt','.md','.toml','.ini','.cfg'}

def run(cmd, cwd=None, env=None):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)

def load_config(path):
    cfg=json.loads(Path(path).read_text())
    for key in ['workspace', 'data_root']:
        if not Path(cfg[key]).is_absolute():raise ValueError(key+' must be absolute')
    return cfg

def mapping(cfg):
    w=Path(cfg['workspace'])
    return {
        '/data/users/jiachangzhang/DNFT4RP/upstream/vla_adapter/datasets/rlds': cfg['data_root'],
        '/data/users/jiachangzhang/DNFT4RP/upstream/libero': (w/'repos/LIBERO').as_posix(),
        '/data/users/jiachangzhang/DNFT4RP/.libero-config': (w/'experiments/action_only/clean_config').as_posix(),
        '/home/jiachangzhang/dream_jepa_dualview_20261023': (w/'runs/dream_jepa').as_posix(),
        OLD: w.as_posix(),
    }

def relocate(text, cfg):
    for before, after in sorted(mapping(cfg).items(),key=lambda pair:len(pair[0]),reverse=True):
        text=text.replace(before,after)
    return text

def copy_sources(cfg):
    w=Path(cfg['workspace']);w.mkdir(parents=True,exist_ok=True)
    for name in ['ctp_full','ctp_v1','elaf','dream_baseline','scripts','tests','experiments']:
        source=REPO/name
        if not source.exists():continue
        for p in source.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            q=w/name/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix in TEXT:q.write_text(relocate(p.read_text(encoding='utf-8'),cfg),encoding='utf-8')
            else:shutil.copy2(p,q)
    for name in ['cache/tmp','runs','checkpoints','handoff','experiments/dream_jepa_dualview_20261023']:
        (w/name).mkdir(parents=True,exist_ok=True)
    # Config destinations remain separate from source code, preserving historical snapshots.
    for p in (REPO/'configs').rglob('*'):
        if p.is_file():
            q=w/'experiments'/p.relative_to(REPO/'configs');q.parent.mkdir(parents=True,exist_ok=True)
            q.write_text(relocate(p.read_text(encoding='utf-8'),cfg),encoding='utf-8')

def materialize_one(spec, cfg):
    """Pinned checkout + checked patch + untracked overlay; idempotent receipt."""
    w=Path(cfg['workspace']);target=w/DESTINATIONS[spec['name']]
    overlay=REPO/'upstream'/spec['name'];patch=overlay/'changes.patch'
    signature=hashlib.sha256(json.dumps(dict(spec=spec,mapping=mapping(cfg)),sort_keys=True).encode())
    if overlay.exists():
        for p in sorted(overlay.rglob('*')):
            if p.is_file():signature.update(str(p.relative_to(overlay)).encode());signature.update(p.read_bytes())
    receipt=w/'handoff'/('source_'+spec['name']+'.json');digest=signature.hexdigest()
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['signature']==digest and (target/'.git').exists():return
        raise RuntimeError(f'Source recipe changed: {target}. Use a new workspace; no reset/overwrite is automatic.')
    if target.exists() and any(target.iterdir()):raise RuntimeError('Unmanaged nonempty source directory: '+str(target))
    target.mkdir(parents=True,exist_ok=True)
    run(['git','init',str(target)])
    run(['git','remote','add','origin',spec['origin']],target)
    run(['git','fetch','--depth','1','origin',spec['commit']],target)
    run(['git','-c','core.autocrlf=false','checkout','--detach','FETCH_HEAD'],target)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
    if actual!=spec['commit']:raise RuntimeError('Upstream commit mismatch')
    if patch.exists():
        run(['git','apply','--check',str(patch)],target)
        run(['git','apply','--whitespace=nowarn',str(patch)],target)
    if (overlay/'overlay').exists():shutil.copytree(overlay/'overlay',target,dirs_exist_ok=True)
    rewrites=[]
    for p in target.rglob('*'):
        if '.git' in p.parts or not p.is_file() or p.suffix not in TEXT or p.stat().st_size>8*1024**2:continue
        try:text=p.read_text(encoding='utf-8')
        except UnicodeError:continue
        changed=relocate(text,cfg)
        # Evaluation uses encode_videos only, and does not need a training feature cache.
        if spec['name']=='dream_jepa_final' and p.name=='jepa_runtime.py':
            old="self.index = json.loads((root / 'experiments/dream_jepa_dualview_20261023/episode_index.json').read_text())"
            new="index_path = root / 'experiments/dream_jepa_dualview_20261023/episode_index.json'\n        self.index = json.loads(index_path.read_text()) if index_path.exists() else None"
            if old not in changed:raise RuntimeError('Unexpected FrozenGrid11k source; inspect before adaptation')
            changed=changed.replace(old,new).replace('started = time.monotonic()','if self.index is None:\n            raise RuntimeError("Training episode index is missing; run handoff.py cache first")\n        started = time.monotonic()',1)
        if changed!=text:
            rewrites.append(dict(path=str(p.relative_to(target)),before=hashlib.sha256(p.read_bytes()).hexdigest(),after=hashlib.sha256(changed.encode()).hexdigest()))
            p.write_text(changed,encoding='utf-8')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(dict(signature=digest,commit=actual,rewrites=rewrites),indent=2))

def sources(cfg, names=None):
    copy_sources(cfg)
    specs=json.loads((REPO/'provenance/upstream_repositories.json').read_text())
    wanted=set(names or ['dream_vla_baseline','dream_jepa_final','roboactualizer','vjepa2','libero','libero_plus'])
    if wanted-set(DESTINATIONS):raise ValueError('Unknown upstream name')
    for spec in specs:
        if spec['name'] in wanted:materialize_one(spec,cfg)
    w=Path(cfg['workspace'])
    for kind,repo in [('clean','LIBERO'),('plus','LIBERO-plus-official')]:
        target=w/'experiments/action_only'/f'{kind}_config';target.mkdir(parents=True,exist_ok=True)
        base=w/'repos'/repo/'libero/libero'
        paths={k:str(base/v) for k,v in [('benchmark_root',''),('bddl_files','bddl_files'),('init_states','init_files'),('assets','assets')]}
        if kind=='plus':paths['assets']=str(w/'checkpoints/benchmark_assets/libero_plus/assets')
        paths['datasets']=cfg['data_root']
        (target/'config.yaml').write_text('\n'.join(k+': '+json.dumps(v) for k,v in paths.items())+'\n')

def python_path(cfg):return Path(cfg['workspace'])/'envs/dream_vla_official/bin/python'

def setup_env(cfg):
    if sys.platform!='linux':raise RuntimeError('Environment setup requires Linux; use WSL only for non-GPU validation')
    w=Path(cfg['workspace']);py=python_path(cfg)
    if not shutil.which('uv'):raise RuntimeError('Install uv or ask cluster Codex to provision Python 3.10, then retry')
    if not py.exists():run(['uv','venv','--python','3.10','--seed',str(py.parent.parent)])
    run(['uv','pip','install','--python',py,'--index-url','https://download.pytorch.org/whl/cu121','torch==2.5.1+cu121','torchvision==0.20.1+cu121'])
    run(['uv','pip','install','--python',py,'-r',REPO/'environments/dream-portable.txt'])
    run([py,'-m','pip','install','--no-build-isolation','flash-attn==2.5.5'],env={**os.environ,'MAX_JOBS':os.environ.get('MAX_JOBS','4')})
    run([py,'-m','pip','install','--no-deps','git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46'])
    run([py,'-m','pip','install','--no-deps','-e',str(w/'repos/LIBERO')])
    run([py,'-m','pip','check'])

def assets(cfg, selected):
    py=python_path(cfg);w=Path(cfg['workspace'])
    for name in selected:
        if name=='rlds':
            run([py,w/'scripts/download_official_libero_rlds.py','--output',cfg['data_root']]);continue
        if name=='plus':
            run([py,w/'scripts/prepare_plus_assets.py','--root',w])
            run([py,w/'scripts/extract_plus_assets.py'],cwd=w);continue
        repo,rel=MODELS[name]
        if name=='base':run([py,w/'scripts/download_dream_vla.py','--output',w/rel]);continue
        script='from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1],local_dir=sys.argv[2])'
        run([py,'-c',script,repo,w/rel])

def environment(cfg,gpus=None):
    w=Path(cfg['workspace'])
    result={**os.environ,'PYTHONPATH':str(w),'TMPDIR':str(w/'cache/tmp'),'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','WANDB_MODE':'disabled'}
    if gpus is not None:result['CUDA_VISIBLE_DEVICES']=gpus
    return result

def policy_config(cfg,model):
    w=Path(cfg['workspace'])
    policy=json.loads((REPO/'elaf/evaluation/configs/dream_vla_official_mixed.json').read_text())
    policy.update(repo=str(w/DESTINATIONS['dream_vla_baseline' if model=='baseline' else 'dream_jepa_final']),checkpoint=str(w/MODELS[model][1]),base_checkpoint=str(w/MODELS['base'][1]),root=str(w),num_steps=4)
    if model=='dream-jepa':policy.update(type='dream_jepa_dualview',encoder_checkpoint=str(w/MODELS['grid11k'][1]/'embodied_vjepa_encoder_011000.pt'),language_subgoals=False)
    return policy

def evaluate(cfg,args):
    w=Path(cfg['workspace']);py=python_path(cfg)
    # Generate fresh manifests so stored path hashes match the relocated files.
    manifest=w/'handoff'/f'{args.benchmark}_manifest.json'
    code='import json,sys; from elaf.evaluation.libero_env import make_manifest; from pathlib import Path; Path(sys.argv[2]).write_text(json.dumps(make_manifest(sys.argv[1],root=Path(sys.argv[3]))))'
    run([py,'-c',code,args.benchmark,manifest,w],cwd=w,env=environment(cfg))
    if args.episodes:
        values=json.loads(manifest.read_text())[:args.episodes];manifest.write_text(json.dumps(values))
    p=w/'handoff'/f'{args.model}_policy.json';p.write_text(json.dumps(policy_config(cfg,args.model),indent=2))
    output=Path(args.output).resolve() if args.output else w/'runs'/f'eval_{args.model}_{args.benchmark}_{time.time_ns()}'
    run([py,'-m','elaf.evaluation.launch','--root',w,'--policy-config',p,'--manifest',manifest,'--benchmark',args.benchmark,'--gpus',args.gpus,'--environments',args.environments,'--batch-size',args.batch_size,'--execute-steps','8','--output',output],cwd=w,env=environment(cfg))

def cache(cfg,args):
    w=Path(cfg['workspace']);py=python_path(cfg);out=w/'data/elaf_v0_grid11k_raw'
    stats=w/MODELS['baseline'][1]/'dataset_statistics.json';encoder=w/MODELS['grid11k'][1]/'embodied_vjepa_encoder_011000.pt'
    base=[py,'-m','elaf.v0_cache']
    run(base+['export','--root',w,'--output',out,'--data-root',cfg['data_root'],'--stats',stats],cwd=w,env=environment(cfg,''))
    gpu_ids=args.gpus.split(',');jobs=[]
    try:
        for shard,gpu in enumerate(gpu_ids):
            cmd=base+['worker','--root',str(w),'--output',str(out),'--checkpoint',str(encoder),'--shard',str(shard),'--shards',str(len(gpu_ids)),'--batch-size',str(args.batch_size)]
            jobs.append(subprocess.Popen(list(map(str,cmd)),cwd=w,env=environment(cfg,gpu)))
        for job in jobs:
            if job.wait()!=0:raise RuntimeError('Cache worker failed; completed shards remain resumable')
    finally:
        for job in jobs:
            if job.poll() is None:job.terminate()
    run(base+['finalize','--output',out],cwd=w,env=environment(cfg,''))
    run([py,w/'scripts/build_dream_jepa_episode_index.py'],cwd=w,env=environment(cfg,''))

def train(cfg,args):
    w=Path(cfg['workspace']);py=python_path(cfg)
    repo=w/DESTINATIONS['dream_vla_baseline' if args.model=='baseline' else 'dream_jepa_final']
    env=environment(cfg,args.gpus);env['PYTHONPATH']=str(repo/'vla')+os.pathsep+str(w)
    env['DREAM_JEPA_EPISODE_INDEX']=str(w/'experiments/dream_jepa_dualview_20261023/episode_index.json')
    env['WANDB_MODE']=args.wandb_mode
    n=len(args.gpus.split(','))
    cmd=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(n),repo/'vla/vla-scripts/finetune_dream.py','--vla_path',w/MODELS['base'][1],'--data_root_dir',cfg['data_root'],'--dataset_name','libero_mix4','--run_root_dir',w/'runs/fresh','--use_l1_regression','False','--use_diffusion','False','--use_flow_matching','True','--use_film','False','--num_images_in_input','2','--use_proprio','True','--batch_size','8','--learning_rate','1e-4','--num_steps_before_decay','100000','--max_steps',str(args.steps),'--save_freq','5000','--save_latest_checkpoint_only','True','--image_aug','True','--lora_rank','32','--wandb_project','eb_vjepa','--run_id_note','portable_'+args.model]
    if args.model=='dream-jepa':cmd+=['--jepa_joint','True','--jepa_root',w,'--jepa_checkpoint',w/MODELS['grid11k'][1]/'embodied_vjepa_encoder_011000.pt','--gradient_checkpointing','False','--merge_lora_during_training','False']
    if args.execute:run(cmd,cwd=repo/'vla',env=env)
    else:print(json.dumps(dict(command=list(map(str,cmd)),cwd=str(repo/'vla'),global_batch=8*n,note='Fresh training only; add --execute to run. This does not resume step156500.'),indent=2))

def doctor(cfg,args):
    w=Path(cfg['workspace']);py=python_path(cfg);checks={}
    required={'python':py,'baseline_config':w/MODELS['baseline'][1]/'config.json','baseline_weights':w/MODELS['baseline'][1]/'model.safetensors.index.json','grid11k':w/MODELS['grid11k'][1]/'embodied_vjepa_encoder_011000.pt','dream_jepa_state':w/MODELS['dream-jepa'][1]/'training_rank_0.pt','base_config':w/MODELS['base'][1]/'config.json','clean_assets':w/'repos/LIBERO/libero/libero/assets','plus_assets':w/'checkpoints/benchmark_assets/libero_plus/assets'}
    checks.update({k:dict(path=str(p),exists=p.exists()) for k,p in required.items()})
    if args.gpu and py.exists():
        run([py,REPO/'tools/gpu_check.py'],env=environment(cfg))
    (w/'handoff').mkdir(parents=True,exist_ok=True)
    (w/'handoff/doctor.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(REPO/'cluster.local.json'))
    sub=p.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init');init.add_argument('--workspace',required=True);init.add_argument('--data-root',required=True)
    src=sub.add_parser('sources');src.add_argument('--only',nargs='+',choices=list(DESTINATIONS))
    sub.add_parser('env')
    a=sub.add_parser('assets');a.add_argument('names',nargs='+',choices=list(MODELS)+['rlds','plus'])
    d=sub.add_parser('doctor');d.add_argument('--gpu',action='store_true')
    e=sub.add_parser('eval');e.add_argument('--model',choices=['baseline','dream-jepa'],required=True);e.add_argument('--benchmark',choices=['clean','plus'],default='clean');e.add_argument('--gpus',default='0');e.add_argument('--environments',type=int,default=8);e.add_argument('--batch-size',type=int,default=1);e.add_argument('--episodes',type=int,default=0);e.add_argument('--output')
    c=sub.add_parser('cache');c.add_argument('--gpus',required=True);c.add_argument('--batch-size',type=int,default=4)
    t=sub.add_parser('train');t.add_argument('--model',choices=['baseline','dream-jepa'],required=True);t.add_argument('--gpus',default='0,1,2,3,4,5');t.add_argument('--steps',type=int,default=200000);t.add_argument('--execute',action='store_true');t.add_argument('--wandb-mode',choices=['disabled','offline','online'],default='offline')
    args=p.parse_args()
    if args.command=='init':
        cfg=dict(workspace=str(Path(args.workspace).expanduser().resolve()),data_root=str(Path(args.data_root).expanduser().resolve()))
        if Path(cfg['workspace'])==REPO or REPO.is_relative_to(Path(cfg['workspace'])):raise ValueError('Use a separate workspace, not the code repository or its parent')
        target=Path(args.config)
        if target.exists() and json.loads(target.read_text())!=cfg:raise RuntimeError('Config exists with different paths; choose a new --config')
        target.write_text(json.dumps(cfg,indent=2));print(target);return
    cfg=load_config(args.config)
    if args.command=='sources':sources(cfg,args.only)
    elif args.command=='env':setup_env(cfg)
    elif args.command=='assets':assets(cfg,args.names)
    elif args.command=='doctor':doctor(cfg,args)
    elif args.command=='eval':evaluate(cfg,args)
    elif args.command=='cache':cache(cfg,args)
    elif args.command=='train':train(cfg,args)

if __name__=='__main__':main()
