"""Full official RLDS -> episode-local frozen grid11k raw feature cache.

Checklist: preserve all source rows; hash source and output; no temporal crossing;
verify raw encoder shape; atomic episode commits; publish completion last.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

SUITES = ('spatial', 'object', 'goal', '10')
ENCODER_SHA = '468d7af3a5bdae9fb2205718ab0f86a13f5395afcc53700500439d993baf987a'
PREPROCESS = dict(camera='image', window=4, stride=1, resize=[384,384],
    interpolation='torch_bilinear_antialias', mean=[.485,.456,.406], std=[.229,.224,.225],
    augmentation='none_shared_deterministic', stage='encoder_raw', pooling='none',
    storage='bfloat16_bits_uint16', temporal_padding='repeat_episode_first_frame')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w') as f:
        json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def normalization(x,stat):
    x=np.asarray(x,dtype=np.float32)
    lo=np.asarray(stat['q01'],dtype=np.float32);hi=np.asarray(stat['q99'],dtype=np.float32)
    mask=np.asarray(stat.get('mask',[True]*x.shape[-1]))
    result=np.where(mask,np.clip(2*(x-lo)/(hi-lo+1e-8)-1,-1,1),x)
    return np.where(np.asarray(stat['min'])==np.asarray(stat['max']),0,result).astype(np.float32)

def export(args):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    stats=json.load(open(args.stats));shutil.copyfile(args.stats,out/'dataset_statistics.json')
    episodes=[];source=[]
    for suite in SUITES:
        directory=Path(args.data_root)/f'libero_{suite}_no_noops'/'1.0.0'
        for p in sorted(directory.iterdir()):
            if p.is_file() and ('.tfrecord-' in p.name or p.suffix=='.json'):
                source.append(dict(path=str(p),size=p.stat().st_size,sha256=sha256(p)))
        builder=tfds.builder_from_directory(str(directory))
        for index,episode in enumerate(builder.as_dataset(split='train',shuffle_files=False)):
            eid=f'{suite}_{index:05d}';dest=out/'episodes'/eid;receipt=dest/'source.json'
            if receipt.exists():
                ep=json.load(open(receipt))
                for name,digest in ep['source_files'].items():
                    if sha256(dest/name)!=digest:raise ValueError(f'Corrupt source {eid}/{name}')
                episodes.append(ep);continue
            rows=list(tfds.as_numpy(episode['steps']));n=len(rows)
            if n<2 or not rows[0]['is_first'] or not rows[-1]['is_last'] or any(r['is_last'] for r in rows[:-1]):
                raise ValueError(f'Invalid episode flags {eid}')
            tasks={r['language_instruction'].decode().strip().lower() for r in rows}
            if len(tasks)!=1 or not next(iter(tasks)):raise ValueError(f'Language mismatch {eid}')
            tmp=dest.with_name(dest.name+'.partial');tmp.mkdir(parents=True,exist_ok=True)
            action=np.stack([r['action'] for r in rows]);action[:,-1]=1-np.clip(action[:,-1],0,1)
            st=stats[f'libero_{suite}_no_noops']
            arrays={'images':np.stack([r['observation']['image'] for r in rows]),
                'wrist_images':np.stack([r['observation']['wrist_image'] for r in rows]),
                'actions':normalization(action,st['action']),
                'proprio':normalization(np.stack([r['observation']['state'] for r in rows]),st['proprio'])}
            for name,array in arrays.items():np.save(tmp/(name+'.npy'),array,allow_pickle=False)
            ep=dict(episode_id=eid,suite=suite,path=f'episodes/{eid}',length=n,num_actions=n,
                instruction=next(iter(tasks)),terminal_action='retained_source_non_dummy_control',
                final_post_action_observation_available=False,
                source_files={p.name:sha256(p) for p in tmp.glob('*.npy')})
            atomic_json(tmp/'source.json',ep);os.replace(tmp,dest);episodes.append(ep)
            print(json.dumps(dict(event='export',episode=eid,frames=n)),flush=True)
    plan=dict(schema_version=2,complete=False,episodes=episodes,source_files=source,
        encoder_sha256=ENCODER_SHA,preprocessing=PREPROCESS,
        preprocessing_sha256=hashlib.sha256(json.dumps(PREPROCESS,sort_keys=True).encode()).hexdigest(),
        feature_shape=[1152,1024],feature_dtype='bfloat16_bits_uint16',
        stats_path='dataset_statistics.json',stats_sha256=sha256(out/'dataset_statistics.json'),
        frame_count=sum(ep['length'] for ep in episodes),
        temporal_contract='t:t+8 controls, observation t+8, retained no_noops row clock; no episode crossing')
    atomic_json(out/'source_manifest.json',plan)
    print(json.dumps(dict(event='export_complete',frames=plan['frame_count'],episodes=len(episodes))),flush=True)

def load_teacher(root,checkpoint,device):
    import torch
    if sha256(checkpoint)!=ENCODER_SHA:raise ValueError('Wrong grid11k checkpoint')
    path=Path(root)/'repos/RoboActualizer/src/Roboactualizer/models/roboactualizer/vjepa.py'
    spec=importlib.util.spec_from_file_location('elaf_vjepa_factory',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.build_vjepa_encoder(ckpt_path=str(checkpoint),model_size='large',image_size=(384,384),
        torch_hub_dir=str(Path(root)/'repos/vjepa2')).eval().requires_grad_(False).to(device)

def encode_windows(teacher,images,ends,device):
    import torch
    import torch.nn.functional as F
    indices=np.maximum(np.asarray(ends)[:,None]+np.arange(-3,1),0)
    clip=torch.from_numpy(np.asarray(images[indices]).copy()).to(device).permute(0,1,4,2,3).float()/255
    b,t,c,h,w=clip.shape
    clip=F.interpolate(clip.reshape(b*t,c,h,w),(384,384),mode='bilinear',align_corners=False,antialias=True)
    mean=clip.new_tensor(PREPROCESS['mean'])[None,:,None,None];std=clip.new_tensor(PREPROCESS['std'])[None,:,None,None]
    clip=((clip-mean)/std).reshape(b,t,c,384,384).permute(0,2,1,3,4)
    with torch.inference_mode(),torch.autocast(device_type=clip.device.type,dtype=torch.bfloat16):
        features=teacher(clip)
    if features.shape!=(b,1152,1024) or not torch.isfinite(features).all():
        raise ValueError(f'Invalid raw encoder output {features.shape}')
    return features.to(dtype=torch.bfloat16,device='cpu').contiguous().view(torch.uint16).numpy()

def worker(args):
    import torch
    torch.set_num_threads(4)
    out=Path(args.output);plan=json.load(open(out/'source_manifest.json'))
    device=args.device
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA encoder unavailable')
    teacher=load_teacher(args.root,args.checkpoint,device)
    start=time.time();count=0;completed=0
    for i,ep in enumerate(plan['episodes']):
        if i%args.shards!=args.shard:continue
        dest=out/ep['path'];receipt=dest/'features.json'
        if receipt.exists():
            old=json.load(open(receipt))
            if old['encoder_sha256']!=ENCODER_SHA or sha256(dest/'features.npy')!=old['sha256']:
                raise ValueError(f'Corrupt completed feature shard {ep["episode_id"]}')
            continue
        tmp=dest/'features.partial.npy';shape=(ep['length'],1152,1024)
        if shutil.disk_usage(out).free<np.prod(shape)*2+30*1024**3:raise RuntimeError('Insufficient cache disk reserve')
        result=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint16,shape=shape)
        images=np.load(dest/'images.npy',mmap_mode='r')
        for t in range(0,ep['length'],args.batch_size):
            ends=list(range(t,min(t+args.batch_size,ep['length'])))
            result[t:t+len(ends)]=encode_windows(teacher,images,ends,device)
        result.flush();del result;os.replace(tmp,dest/'features.npy')
        atomic_json(receipt,dict(encoder_sha256=ENCODER_SHA,preprocessing_sha256=plan['preprocessing_sha256'],
            shape=list(shape),sha256=sha256(dest/'features.npy')))
        count+=ep['length'];elapsed=time.time()-start
        atomic_json(out/f'worker_{args.shard}.json',dict(episode=ep['episode_id'],frames=count,elapsed=elapsed,
            frames_per_second=count/elapsed,pid=os.getpid()))
        print(json.dumps(dict(event='encoded',episode=ep['episode_id'],frames=count,fps=count/elapsed)),flush=True)
        completed+=1
        if args.max_episodes and completed>=args.max_episodes:break

def finalize(args):
    import torch
    out=Path(args.output);plan=json.load(open(out/'source_manifest.json'))
    for ep in plan['episodes']:
        dest=out/ep['path'];r=json.load(open(dest/'features.json'))
        if r['encoder_sha256']!=ENCODER_SHA or r['preprocessing_sha256']!=plan['preprocessing_sha256']:
            raise ValueError('Feature identity mismatch')
        if sha256(dest/'features.npy')!=r['sha256']:raise ValueError('Feature checksum mismatch')
        for name,digest in ep['source_files'].items():
            if sha256(dest/name)!=digest:raise ValueError('Source cache checksum mismatch')
        ep['feature_sha256']=r['sha256']
    rng=np.random.default_rng(239);current=[];future=[];ids=[]
    for suite in SUITES:
        candidates=[(ep,t) for ep in plan['episodes'] if ep['suite']==suite for t in range(ep['length']-8)]
        for index in rng.choice(len(candidates),256,replace=False):
            ep,t=candidates[index];a=np.load(out/ep['path']/'features.npy',mmap_mode='r')
            current.append(torch.from_numpy(a[t].copy()).view(torch.bfloat16))
            future.append(torch.from_numpy(a[t+8].copy()).view(torch.bfloat16));ids.append(f'{ep["episode_id"]}:{t}')
    monitor=dict(schema_version=1,manifest=dict(encoder_checkpoint_sha256=ENCODER_SHA,
        preprocessing_sha256=plan['preprocessing_sha256'],selection_seed=239,suites_equal=True),
        sample_ids=ids,current_features=torch.stack(current),future_features=torch.stack(future))
    torch.save(monitor,out/'monitor.partial.pt');os.replace(out/'monitor.partial.pt',out/'monitor.pt')
    from elaf.monitor import monitor_manifest_sha256
    plan.update(complete=True,monitor_path='monitor.pt',monitor_sha256=sha256(out/'monitor.pt'),
        monitor_manifest_sha256=monitor_manifest_sha256(monitor),
        completed_at=time.time(),all_file_hashes_verified=True)
    atomic_json(out/'manifest.json',plan)
    atomic_json(out/'completion.json',dict(status='complete',manifest_sha256=sha256(out/'manifest.json'),
        frames=plan['frame_count'],episodes=len(plan['episodes']),completed_at=time.time()))
    print('CACHE_COMPLETE',flush=True)

def gpu_snapshot():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    active={line.split(',')[0].strip() for line in processes if ',' in line}
    return {int(parts[0]):(parts[1] not in active and int(parts[2])<512 and int(parts[3])<5)
        for parts in ([x.strip() for x in row.split(',')] for row in rows)}

def supervise(args):
    """Wait for free cards, launch immutable worker processes, finalize once."""
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(out/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (out/'source_manifest.json').exists():
        atomic_json(out/'supervisor.json',dict(status='waiting_for_export',pid=os.getpid()))
        time.sleep(10)
    plan=json.load(open(out/'source_manifest.json'))
    required=sum(ep['length'] for ep in plan['episodes'])*1152*1024*2
    existing=sum((out/ep['path']/'features.npy').stat().st_size for ep in plan['episodes']
        if (out/ep['path']/'features.npy').exists())
    if shutil.disk_usage(out).free<required-existing+30*1024**3:raise RuntimeError('Full cache does not fit disk')
    candidates=[0,1,2,3,4,5]
    from scripts.watch_elaf_v0 import IdleTimer, query_gpus
    timer=IdleTimer(30)
    while True:
        baseline_alive=args.wait_pid is not None and Path(f'/proc/{args.wait_pid}').exists()
        if baseline_alive:
            timer.reset()
            ready=False
        else:
            ready=timer.observe(query_gpus(candidates),time.monotonic())
        atomic_json(out/'supervisor.json',dict(status='waiting_for_baseline' if baseline_alive else 'waiting_for_idle_gpu',
            pid=os.getpid(),wait_pid=args.wait_pid,gpus=candidates,
            idle_seconds=0 if timer.since is None else time.monotonic()-timer.since,workers={},done=[]))
        if ready:break
        time.sleep(5)
    workers={};done=set();idle_since={gpu:timer.since for gpu in candidates}
    python=str(Path(args.root)/'envs/eb_vjepa_pretrain/bin/python')
    if not Path(python).exists():raise FileNotFoundError(python)
    while len(done)<args.shards:
        for shard,(proc,gpu) in list(workers.items()):
            rc=proc.poll()
            if rc is not None:
                if rc!=0:raise RuntimeError(f'Feature worker {shard} failed rc={rc}; inspect worker_{shard}.log')
                done.add(shard);del workers[shard]
        free=gpu_snapshot();now=time.monotonic()
        for gpu in candidates:
            if free.get(gpu):idle_since.setdefault(gpu,now)
            else:idle_since.pop(gpu,None)
        for shard in range(args.shards):
            if shard in workers or shard in done:continue
            available=[g for g in candidates if g in idle_since and now-idle_since[g]>=30
                and g not in [v[1] for v in workers.values()]]
            if not available:break
            gpu=available[0]
            # Recheck immediately before starting; never evict existing processes.
            if not gpu_snapshot().get(gpu):continue
            log=open(out/f'worker_{shard}.log','ab',buffering=0)
            command=[python,'-u','-m','elaf.v0_cache','worker','--root',args.root,'--output',args.output,
                '--checkpoint',args.checkpoint,'--shard',str(shard),'--shards',str(args.shards),
                '--batch-size',str(args.batch_size)]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'}
            proc=subprocess.Popen(command,cwd=args.root,env=env,stdout=log,stderr=subprocess.STDOUT)
            log.close();workers[shard]=(proc,gpu);idle_since.pop(gpu,None)
        atomic_json(out/'supervisor.json',dict(status='encoding' if workers else 'waiting_for_idle_gpu',
            pid=os.getpid(),workers={s:dict(pid=p.pid,gpu=g) for s,(p,g) in workers.items()},done=sorted(done)))
        if len(done)<args.shards:time.sleep(5)
    subprocess.run([python,'-u','-m','elaf.v0_cache','finalize','--output',args.output],cwd=args.root,check=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4'})
    atomic_json(out/'supervisor.json',dict(status='complete',pid=os.getpid()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['export','worker','finalize','supervise'])
    p.add_argument('--root',default='/data/users/jiachangzhang/eb_vjepa');p.add_argument('--output',required=True)
    p.add_argument('--data-root',default='/data/users/jiachangzhang/DNFT4RP/upstream/vla_adapter/datasets/rlds')
    p.add_argument('--stats');p.add_argument('--checkpoint');p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=2);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--max-episodes',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--wait-pid',type=int,help='Wait for this baseline launcher to exit before the 30-second GPU idle timer')
    args=p.parse_args();globals()[args.command](args)

if __name__=='__main__':main()
