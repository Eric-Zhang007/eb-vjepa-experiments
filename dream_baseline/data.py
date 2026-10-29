"""Dream-VLA LIBERO training data, native processor, per-suite train statistics.

CLI creates a full metadata manifest. Optional --image-cache predecodes both views
to uint8 memory-mappable arrays for shuffled training without repeated AV1 decode.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance

SUITES=('spatial','object','goal','10')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def read_rows(path):
    import pyarrow.parquet as pq
    table=pq.read_table(path,columns=['action','observation.state'])
    return {key:np.asarray(table[key].to_pylist(),dtype=np.float32) for key in table.column_names}


def normalize(array,stats):
    low=np.asarray(stats['q01']); high=np.asarray(stats['q99']); mask=np.asarray(stats['mask'])
    value=np.where(mask,np.clip(2*(array-low)/(high-low+1e-8)-1,-1,1),array)
    return np.where(np.asarray(stats['min'])==np.asarray(stats['max']),0,value).astype(np.float32)


def decode(path):
    import av
    with av.open(path) as stream:
        return np.stack([frame.to_ndarray(format='rgb24') for frame in stream.decode(video=0)])


def cache_episode(ep,cache):
    result=dict(ep)
    for view in ('image','wrist_image'):
        path=Path(cache)/ep['suite']/f"{ep['episode_index']:06d}_{view}.npy"
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            frames=decode(ep[view])
            if len(frames)!=ep['length']: raise ValueError(f'Video length mismatch {ep[view]}')
            temporary=path.with_suffix('.partial')
            with temporary.open('wb') as stream: np.save(stream,frames,allow_pickle=False)
            os.replace(temporary,path)
        frames=np.load(path,mmap_mode='r')
        if frames.shape!=(ep['length'],256,256,3) or frames.dtype!=np.uint8:
            raise ValueError(f'Invalid image cache {path}')
        result[view+'_cache']=str(path)
        result[view+'_cache_sha256']=sha256(path)
    return result


def prepare(data_root,output,image_cache=None,workers=16):
    data_root=Path(data_root); output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    episodes=[]; statistics={}; sources=[]
    for suite in SUITES:
        directory=data_root/f'libero_{suite}_no_noops_1.0.0_lerobot'
        info=json.loads((directory/'meta/info.json').read_text())
        start,end=map(int,info['splits']['train'].split(':'))
        rows=[]
        for line in (directory/'meta/episodes.jsonl').read_text().splitlines():
            ep=json.loads(line); index=ep['episode_index']
            if not start<=index<end or (suite=='goal' and index==82): continue
            params=dict(episode_index=index,episode_chunk=index//info['chunks_size'])
            parquet=directory/info['data_path'].format(**params)
            row=read_rows(parquet)
            if len(row['action'])!=ep['length']: raise ValueError(f'Length mismatch {parquet}')
            if not np.isin(row['action'][:,-1],[0,1]).all(): raise ValueError('Expected converted LeRobot gripper open01')
            rows.append(row)
            entry=dict(suite=suite,episode_index=index,length=ep['length'],task=ep['tasks'][0],fps=info['fps'],
                       parquet=str(parquet),parquet_sha256=sha256(parquet))
            for view in ('image','wrist_image'):
                video=directory/info['video_path'].format(**params,video_key='observation.images.'+view)
                entry[view]=str(video)
                entry[view+'_bytes']=video.stat().st_size
            episodes.append(entry)
        statistics[suite]={}
        for source,key in [('action','action'),('observation.state','proprio')]:
            array=np.concatenate([row[source] for row in rows]).astype(np.float64)
            statistics[suite][key]={name:fn(array,axis=0).tolist() for name,fn in
                [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]}
            statistics[suite][key].update(q01=np.quantile(array,.01,axis=0).tolist(),q99=np.quantile(array,.99,axis=0).tolist(),
                mask=([True]*6+[False]) if key=='action' else [True]*8)
        statistics[suite]['frames']=sum(len(row['action']) for row in rows)
        for filename in ('info.json','episodes.jsonl','tasks.jsonl'):
            path=directory/'meta'/filename;sources.append(dict(path=str(path),sha256=sha256(path)))
    if image_cache:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            episodes=list(executor.map(partial(cache_episode,cache=str(Path(image_cache).resolve())),episodes))
    stat_path=output/'dataset_statistics.json';stat_path.write_text(json.dumps(statistics,indent=2))
    manifest=dict(schema_version=1,episodes=episodes,statistics_path='dataset_statistics.json',statistics_sha256=sha256(stat_path),
        sources=sources,excluded=['goal:82'],split='train',normalization='per-suite BOUNDS_Q99; action gripper unchanged open01',
        action_horizon=8,tail_policy='exclude last7starts, matching official effective_traj_len=T-7',
        images='stored LeRobot orientation; both views resized224; no additional rotation',
        augmentation='deterministic per seed/epoch/index; crop area0.9 square then brightness additive +/-0.2, contrast/saturation0.8..1.2,hue+/-0.05; PIL differs from official TF interpolation/rounding',
        frame_count=sum(e['length'] for e in episodes),sample_count=sum(max(0,e['length']-7) for e in episodes))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'receipt.json').write_text(json.dumps(dict(manifest_sha256=sha256(output/'manifest.json'),
        episodes=len(episodes),samples=manifest['sample_count'],statistics_sha256=sha256(stat_path)),indent=2))
    return output/'manifest.json'


def augment_image(image,rng):
    image=image.resize((224,224),Image.Resampling.BILINEAR)
    side=round(224*np.sqrt(.9)); x=int(rng.integers(0,225-side));y=int(rng.integers(0,225-side))
    image=image.crop((x,y,x+side,y+side)).resize((224,224),Image.Resampling.BILINEAR)
    image=Image.fromarray(np.clip(np.asarray(image).astype(np.float32)+rng.uniform(-.2,.2)*255,0,255).astype(np.uint8))
    image=ImageEnhance.Contrast(image).enhance(float(rng.uniform(.8,1.2)))
    image=ImageEnhance.Color(image).enhance(float(rng.uniform(.8,1.2)))
    hsv=np.array(image.convert('HSV'));hsv[:,:,0]=(hsv[:,:,0].astype(int)+round(rng.uniform(-.05,.05)*255))%256
    return Image.fromarray(hsv,mode='HSV').convert('RGB')


class DreamLiberoDataset(Dataset):
    def __init__(self,manifest,processor,suite='spatial',augment=True,seed=239,cache_episodes=2,verify_cache=True):
        self.path=Path(manifest);self.manifest=json.loads(self.path.read_text())
        stat_path=self.path.parent/self.manifest['statistics_path']
        if sha256(stat_path)!=self.manifest['statistics_sha256']: raise ValueError('Statistics identity mismatch')
        self.stats=json.loads(stat_path.read_text())[suite]
        self.episodes=[ep for ep in self.manifest['episodes'] if ep['suite']==suite]
        if not self.episodes or cache_episodes<1:raise ValueError('Nonempty suite and positive cache_episodes required')
        if verify_cache:
            for ep in self.episodes:
                for view in ('image','wrist_image'):
                    if view+'_cache' in ep and sha256(ep[view+'_cache'])!=ep[view+'_cache_sha256']:
                        raise ValueError(f'Image cache identity changed: {ep[view+"_cache"]}')
        self.ends=np.cumsum([max(0,ep['length']-7) for ep in self.episodes])
        self.processor=processor; self.augment=augment;self.seed=seed;self.epoch=0;self.cache_episodes=cache_episodes;self.cache=OrderedDict()
        processor.tokenizer.add_tokens([f'<|action_{n}|>' for n in range(256)])
        self.action_ids=[processor.tokenizer.convert_tokens_to_ids(f'<|action_{n}|>') for n in range(256)]
        if self.action_ids!=list(range(len(processor.tokenizer)-256,len(processor.tokenizer))):
            raise ValueError('Official action tokens must occupy final256 tokenizer positions')

    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __len__(self): return int(self.ends[-1]) if len(self.ends) else 0

    def _episode(self,index):
        if index not in self.cache:
            ep=self.episodes[index]
            if sha256(ep['parquet'])!=ep['parquet_sha256']: raise ValueError('Parquet changed')
            row=read_rows(ep['parquet'])
            for view in ('image','wrist_image'):
                row[view]=np.load(ep[view+'_cache'],mmap_mode='r') if view+'_cache' in ep else decode(ep[view])
                if len(row[view])!=ep['length']: raise ValueError('Video length mismatch')
            self.cache[index]=row
            while len(self.cache)>self.cache_episodes:self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        ep_index=int(np.searchsorted(self.ends,index,side='right')); t=int(index-(self.ends[ep_index-1] if ep_index else 0))
        ep=self.episodes[ep_index];row=self._episode(ep_index)
        actions=normalize(row['action'][t:t+8],self.stats['action'])
        proprio=normalize(row['observation.state'][t],self.stats['proprio'])
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,int(index)]))
        images=[]
        for view in ('image','wrist_image'):
            im=Image.fromarray(row[view][t]);images.append(augment_image(im,rng) if self.augment else im.resize((224,224),Image.Resampling.BILINEAR))
        return build_sample(self.processor,images,ep['task'],actions,proprio,dict(suite=ep['suite'],episode_index=ep['episode_index'],frame_index=t))


def build_sample(processor,images,instruction,actions,proprio,provenance=None):
    actions=np.asarray(actions,dtype=np.float32);proprio=np.asarray(proprio,dtype=np.float32)
    if len(images)!=2 or actions.shape!=(8,7) or proprio.shape!=(8,):raise ValueError('Expected two images, action8x7, proprio8')
    if not np.isfinite(actions).all() or not np.isfinite(proprio).all():raise ValueError('Nonfinite action or proprio')
    token_ids=len(processor.tokenizer)-np.digitize(np.clip(actions.reshape(-1),-1,1),np.linspace(-1,1,256))
    action_string=processor.tokenizer.decode(token_ids.tolist())
    conversation=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':f"What action should the robot take to {instruction.lower()}?"}]},
                  {'role':'assistant','content':[{'type':'text','text':action_string}]}]
    text=processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=False)
    batch=processor(text=[text],images=images,padding=True,return_tensors='pt')
    ids=batch['input_ids'][0];labels=ids.clone();labels[:-58]=-100
    mask=(ids>=len(processor.tokenizer)-256)&(ids<len(processor.tokenizer))
    if mask.sum()!=56 or not torch.equal(ids[mask],torch.tensor(token_ids)):
        raise ValueError('Native processor did not preserve56 action tokens')
    return dict(input_ids=ids,labels=labels,action_mask=mask,pixel_values=batch['pixel_values'],image_grid_thw=batch['image_grid_thw'],
        actions=torch.from_numpy(actions),proprio=torch.from_numpy(proprio),action_valid=torch.ones(8,dtype=torch.bool),
        provenance=provenance or {},dataset_name=(provenance or {}).get('suite','unknown'))


def collate_fn(samples,pad_token_id=0):
    from torch.nn.utils.rnn import pad_sequence
    result={key:pad_sequence([s[key] for s in samples],batch_first=True,padding_value=value)
            for key,value in [('input_ids',pad_token_id),('labels',-100),('action_mask',False)]}
    result['attention_mask']=torch.arange(result['input_ids'].shape[1])[None,:]<torch.tensor([len(s['input_ids']) for s in samples])[:,None]
    for key in ('pixel_values','image_grid_thw'):result[key]=torch.cat([s[key] for s in samples])
    for key in ('actions','proprio','action_valid'):result[key]=torch.stack([s[key] for s in samples])
    result['provenance']=[s['provenance'] for s in samples]
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA')
    p.add_argument('--output',required=True);p.add_argument('--image-cache');p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();print(prepare(a.data_root,a.output,a.image_cache,a.workers))
