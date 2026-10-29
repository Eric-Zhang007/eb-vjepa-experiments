"""Memory-mapped frozen raw features and native Dream two-camera conditions."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from elaf.v0_data import make_temporal_pair
from elaf.v0_cache import ENCODER_SHA, sha256


class CachedV0Dataset(Dataset):
    def __init__(self,manifest_path,processor_path=None,*,processor=None,allow_incomplete=False,image_aug=False):
        self.image_aug=image_aug
        self.path=Path(manifest_path);self.root=self.path.parent
        self.manifest=json.loads(self.path.read_text())
        if not allow_incomplete and not self.manifest.get('complete'):
            raise ValueError('Full feature cache is not complete')
        if self.manifest['encoder_sha256']!=ENCODER_SHA:raise ValueError('Wrong frozen encoder')
        if sha256(self.root/self.manifest['stats_path'])!=self.manifest['stats_sha256']:
            raise ValueError('Normalization statistics checksum mismatch')
        self.episodes=self.manifest['episodes'];self.samples=[];self.suite_indices={}
        for i,ep in enumerate(self.episodes):
            for t in range(ep['num_actions']):
                self.suite_indices.setdefault(ep['suite'],[]).append(len(self.samples));self.samples.append((i,t))
        self.maps=OrderedDict();self.processor=processor
        if self.processor is None and processor_path is not None:
            from transformers import AutoProcessor
            self.processor=AutoProcessor.from_pretrained(processor_path,trust_remote_code=True,local_files_only=True,use_fast=False)
            self.processor.tokenizer.padding_side='left'
        self.pad_token_id=self.processor.tokenizer.pad_token_id if self.processor is not None else 0
        self._tf_ready=False

    def __len__(self):return len(self.samples)

    def _arrays(self,i):
        if i not in self.maps:
            ep=self.episodes[i]
            self.maps[i]={k:np.load(self.root/ep['path']/(k+'.npy'),mmap_mode='r')
                for k in ('features','actions','proprio','images','wrist_images')}
            arrays=self.maps[i]
            expected=self.manifest.get('feature_shape',list(arrays['features'].shape[1:]))
            if arrays['features'].shape!=(ep['length'],*expected) or arrays['features'].dtype!=np.uint16:
                raise ValueError(f'Invalid raw feature array for {ep["episode_id"]}')
            if arrays['actions'].shape!=(ep['num_actions'],7) or arrays['proprio'].shape!=(ep['length'],8):
                raise ValueError(f'Invalid action/state array for {ep["episode_id"]}')
            while len(self.maps)>16:self.maps.popitem(last=False)
        self.maps.move_to_end(i);return self.maps[i]

    def __getitem__(self,index):
        return self.get_item(index)

    def get_item(self,index,*,augmentation_generator=None):
        i,t=self.samples[index];ep=self.episodes[i];a=self._arrays(i)
        pair=make_temporal_pair(ep['episode_id'],t,ep['length'],ep['num_actions'])
        result=dict(current_features=torch.from_numpy(a['features'][t].copy()).view(torch.bfloat16),
            future_features=torch.from_numpy(a['features'][pair['future_end']].copy()).view(torch.bfloat16),
            actions=torch.from_numpy(a['actions'][pair['action_indices']].copy()),
            proprio=torch.from_numpy(a['proprio'][t].copy()),
            action_valid=torch.tensor(pair['action_valid']),future_valid=torch.tensor(pair['future_valid']),
            sample_id=f'{ep["episode_id"]}:{t}',suite=ep['suite'],temporal_pair=pair)
        if self.processor is not None:
            from PIL import Image
            import tensorflow as tf
            if not self._tf_ready:
                tf.config.set_visible_devices([],'GPU')
                tf.config.threading.set_inter_op_parallelism_threads(2)
                tf.config.threading.set_intra_op_parallelism_threads(2)
                self._tf_ready=True
            def native_resize(image):
                resized=tf.image.resize(image,(224,224),method='lanczos3',antialias=True)
                resized=tf.cast(tf.clip_by_value(tf.round(resized),0,255),tf.uint8)
                if self.image_aug:
                    import dlimp as dl
                    # Torch RNG is saved per rank with the optimizer checkpoint.
                    seed=torch.randint(0,2147483647,(2,),generator=augmentation_generator).numpy()
                    resized=dl.transforms.augment_image(resized,seed=tf.constant(seed,dtype=tf.int32),
                        random_resized_crop=dict(scale=[0.9,0.9],ratio=[1.0,1.0]),
                        random_brightness=[0.2],random_contrast=[0.8,1.2],
                        random_saturation=[0.8,1.2],random_hue=[0.05],
                        augment_order=['random_resized_crop','random_brightness','random_contrast',
                                       'random_saturation','random_hue'])
                return Image.fromarray(resized.numpy())
            images=[native_resize(a[key][t]) for key in ('images','wrist_images')]
            conversation=[{'role':'user','content':[{'type':'image'},{'type':'image'},
                {'type':'text','text':f'What action should the robot take to {ep["instruction"]}?'}]}]
            text=self.processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
            inputs=self.processor(text=[text],images=images,padding=True,return_tensors='pt')
            for key in ('input_ids','attention_mask'):result[key]=inputs[key][0]
            for key in ('pixel_values','image_grid_thw'):result[key]=inputs[key]
        return result


def collate_v0(samples,pad_token_id=0):
    result={k:torch.stack([s[k] for s in samples]) for k in
        ('current_features','future_features','actions','proprio','action_valid','future_valid')}
    for key in ('sample_id','suite','temporal_pair'):result[key]=[s[key] for s in samples]
    if 'input_ids' in samples[0]:
        length=max(len(s['input_ids']) for s in samples)
        ids=torch.full((len(samples),length),pad_token_id,dtype=torch.long);mask=torch.zeros_like(ids)
        for i,s in enumerate(samples):
            n=len(s['input_ids']);ids[i,-n:]=s['input_ids'];mask[i,-n:]=s['attention_mask']
        result.update(input_ids=ids,attention_mask=mask,
            pixel_values=torch.cat([s['pixel_values'] for s in samples]),
            image_grid_thw=torch.cat([s['image_grid_thw'] for s in samples]))
    return result
