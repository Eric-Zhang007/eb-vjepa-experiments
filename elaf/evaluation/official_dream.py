"""Official Dream-VLA checkpoint loader and batch-safe native flow inference."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from elaf.evaluation.policy import sha256, dream_eval_image, validate_observations, initial_noise, _result


def strip_ddp(state):
    return {k.removeprefix('module.'):v for k,v in state.items()}


def norm_key(suite, stats, mapping):
    if not suite or suite not in mapping:
        raise ValueError(f'Explicit suite normalization mapping required: {suite!r}')
    key=mapping[suite]
    if key not in stats:raise ValueError(f'Checkpoint has no normalization statistics for {key}')
    return key


class OfficialDreamPolicy:
    def __init__(self,cfg):
        from transformers import AutoProcessor
        from peft import PeftModel
        repo=Path(cfg['repo']);sys.path.insert(0,str(repo/'vla'))
        from dreamvla.configuration_dreamvl import DreamVLAConfig
        from dreamvla.modeling_dreamvl import DreamVLAForActionPrediction
        # Upstream selects action chunk constants from argv at import time.
        argv=sys.argv[:]
        try:
            sys.argv.append('libero')
            from prismatic.models.action_heads import FlowMatchingActionHead
            from prismatic.models.projectors import ProprioProjector,NoisyActionProjector
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK,ACTION_DIM,PROPRIO_DIM
        finally:
            sys.argv[:]=argv
        if (NUM_ACTIONS_CHUNK,ACTION_DIM,PROPRIO_DIM)!=(8,7,8):
            raise ValueError('Official LIBERO constants must be loaded before model modules')
        self.device=torch.device(cfg.get('device','cuda:0'));self.steps=int(cfg.get('num_steps',4))
        if self.steps<1:raise ValueError('positive num_steps required')
        checkpoint=Path(cfg['checkpoint']).resolve();self.stats=json.loads((checkpoint/'dataset_statistics.json').read_text())
        self.mapping=cfg['suite_norm_map'];self.center_crop=cfg.get('center_crop',True)
        for suite in self.mapping:norm_key(suite,self.stats,self.mapping)
        merged=bool(list(checkpoint.glob('model*.safetensors')))
        source=checkpoint if merged else Path(cfg['base_checkpoint'])
        model_cfg=DreamVLAConfig.from_pretrained(checkpoint,n_action_bins=256,norm_stats=self.stats)
        model_cfg._attn_implementation=cfg.get('attention_implementation','flash_attention_2')
        model_cfg.vision_config._attn_implementation=cfg.get('vision_attention_implementation',model_cfg._attn_implementation)
        self.model=DreamVLAForActionPrediction.from_pretrained(source,config=model_cfg,torch_dtype=torch.bfloat16,local_files_only=True)
        if not merged:
            self.model=PeftModel.from_pretrained(self.model,checkpoint/'lora_adapter')
            if not cfg.get('retain_peft',False):self.model=self.model.merge_and_unload()
        self.model.to(self.device).eval();dim=self.model.config.hidden_size
        self.head=FlowMatchingActionHead(input_dim=dim,hidden_dim=dim,action_dim=7)
        self.proprio=ProprioProjector(llm_dim=dim,proprio_dim=8)
        self.noisy=NoisyActionProjector(llm_dim=dim)
        hashes={}
        for name,module in [('action_head',self.head),('proprio_projector',self.proprio),('noisy_action_projector',self.noisy)]:
            files=list(checkpoint.glob(name+'--*.pt'))
            if len(files)!=1:raise ValueError(f'Expected unique official {name} checkpoint; got {files}')
            module.load_state_dict(strip_ddp(torch.load(files[0],map_location='cpu',weights_only=True)),strict=True)
            module.to(self.device,dtype=torch.bfloat16).eval();hashes[files[0].name]=sha256(files[0])
        self.processor=AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True,use_fast=False)
        weightfiles=list(checkpoint.glob('model*.safetensors')) if merged else list((checkpoint/'lora_adapter').glob('*.safetensors'))
        for file in weightfiles:hashes[str(file.relative_to(checkpoint))]=sha256(file)
        self.identity=dict(type='dream_vla_official',policy_family='Official Dream-VLA',checkpoint=str(checkpoint),checkpoint_sha256=hashes,stats_sha256=sha256(checkpoint/'dataset_statistics.json'),suite_norm_map=self.mapping,action_dim=7,proprio_dim=8,action_horizon=8,recommended_execute_steps=8,history_length=1,action_space='libero_delta_ee',action_coordinates='raw_dataset_gripper_0close_1open',num_steps=self.steps,batch_adaptation='native forward, shifted56 action positions, native Euler flow; no batch1 reshape',center_crop=self.center_crop)
    def fingerprint(self):return self.identity.copy()
    @torch.inference_mode()
    def predict(self,observations):
        validate_observations(observations)
        texts=[];images=[];states=[];keys=[]
        for obs in observations:
            key=norm_key(obs.get('suite'),self.stats,self.mapping);keys.append(key)
            images.extend([dream_eval_image(obs['image'],self.center_crop),dream_eval_image(obs['wrist_image'],self.center_crop)])
            conversation=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':f"What action should the robot take to {obs['instruction']}?"}]}]
            texts.append(self.processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True))
            st=self.stats[key]['proprio'];low=np.asarray(st['q01']);high=np.asarray(st['q99']);mask=np.asarray(st.get('mask',[True]*8))
            states.append(np.clip(np.where(mask,2*(np.asarray(obs['proprio'])-low)/(high-low+1e-8)-1,obs['proprio']),-1,1))
        # Left padding preserves native suffix-based shifted hidden selection for variable prompts.
        self.processor.tokenizer.padding_side='left'
        inputs=self.processor(text=texts,images=images,padding=True,return_tensors='pt')
        inputs={k:v.to(self.device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
        ids,mask=self.model._prepare_input_for_action_prediction(inputs.pop('input_ids'),inputs.pop('attention_mask'),7,8)
        actionmask=torch.zeros_like(ids,dtype=torch.bool);actionmask[:,-58:-2]=True
        states=torch.as_tensor(np.stack(states),device=self.device,dtype=torch.bfloat16)
        actions=initial_noise(observations,(8,7),self.device,torch.bfloat16)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for step in range(self.steps):
                t=torch.full((len(observations),),step/self.steps,device=self.device,dtype=torch.bfloat16)
                output=self.model(input_ids=ids,attention_mask=mask,noisy_action_features=self.noisy(actions.reshape(len(observations),56,1)),all_actions_mask=actionmask,timestep_embeddings=self.head.time_encoder(t).to(actions.dtype).unsqueeze(1),proprio=states,proprio_projector=self.proprio,return_dict=True,**inputs)
                hidden=output.hidden_states[:,:-1,:][:,-58:-2,:]
                actions+=self.head.predict_velocity(hidden)/self.steps
        raw=[]
        for values,key in zip(actions.float(),keys):
            st=self.stats[key]['action'];lo=torch.as_tensor(st['q01'],device=self.device);hi=torch.as_tensor(st['q99'],device=self.device);mask=torch.as_tensor(st.get('mask',[True]*7),device=self.device,dtype=torch.bool)
            raw.append(torch.where(mask,.5*(values+1)*(hi-lo+1e-8)+lo,values))
        return _result(torch.stack(raw),len(observations))
