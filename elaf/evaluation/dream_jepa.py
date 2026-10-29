"""Actual dual-view Dream-JEPA checkpoint, with seeded joint Euler sampling."""
from pathlib import Path
import json
import re
import numpy as np
import torch
from elaf.evaluation.official_dream import OfficialDreamPolicy, norm_key
from elaf.evaluation.policy import validate_observations, dream_eval_image, initial_noise, _result, sha256


class DreamJEPAPolicy(OfficialDreamPolicy):
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        cfg = dict(cfg, retain_peft=True, attention_implementation='sdpa',
                   vision_attention_implementation='flash_attention_2')
        super().__init__(cfg)
        from dreamvla.jepa_tokens import DualViewJEPATokens
        from dreamvla.jepa_runtime import FrozenGrid11k
        from prismatic.models.action_heads import MLPResNet, SinusoidalPositionalEncoding
        base = self.model.get_base_model()
        dim = base.config.hidden_size
        base.jepa_tokens = DualViewJEPATokens(dim, (2,24,24)).to(self.device)
        base.jepa_time = SinusoidalPositionalEncoding(dim).to(self.device)
        base.jepa_future_head = MLPResNet(2,dim,dim,1024).to(self.device,dtype=torch.bfloat16)
        path = Path(cfg['checkpoint'])/'training_rank_0.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        modules = dict(vla=self.model, action_head=self.head, proprio_projector=self.proprio,
                       noisy_action_projector=self.noisy)
        with torch.no_grad():
            for name, saved in payload['modules'].items():
                if name not in modules:
                    raise ValueError(f'Unrecognized saved module {name}')
                params = dict(modules[name].named_parameters())
                for key, value in saved.items():
                    if key not in params or params[key].shape != value.shape:
                        raise ValueError(f'Incompatible saved tensor {name}.{key}')
                    params[key].copy_(value)
                for key, value in payload['buffers'][name].items():
                    dict(modules[name].named_buffers())[key].copy_(value)
        expected = {k for k in dict(self.model.named_parameters()) if '.jepa_' in k}
        if not expected or not expected.issubset(payload['modules']['vla']):
            raise ValueError('Checkpoint missing JEPA trainable parameters')
        step = payload['progress']['completed_updates']
        del payload
        self.encoder = FrozenGrid11k(cfg['root'], cfg['encoder_checkpoint'], self.device, batch_size=4)
        self.model.eval().requires_grad_(False)
        self.qk_probe_path = self.cfg.get('qk_probe_path')
        self._qk_probe_done = False
        self._qk_probe_handles = []
        self._qk_step = {}
        if self.qk_probe_path:
            # Projected Q/K RMS probe.  This avoids changing the production
            # SDPA path and records the scale of action queries and JEPA/text/
            # image keys at representative transformer depths.
            layers = self.model.get_base_model().model.layers
            for layer_index in sorted(set([0, len(layers)//4, len(layers)//2,
                                           max(0, len(layers)-4), len(layers)-1])):
                layer = layers[layer_index]
                for kind, module in (('q', layer.self_attn.q_proj), ('k', layer.self_attn.k_proj)):
                    self._qk_probe_handles.append(module.register_forward_hook(
                        lambda _m, _inp, out, li=layer_index, k=kind: self._qk_hook(li, k, out)))
        self.language_helper = None
        if cfg.get("language_subgoals", False):
            from elaf.evaluation.language_subgoals import LanguageSubgoals
            self.language_helper = LanguageSubgoals(self, cfg)
        self.identity.update(type='dream_jepa_dualview',policy_family='Dream-JEPA dual-view joint flow',
            step=step,history_length=4,current_jepa_tokens=32,future_tokens=16,future_enabled=True,
            jepa_ablation=self.cfg.get('jepa_ablation','normal'),
            instruction_variant=self.cfg.get('instruction_variant','normal'),
            future_target='primary absolute fixed spatial pool',joint_flow_steps=self.steps,
            checkpoint_full_state_sha256=sha256(path),encoder_sha256=sha256(cfg['encoder_checkpoint']),
            jepa_preprocessing='four consecutive frames, repeat first, bilinear antialias384, ImageNet norm',
            noise='per_observation_seed separate action/future streams',lora_merged=False)

    def _qk_hook(self, layer_index, kind, output):
        self._qk_step[(layer_index, kind)] = output.detach().float()

    @staticmethod
    def _rms(x):
        return float(x.square().mean().sqrt().item()) if x.numel() else None

    def _record_qk(self, layout):
        if not self.qk_probe_path or self._qk_probe_done:
            return
        rows = []
        for (layer, kind), value in sorted(self._qk_step.items()):
            # Hooks see [B, sequence, heads*head_dim] before rotary embedding.
            groups = {
                'action': layout['action'], 'instruction_or_special': layout['instruction'],
                'image': layout['image'], 'current_jepa': layout['current'],
                'future_jepa': layout['future'], 'all': torch.arange(value.shape[1], device=value.device),
            }
            row = {'layer': layer, 'projection': kind}
            for name, positions in groups.items():
                pos = positions if torch.is_tensor(positions) else torch.as_tensor(positions, device=value.device)
                pos = pos.to(value.device)
                row[f'{name}_rms'] = self._rms(value[:, pos, :])
            rows.append(row)
        Path(self.qk_probe_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.qk_probe_path).write_text(json.dumps({'sequence_layout': layout, 'rows': rows}, indent=2))
        self._qk_probe_done = True

    def _instruction_variant(self, text):
        variant = self.cfg.get('instruction_variant', 'normal')
        if variant == 'normal':
            return text
        if variant == 'swap_stove_cabinet':
            # Simultaneous replacement prevents the second substitution from
            # undoing the first one.
            return re.sub(r'\b(stove|cabinet)\b',
                          lambda m: '__CABINET_TMP__' if m.group(1) == 'stove' else '__STOVE_TMP__',
                          text, flags=re.IGNORECASE).replace('__CABINET_TMP__', 'cabinet').replace('__STOVE_TMP__', 'stove')
        if variant == 'drop_function_words':
            return re.sub(r'\b(the|a|an|in|on|to|and|of|at|from|with)\b', '', text,
                          flags=re.IGNORECASE).replace('  ', ' ').strip()
        if variant == 'mask_content_words':
            return re.sub(r'\b(stove|cabinet|bowl|plate|drawer|basket|milk|cheese|bottle|object)\b',
                          'thing', text, flags=re.IGNORECASE)
        raise ValueError(f'Unknown instruction_variant: {variant}')

    @torch.inference_mode()
    def predict(self, observations):
        validate_observations(observations)
        texts=[];images=[];states=[];keys=[];clips=[]
        supplements = [self.language_helper.get(obs) if self.language_helper else "" for obs in observations]
        for obs, supplement in zip(observations, supplements):
            key=norm_key(obs.get('suite'),self.stats,self.mapping);keys.append(key)
            images.extend([dream_eval_image(obs['image'],self.center_crop),dream_eval_image(obs['wrist_image'],self.center_crop)])
            instruction = self._instruction_variant(obs['instruction'])
            prompt=f"What action should the robot take to {instruction}?"
            if supplement:
                prompt += "\nAuxiliary task explanation and subgoals: " + supplement + "\nFollow the original instruction above."
            conversation=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':prompt}]}]
            texts.append(self.processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True))
            st=self.stats[key]['proprio'];lo=np.asarray(st['q01']);hi=np.asarray(st['q99']);mask=np.asarray(st.get('mask',[True]*8))
            states.append(np.clip(np.where(mask,2*(np.asarray(obs['proprio'])-lo)/(hi-lo+1e-8)-1,obs['proprio']),-1,1))
            history=obs['history']
            if not history:raise ValueError('Every-step camera history required')
            indices=[max(0,len(history)-4+i) for i in range(4)]
            for view in ['image','wrist_image']:
                clips.append(np.stack([history[i][view] for i in indices]))
        features=self.encoder.encode_videos(torch.from_numpy(np.stack(clips)).pin_memory(),pad_batch=True)
        # Training primary cache stores BF16 bits; online wrist output is FP32.
        primary,wrist=features[0::2].to(torch.bfloat16),features[1::2]
        # Counterfactual JEPA probes.  These preserve the feature distribution
        # while breaking its correspondence with the current observation.
        # They are evaluation-only and are selected in policy.json.
        ablation = self.cfg.get('jepa_ablation', 'normal')
        if ablation == 'swap_primary':
            primary = torch.roll(primary, shifts=1, dims=0)
        elif ablation == 'swap_wrist':
            wrist = torch.roll(wrist, shifts=1, dims=0)
        elif ablation == 'swap_both':
            primary = torch.roll(primary, shifts=1, dims=0)
            wrist = torch.roll(wrist, shifts=1, dims=0)
        elif ablation == 'zero_primary':
            primary = torch.zeros_like(primary)
        elif ablation == 'zero_wrist':
            wrist = torch.zeros_like(wrist)
        elif ablation != 'normal':
            raise ValueError(f'Unknown jepa_ablation: {ablation}')
        self.processor.tokenizer.padding_side='left'
        inputs=self.processor(text=texts,images=images,padding=True,return_tensors='pt')
        inputs={k:v.to(self.device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
        base=self.model.get_base_model()
        ids,mask=base._prepare_input_for_action_prediction(inputs.pop('input_ids'),inputs.pop('attention_mask'),7,8)
        actionmask=torch.zeros_like(ids,dtype=torch.bool);actionmask[:,-58:-2]=True
        if not torch.all(actionmask.sum(1)==56):raise ValueError('Unexpected action-token layout')
        states=torch.as_tensor(np.stack(states),device=self.device,dtype=torch.bfloat16)
        if self.qk_probe_path and not self._qk_probe_done:
            base_ids = ids[0]
            prefix = 2  # proprio token + flow-time token
            base_len = ids.shape[1]
            image_id = int(base.config.image_token_id)
            image = torch.where(base_ids == image_id)[0] + prefix
            action = torch.where(actionmask[0])[0] + prefix
            excluded = torch.zeros(base_len, dtype=torch.bool, device=ids.device)
            excluded[torch.where(actionmask[0])[0]] = True
            excluded[torch.where(base_ids == image_id)[0]] = True
            instruction = torch.where(~excluded)[0] + prefix
            current = torch.arange(prefix + base_len, prefix + base_len + 32, device=ids.device)
            future = torch.arange(prefix + base_len + 33, prefix + base_len + 49, device=ids.device)
            self._qk_layout = {'prefix': prefix, 'base_length': base_len,
                               'action': action.tolist(), 'instruction': instruction.tolist(),
                               'image': image.tolist(), 'current': current.tolist(), 'future': future.tolist()}
        actions=initial_noise(observations,(8,7),self.device,torch.float32)
        future=initial_noise(observations,(16,1024),self.device,torch.float32,stream=1)
        for step in range(self.steps):
            t=torch.full((len(observations),),step/self.steps,device=self.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=self.model(input_ids=ids,attention_mask=mask,
                    noisy_action_features=self.noisy(actions.to(torch.bfloat16).reshape(len(observations),56,1)),
                    all_actions_mask=actionmask,timestep_embeddings=self.head.time_encoder(t.to(torch.bfloat16)).unsqueeze(1).to(torch.bfloat16),
                    proprio=states,proprio_projector=self.proprio,jepa_primary=primary,jepa_wrist=wrist,
                    noisy_future=future,future_time=t,future_dropped=torch.zeros(len(observations),device=self.device,dtype=torch.bool),
                    use_cache=False,num_logits_to_keep=1,return_dict=True,**inputs)
                hidden=output.hidden_states[:,:-1][actionmask[:,1:]].reshape(len(observations),56,-1)
                velocity=self.head.predict_velocity(hidden)
            actions=actions+velocity.float()/self.steps
            future=future+output.future_velocity.float()/self.steps
            inputs['inputs_embeds']=output.inputs_embeds
            inputs.pop('pixel_values',None);inputs.pop('image_grid_thw',None)
            if self.qk_probe_path and not self._qk_probe_done:
                self._record_qk(self._qk_layout)
        raw=[]
        for values,key in zip(actions,keys):
            st=self.stats[key]['action'];lo=torch.as_tensor(st['q01'],device=self.device);hi=torch.as_tensor(st['q99'],device=self.device);mask=torch.as_tensor(st.get('mask',[True]*7),device=self.device,dtype=torch.bool)
            raw.append(torch.where(mask,.5*(values+1)*(hi-lo+1e-8)+lo,values))
        return _result(torch.stack(raw),len(observations))
