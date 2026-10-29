"""Once-per-episode native Dream language diffusion, followed by action flow."""
import hashlib
import json
import os
import re
from pathlib import Path
import torch
from elaf.evaluation.policy import dream_eval_image

class LanguageSubgoals:
    def __init__(self, policy, cfg):
        self.policy = policy
        self.cache = {}
        self.prompt_template = cfg.get("language_prompt_template")
        self.suppress_action_tokens = cfg.get("language_suppress_action_tokens", False)
        self.diagnostic_only = cfg.get("language_diagnostic_only", False)
        self.steps = int(cfg.get('language_steps', 16))
        self.tokens = int(cfg.get('language_tokens', 64))
        self.log = Path(cfg['language_log_dir']) / f'generated_{os.getpid()}.jsonl'
        self.log.parent.mkdir(parents=True, exist_ok=True)
        policy.identity['language_subgoals'] = dict(generator='same base with LoRA disabled',
            steps=self.steps, max_new_tokens=self.tokens, frequency='episode start',
            original_instruction_preserved=True, joint_sampling=False)

    @torch.inference_mode()
    def get(self, obs):
        key = obs['episode_key']
        if key in self.cache:
            return self.cache[key]
        p = self.policy
        from dreamvla.generation_utils import DreamVLGenerationConfig
        request = ('Given these robot camera images and the original instruction below, '
            'briefly restate the intended manipulation and give at most three short subgoals. '
            'Preserve the target objects, destination, spatial relations, and operation order. '
            'Do not invent objects or report that actions have already succeeded. '
            'Return only the explanation and subgoals. Original instruction: ' + obs['instruction'])
        if self.prompt_template:
            request = self.prompt_template + '\nOriginal instruction: ' + obs['instruction']
        chat = [{'role':'user','content':[{'type':'image'},{'type':'image'},
            {'type':'text','text':request}]}]
        text = p.processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
        images = [dream_eval_image(obs[k],p.center_crop) for k in ['image','wrist_image']]
        inputs = p.processor(text=[text],images=images,return_tensors='pt')
        inputs = {k:v.to(p.device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
        ids = inputs.pop('input_ids')
        tokenizer = p.processor.tokenizer
        mask = tokenizer.convert_tokens_to_ids('<|mask|>')
        if mask is None or tokenizer.convert_ids_to_tokens(mask) != '<|mask|>':
            raise ValueError('Dream mask token not found')
        config = DreamVLGenerationConfig(max_new_tokens=self.tokens,steps=self.steps,
            temperature=0.0,alg='origin',mask_token_id=mask,
            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True)
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
        extra = {}
        if self.suppress_action_tokens:
            banned = [v for k,v in tokenizer.get_vocab().items() if k.startswith('<|action_')]
            def logits_hook(step, x, logits):
                logits[..., banned] = -torch.inf
                return logits
            extra['generation_logits_hook_func'] = logits_hook
        with torch.random.fork_rng(devices=[p.device]), p.model.disable_adapter(), torch.autocast('cuda',dtype=torch.bfloat16):
            torch.manual_seed(seed)
            output = p.model.get_base_model().diffusion_generate(ids,generation_config=config,**inputs,**extra)
        generated_ids = output.sequences[0,ids.shape[1]:].tolist()
        result = tokenizer.decode(generated_ids,skip_special_tokens=True).strip()
        row = dict(episode_key=key,instruction=obs['instruction'],generated=result,
            generated_token_ids=generated_ids,seed=seed,steps=self.steps)
        with self.log.open('a') as f:
            f.write(json.dumps(row)+'\n')
        if not self.diagnostic_only and (not re.sub(r'<\|.*?\|>', '', result).strip() or '<|action_' in result or '<|mask|>' in result):
            raise ValueError('Invalid language subgoal; inspect language log; action-token output must not be used as text')
        self.cache[key] = result
        return result
