"""Encode this experiment's train and evaluation prompts with frozen T5-v1.1-XXL."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5EncoderModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root, out = args.root, args.output
    template = "A video recorded from a robot's point of view executing the following instruction: {task}"
    prompts = set()
    data_root = Path('/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA')
    for f in data_root.glob('libero_*_no_noops_1.0.0_lerobot/meta/tasks.jsonl'):
        for line in f.read_text().splitlines():
            d = json.loads(line); prompts.add(template.format(task=d['task']))
    # Historical corrected evaluation caches contain exact clean/Plus prompts.
    for directory in [root/'experiments/action_only/text_cache',
                      root/'experiments/action_only/libero_10k_egl_200/text_cache',
                      root/'experiments/action_only/libero_20k_plus_280_shared_run/text_cache']:
        for f in directory.glob('*.pt'):
            d = torch.load(f, map_location='cpu', weights_only=True)
            if 'prompt' in d: prompts.add(d['prompt'])
    plus = json.loads((root/'experiments/grid11k_latent_wm_20261021/evaluation_020000/plus/manifest.json').read_text())
    entries = plus if isinstance(plus, list) else plus.get('entries', plus.get('tasks', plus.get('episodes', [])))
    for d in entries: prompts.add(template.format(task=d['language']))
    clean = json.loads((root/'experiments/grid11k_latent_wm_20261021/evaluation_020000/clean/summary.json').read_text())
    for d in clean['records']: prompts.add(template.format(task=d['input_instruction']))
    if len(prompts) < 40: raise RuntimeError('Incomplete instruction coverage')
    import shutil
    cache = out/'text_cache_umt5'; cache.mkdir(exist_ok=True)
    sources = [root/'experiments/action_only/text_cache',root/'experiments/action_only/libero_10k_egl_200/text_cache',root/'experiments/action_only/libero_20k_plus_280_shared_run/text_cache']
    missing=[]
    for prompt in sorted(prompts):
        name=hashlib.sha256(prompt.encode()).hexdigest()+'.t5_len128.wan22ti2v5b.pt'
        candidate=next((d/name for d in sources if (d/name).exists()),None)
        if candidate is None:missing.append((prompt,name));continue
        payload=torch.load(candidate,map_location='cpu',weights_only=True)
        assert payload.get('prompt',prompt)==prompt and payload['context'].shape==(128,4096) and torch.isfinite(payload['context']).all()
        payload['prompt'] = prompt
        torch.save(payload, cache/name)
    print('reused',len(prompts)-len(missing),'missing',len(missing),flush=True)
    if missing:
        from Roboactualizer.models.roboactualizer.runtime import load_text_encoder_components
        encoder,tokenizer=load_text_encoder_components(model_id='Wan-AI/Wan2.2-TI2V-5B',tokenizer_model_id='Wan-AI/Wan2.1-T2V-1.3B',context_len=128,device='cuda',dtype=torch.bfloat16)
        with torch.inference_mode():
            for prompt,name in missing:
                ids,mask=tokenizer([prompt],return_mask=True,add_special_tokens=True);mask=mask.cuda().bool()
                context=encoder(ids.cuda(),mask);context[~mask]=0
                torch.save(dict(context=context[0].cpu(),mask=mask[0].cpu(),prompt=prompt),cache/name)
    (out/'text_cache_status.json').write_text(json.dumps(dict(status='complete',prompts=len(prompts),encoder='Wan umT5-XXL',context_len=128)))



if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import sys
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1])
            (output/'text_cache_status.json').write_text(json.dumps(
                dict(status='failed', error=repr(exc)), indent=2))
        raise
