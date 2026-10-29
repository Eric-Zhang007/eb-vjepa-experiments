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
    while not (out/'text_download_status.json').exists():
        time.sleep(10)
    receipt = json.loads((out/'text_download_status.json').read_text())
    if receipt['status'] != 'complete':
        raise RuntimeError(receipt)
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
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(receipt['path'], local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(receipt['path'], local_files_only=True,
                                           torch_dtype=torch.bfloat16).eval().requires_grad_(False).cuda()
    cache = out/'text_cache'; cache.mkdir(exist_ok=True)
    prompts = sorted(prompts)
    with torch.inference_mode():
        for start in range(0, len(prompts), 16):
            batch = prompts[start:start+16]
            inputs = tokenizer(batch, padding='max_length', truncation=True, max_length=128, return_tensors='pt').to('cuda')
            mask = inputs['attention_mask'].bool()
            context = encoder(**inputs).last_hidden_state
            context[~mask] = 0
            assert context.shape[1:] == (128,4096) and torch.isfinite(context).all()
            for i, prompt in enumerate(batch):
                name = hashlib.sha256(prompt.encode()).hexdigest()+'.t5_len128.t5v11xxl.pt'
                torch.save(dict(context=context[i].cpu(),mask=mask[i].cpu(),prompt=prompt,
                                encoder='google/t5-v1_1-xxl'),cache/name)
            print('encoded',min(start+16,len(prompts)),len(prompts),flush=True)
    (out/'text_cache_status.json').write_text(json.dumps(dict(status='complete',prompts=len(prompts),
        encoder='google/t5-v1_1-xxl',context_len=128),indent=2))


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
