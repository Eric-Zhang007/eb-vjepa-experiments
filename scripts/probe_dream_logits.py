import json
import sys
from pathlib import Path
import torch

ROOT = Path('/data/users/jiachangzhang/eb_vjepa')
REPO = ROOT / 'experiments/dream_jepa_dualview_20261023/infra_source'
sys.path.insert(0, str(REPO / 'vla'))
from transformers import AutoProcessor
from dreamvla.configuration_dreamvl import DreamVLAConfig
from dreamvla.modeling_dreamvl import DreamVLAForActionPrediction
from dreamvla.generation_utils import DreamVLGenerationConfig

base = ROOT / 'checkpoints/dream_vla/Dream-VLA-7B'
cfg = DreamVLAConfig.from_pretrained(base, local_files_only=True, n_action_bins=256, norm_stats={})
cfg._attn_implementation = 'sdpa'
model = DreamVLAForActionPrediction.from_pretrained(base, config=cfg, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
proc = AutoProcessor.from_pretrained(base, trust_remote_code=True, local_files_only=True, use_fast=False)
tok = proc.tokenizer
chat = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe the current robot task and give three short subgoals for placing an object in a container.'}]}]
text = proc.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[text], return_tensors='pt')
ids, attention = inputs.input_ids.cuda(), inputs.attention_mask.cuda()
mask = tok.convert_tokens_to_ids('<|mask|>')
action = [v for k, v in tok.get_vocab().items() if k.startswith('<|action_')]
special = [v for k, v in tok.get_vocab().items() if k.startswith('<|')]
capture = {}

def hook(i, x, logits):
    if i == 0:
        z = logits[0, -1].float()
        p = torch.softmax(z, dim=-1)
        top = torch.topk(p, 100).indices
        capture.update(
            vocab=len(p), action_count=len(action), special_count=len(special),
            action_mass=float(p[action].sum()), special_mass=float(p[special].sum()),
            mask_prob=float(p[mask]),
            top=[(tok.convert_ids_to_tokens(int(j)), float(p[j])) for j in top[:30]],
            top_non_action=[(tok.convert_ids_to_tokens(int(j)), float(p[j])) for j in top if int(j) not in action][:30],
        )
    return logits

generation = DreamVLGenerationConfig(max_new_tokens=32, steps=16, temperature=0.0, alg='origin',
    mask_token_id=mask, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, return_dict_in_generate=True)
with torch.inference_mode():
    output = model.diffusion_generate(ids, generation_config=generation,
        attention_mask=attention, generation_logits_hook_func=hook)
capture['decoded'] = tok.decode(output.sequences[0, ids.shape[1]:], skip_special_tokens=False)
(ROOT / 'experiments/dream_vl_download/logit_probe.json').write_text(json.dumps(capture, indent=2))
print(json.dumps(capture, indent=2))
