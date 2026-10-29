"""Matched paraphrase representation audit; fixed scenes and fixed flow states."""
import json,re,hashlib,runpy,collections
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from elaf.evaluation.dream_jepa import DreamJEPAPolicy
from elaf.evaluation.policy import dream_eval_image,initial_noise
from elaf.evaluation.official_dream import norm_key
R=Path('/data/users/jiachangzhang/eb_vjepa');O=R/'experiments/instruction_shift_140k_20261026';O.mkdir(exist_ok=True)
template="A video recorded from a robot's point of view executing the following instruction: {task}"
entries=[x for x in json.loads((R/'experiments/elaf/evaluation_manifests/plus.json').read_text()) if x['category']=='Language Instructions']
clean=runpy.run_path('/data/users/jiachangzhang/DNFT4RP/upstream/libero/libero/libero/benchmark/libero_suite_task_map.py')['libero_task_map']
pairs=[]
for x in entries:
 matches=[n for n in clean[x['suite']] if x['name'].startswith(n+'_language_')];assert len(matches)==1
 canonical=re.sub(r'^.*SCENE\d+_','',matches[0]).replace('_',' ')
 pairs.append(dict(group=matches[0],canonical=canonical,paraphrase=x['language'],suite=x['suite'],task_id=x['task_id']))
(O/'pairs.json').write_text(json.dumps(pairs,indent=2))
texts=list(dict.fromkeys(t for p in pairs for t in [p['canonical'],p['paraphrase']]))
cache_dirs=[R/'experiments/action_only/text_cache',R/'experiments/action_only/libero_10k_egl_200/text_cache',R/'experiments/action_only/libero_20k_plus_280_shared_run/text_cache',R/'experiments/grid11k_small_scratch_20261022/text_cache_umt5']
um={};missing=[]
for text in texts:
 prompt=template.format(task=text);name=hashlib.sha256(prompt.encode()).hexdigest()+'.t5_len128.wan22ti2v5b.pt'
 f=next((d/name for d in cache_dirs if (d/name).exists()),None)
 if f is None:missing.append(text);continue
 z=torch.load(f,map_location='cpu',weights_only=True);mask=z['mask'].bool();um[text]=z['context'][mask].float().mean(0)
(O/'umt5_cache_coverage.json').write_text(json.dumps(dict(total=len(texts),missing=missing)))
print('UMT5 cache',len(um),'missing',len(missing),flush=True)
# Never silently drop unmatched prompts from the paired comparison.
if missing:raise RuntimeError('Missing cached uMT5 instructions: '+repr(missing))
torch.save(um,O/'umt5_pooled.pt')
cfg=json.loads((R/'experiments/dream_jepa_eval_140k_20261026/policy.json').read_text());cfg.update(language_subgoals=False,device='cuda:0')
p=DreamJEPAPolicy(cfg);base=p.model.get_base_model();tok=p.processor.tokenizer
video=R/'experiments/dream_jepa_failures_100k_20261025/videos/libero_object_task7_trial0'
frames=imageio.mimread(video.with_suffix('.mp4'),memtest=False);trace=[json.loads(l) for l in video.with_suffix('.jsonl').read_text().splitlines()]
features={};velocities={};spans={}
@torch.inference_mode()
def one_scene(idx):
 history=[dict(image=np.ascontiguousarray(frames[max(0,k)][:,:256]),wrist_image=np.ascontiguousarray(frames[max(0,k)][:,256:])) for k in range(idx-3,idx+1)]
 clips=np.stack([np.stack([h[v] for h in history]) for v in ['image','wrist_image']])
 enc=p.encoder.encode_videos(torch.from_numpy(clips).pin_memory(),pad_batch=True);primary,wrist=enc[:1].to(torch.bfloat16),enc[1:]
 state=np.asarray(trace[max(1,idx)]['proprio']);key=norm_key('libero_object',p.stats,p.mapping);st=p.stats[key]['proprio'];lo=np.asarray(st['q01']);hi=np.asarray(st['q99']);mask=np.asarray(st.get('mask',[True]*8))
 state=torch.tensor(np.clip(np.where(mask,2*(state-lo)/(hi-lo+1e-8)-1,state),-1,1)[None],device=p.device,dtype=torch.bfloat16)
 noiseobs=[dict(noise_seed=239)];a=initial_noise(noiseobs,(8,7),p.device,torch.float32);future=initial_noise(noiseobs,(16,1024),p.device,torch.float32,stream=1)
 for text in texts:
  prompt='What action should the robot take to '+text+'?'
  chat=[dict(role='user',content=[dict(type='image'),dict(type='image'),dict(type='text',text=prompt)])]
  rendered=p.processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
  inputs=p.processor(text=[rendered],images=[dream_eval_image(history[-1][v],p.center_crop) for v in ['image','wrist_image']],return_tensors='pt')
  inputs={k:v.to(p.device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
  ids0=inputs.pop('input_ids');am=inputs.pop('attention_mask')
  # Find instruction span using token subsequence, preserving boundary-space tokenization.
  match=None
  for variant in [text,' '+text]:
   needle=tok.encode(variant,add_special_tokens=False);hay=ids0[0].tolist()
   matches=[j for j in range(len(hay)-len(needle)+1) if hay[j:j+len(needle)]==needle]
   if len(matches)==1:match=(matches[0],matches[0]+len(needle));break
  if match is None:raise ValueError('No exact instruction token span: '+text)
  start,end=match;spans[text]=dict(start=start,end=end,decoded=tok.decode(ids0[0,start:end]))
  ids,att=base._prepare_input_for_action_prediction(ids0,am,7,8);actionmask=torch.zeros_like(ids,dtype=torch.bool);actionmask[:,-58:-2]=True
  embedding=base.get_input_embeddings()(ids0)[0,start:end].float().mean(0).cpu()
  for tau in [0.,.5]:
   t=torch.tensor([tau],device=p.device)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    out=p.model(input_ids=ids,attention_mask=att,noisy_action_features=p.noisy(a.to(torch.bfloat16).reshape(1,56,1)),all_actions_mask=actionmask,timestep_embeddings=p.head.time_encoder(t.to(torch.bfloat16)).unsqueeze(1).to(torch.bfloat16),proprio=state,proprio_projector=p.proprio,jepa_primary=primary,jepa_wrist=wrist,noisy_future=future,future_time=t,future_dropped=torch.zeros(1,device=p.device,dtype=torch.bool),use_cache=False,num_logits_to_keep=1,return_dict=True,**inputs)
    z=out.hidden_states[0,start:end].float().mean(0).cpu();ah=out.hidden_states[:,:-1][actionmask[:,1:]].reshape(1,56,-1);v=p.head.predict_velocity(ah).float().cpu().flatten()
   label=f'scene{idx}_time{tau}';features.setdefault(label,{})[text]=z;velocities.setdefault(label,{})[text]=v
  features.setdefault('input_embedding',{})[text]=embedding
  print('DONE',idx,text,flush=True)
for idx in [40,80]:one_scene(idx)
torch.save(dict(features=features,velocities=velocities),O/'dream_features.pt');(O/'token_spans.json').write_text(json.dumps(spans,indent=2))
def metrics(vectors):
 # Coordinate standardization estimated only from canonical task vectors.
 groups=list(dict.fromkeys(p['canonical'] for p in pairs));C=torch.stack([vectors[t] for t in groups]).double();var=C.var(0,unbiased=False);floor=var.mean()*1e-3;scale=var.clamp_min(floor).sqrt();center=C.mean(0)
 cn=torch.nn.functional.normalize(C,dim=-1);between=torch.pdist(cn).square().mean();between_std=torch.pdist((C-center)/scale).square().mean()/C.shape[1]
 details=[]
 for pair in pairs:
  x=vectors[pair['canonical']].double();y=vectors[pair['paraphrase']].double();xn=torch.nn.functional.normalize(x,dim=0);yn=torch.nn.functional.normalize(y,dim=0)
  details.append(dict(group=pair['group'],cosine=float(xn@yn),unit_squared_distance=float((xn-yn).square().sum()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-12)),standardized_mse=float(((x-y)/scale).square().mean()),nearest_canonical_correct=bool((cn@yn).argmax()==groups.index(pair['canonical']))))
 within_var=[]
 for g in groups:
  vs=[vectors[g]]+[vectors[q['paraphrase']] for q in pairs if q['canonical']==g];Z=torch.stack(vs).double();within_var.append(float(Z.var(0,unbiased=False).mean()/var.mean()))
 return dict(n_pairs=len(pairs),n_tasks=len(groups),mean_cosine=float(np.mean([x['cosine'] for x in details])),mean_relative_l2=float(np.mean([x['relative_l2'] for x in details])),within_to_between_unit_distance=float(np.mean([x['unit_squared_distance'] for x in details])/between),standardized_shift_to_between=float(np.mean([x['standardized_mse'] for x in details])/between_std),within_variance_to_between_variance=float(np.mean(within_var)),retrieval_accuracy=float(np.mean([x['nearest_canonical_correct'] for x in details])),details=details)
result={'umt5_valid_token_mean':metrics(um)}
for k,v in features.items():result['dream_'+k]=metrics(v)
for k,v in velocities.items():result['action_velocity_'+k]=metrics(v)
(O/'results.json').write_text(json.dumps(result,indent=2));print('COMPLETE',flush=True)
