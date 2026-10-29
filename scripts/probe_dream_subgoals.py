"""Standalone stage snapshots; no action rollout and no W&B."""
import argparse,json,os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from elaf.evaluation.dream_jepa import DreamJEPAPolicy
from elaf.evaluation.language_subgoals import LanguageSubgoals

PROMPTS={
'concise': "The two images show the current robot state from an external camera and a wrist camera. Follow the original instruction exactly. Describe what is visibly true now, identify the next unfinished subgoal, and describe the next local gripper motion. Do not repeat a completed subgoal. Do not assume that proximity means a successful grasp. Use three short labeled lines: Observation; Next subgoal; Local action. If a relation is occluded or depth is ambiguous, say so. Do not invent numeric robot coordinates.",
'structured': "You are interpreting a manipulation task at its CURRENT stage, not planning from the beginning. The first image is an external view and the second is a wrist view. Preserve the objects, destination and operation order in the original instruction. Answer with five concise labeled lines: Visible evidence: target/gripper/destination relations actually visible; Current stage: approach, grasp, transport, place, release, close drawer, done, or uncertain; Next subgoal: exactly one unfinished goal; Local action: qualitative gripper movement, whether to open/close/hold, and the visible condition for ending this motion; Uncertainty: what cannot be determined from these images. Never claim a grasp or completion without visible evidence. Avoid left/right robot-frame commands unless the reference frame is clear. Do not provide numerical coordinates."
}

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--suppress-action-tokens',action='store_true');args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((args.root/'experiments/dream_jepa_eval_100k_20261025/policy.json').read_text())
    cfg.update(device='cuda:0',language_subgoals=False)
    policy=DreamJEPAPolicy(cfg)
    source=args.root/'experiments/dream_jepa_failures_100k_20261025/videos'
    names=['libero_object_task7_trial0','libero_object_task7_trial3','libero_10_task3_trial2','libero_goal_task5_trial1']
    if args.suppress_action_tokens: names=names[:1]
    helpers={k:LanguageSubgoals(policy,dict(language_suppress_action_tokens=args.suppress_action_tokens,language_diagnostic_only=True,language_steps=32,language_tokens=160,language_log_dir=str(args.output/k),language_prompt_template=v)) for k,v in PROMPTS.items()}
    (args.output/'prompts.json').write_text(json.dumps(PROMPTS,indent=2))
    for name in names:
        trace=[json.loads(x) for x in (source/(name+'.jsonl')).read_text().splitlines()]
        frames=imageio.mimread(source/(name+'.mp4'),memtest=False)
        indices=sorted(set([0,len(frames)//3,2*len(frames)//3,len(frames)-1]))
        sheet=Image.new('RGB',(512,280*len(indices)),(255,255,255));draw=ImageDraw.Draw(sheet)
        for j,idx in enumerate(indices):
            frame=np.asarray(frames[idx]);image=Image.fromarray(frame);image.save(args.output/f'{name}_{idx}.png')
            sheet.paste(image,(0,j*280));draw.text((3,j*280+258),f'{name}: control step {idx}',fill='black')
        sheet.save(args.output/f'{name}_sheet.jpg')
        for idx in indices:
            frame=np.asarray(frames[idx]);obs=dict(episode_key=f'{name}:{idx}',instruction=trace[0]['instruction'],image=frame[:,:256],wrist_image=frame[:,256:])
            for label,helper in helpers.items():
                result=helper.get(obs)
                row=dict(clip=name,frame=idx,instruction=obs['instruction'],prompt=label,text=result,terminal_success=trace[-1]['success'])
                with (args.output/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    (args.output/'complete.json').write_text(json.dumps({'complete':True}))
if __name__=='__main__':main()
