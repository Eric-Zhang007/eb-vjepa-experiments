"""Same-frame regional cohesion and separation; no temporal retrieval score."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from inspect_task_region_latents import ANNOTATIONS,mask,normalize

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    a=p.parse_args();root=a.input;out=root/'region_separation';out.mkdir(exist_ok=True)
    bg={8:[10,190,80,230],16:[10,180,65,225],24:[180,175,230,230]}
    models=['official','grid11000','full100000','full103233']
    images=np.load(root/'images.npy');records=[]
    for name in models:
        features=np.load(root/f'{name}.npy',mmap_mode='r')
        for i,ann in ANNOTATIONS.items():
            for t in [0,1]:
                raw=np.asarray(features[i,t],dtype=np.float32).reshape(2,576,1024)[1]
                x=normalize(raw);sim=x@x.T
                for role in ['object','target','gripper']:
                    pos=np.flatnonzero(mask(ann[role][t]));background=np.flatnonzero(mask(bg[i]));dist=np.flatnonzero(mask(ann['distractor'][t]))
                    within=sim[np.ix_(pos,pos)];cohesion=float(within[~np.eye(len(pos),dtype=bool)].mean())
                    b=float(sim[np.ix_(pos,background)].mean());d=float(sim[np.ix_(pos,dist)].mean()) if i!=24 else None
                    records.append(dict(model=name,sample=i,time=t,role=role,patches=len(pos),within_cosine=cohesion,
                        background_cosine=b,distractor_cosine=d,background_margin=cohesion-b,distractor_margin=cohesion-d if d is not None else None))
    (out/'metrics.json').write_text(json.dumps(records,indent=2))
    summary={role:{name:{k:float(np.mean([x[k] for x in records if x['model']==name and x['role']==role and x[k] is not None]))
              for k in ['within_cosine','background_margin','distractor_margin']} for name in models} for role in ['object','target','gripper']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'annotations.json').write_text(json.dumps(dict(regions=ANNOTATIONS,background=bg),indent=2))
    fig,axs=plt.subplots(3,2,figsize=(11,12),layout='constrained')
    colors={'object':'#D55E00','target':'#0072B2','gripper':'#CC79A7','distractor':'#E69F00','background':'#009E73'}
    for row,(i,ann) in enumerate(ANNOTATIONS.items()):
        for t in [0,1]:
            ax=axs[row,t];ax.imshow(images[i,t]);ax.set_title(f'Sample {i}, frame '+['t','t+8'][t])
            for role,c in colors.items():
                if role=='distractor' and i==24:continue
                box=bg[i] if role=='background' else ann[role][t];x,y,xx,yy=box
                ax.add_patch(Rectangle((x,y),xx-x,yy-y,fill=False,edgecolor=c,lw=2,label=role))
            ax.set_xticks([]);ax.set_yticks([])
    axs[0,0].legend(loc='upper left',fontsize=8);fig.savefig(out/'regions.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,key in zip(axs,['background_margin','distractor_margin']):
        for role,c in [('object','#D55E00'),('target','#0072B2'),('gripper','#CC79A7')]:
            ax.plot(models,[summary[role][n][key] for n in models],marker='o',label=role,color=c)
        ax.set_title(key.replace('_',' '));ax.set_ylabel('Within-region cosine minus cross-region cosine');ax.axhline(0,color='gray',lw=.7);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Same-frame region separation: 3 cases, 2 frames each; descriptive only')
    fig.savefig(out/'separation.png',dpi=160);fig.savefig(out/'separation.pdf');plt.close(fig)
    (out/'index.html').write_text('<meta charset="utf-8"><h1>Same-frame task region separation</h1><p>Within-region cosine excludes self-pairs. Equal case/frame weighting. Higher margin means annotated regions are more distinct under cosine; it is not task attention, pose accuracy, or causal action dependence. Three approximate boxes per region; spatial distance and appearance confound the score.</p><img width="95%" src="regions.png"><img width="95%" src="separation.png">')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
