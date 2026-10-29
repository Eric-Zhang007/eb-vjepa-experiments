"""Task-object ROI correspondence on existing matched raw encoder features.

Boxes are human-readable, visually inspected approximate visible regions, not
simulator segmentations. Current ROI prototypes retrieve future image patches.
No language is fed to the encoder. Scores are correspondence, not attention.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# XYXY in the 256px source frames. Exclude sample 0: named bowl is edge-occluded.
ANNOTATIONS = {
    8: {'object_name':'tomato sauce', 'target_name':'basket',
        'object': [[149,139,168,161],[146,143,167,161]],
        'target': [[20,105,77,160],[20,105,77,160]],
        'gripper': [[132,113,184,137],[134,121,187,145]],
        'distractor': [[182,134,202,172],[182,134,202,172]]},
    16: {'object_name':'bowl','target_name':'plate',
         'object': [[114,139,146,159],[114,143,146,160]],
         'target': [[110,166,156,203],[110,166,156,203]],
         'gripper': [[88,113,150,133],[88,128,150,146]],
         'distractor': [[76,152,94,173],[76,152,94,173]]},
    24: {'object_name':'white mug (visible body)','target_name':'plate',
         'object': [[91,174,142,215],[93,189,143,223]],
         'target': [[84,220,153,245],[84,226,153,250]],
         'gripper': [[32,137,147,161],[32,151,146,180]],
         'distractor': [[150,98,166,113],[132,104,148,126]]}
}


def mask(box):
    x,y=np.meshgrid((np.arange(24)+.5)*256/24,(np.arange(24)+.5)*256/24)
    x0,y0,x1,y1=box
    return ((x>=x0)&(x<x1)&(y>=y0)&(y<y1)).ravel()


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def auc(scores, positive, negative):
    a=scores[positive][:,None];b=scores[negative][None,:]
    return float(((a>b)+.5*(a==b)).mean())


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    args=p.parse_args();root=args.input;out=root/'task_regions';out.mkdir(exist_ok=True)
    images=np.load(root/'images.npy');samples=json.loads((root/'samples.json').read_text())
    names=list(json.loads((root/'identities.json').read_text()))
    feats={n:np.load(root/f'{n}.npy',mmap_mode='r') for n in names}
    (out/'annotations.json').write_text(json.dumps(ANNOTATIONS,indent=2))
    rows=[];gallery=[]
    for i,ann in ANNOTATIONS.items():
        for role in ['object','target','gripper']:
            src=mask(ann[role][0]);dst=mask(ann[role][1]);dist=mask(ann['distractor'][1])&~dst
            assert src.any() and dst.any() and dist.any()
            fig,axs=plt.subplots(2,4,figsize=(14,7),layout='constrained');axes=list(axs.flat)
            for t in range(2):
                ax=axes[t];ax.imshow(images[i,t]);x0,y0,x1,y1=ann[role][t]
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='#D55E00',lw=2))
                ax.set_title(['Current query ROI','Future evaluation ROI'][t])
            for j,n in enumerate(names):
                v=np.asarray(feats[n][i],dtype=np.float32).reshape(2,2,576,1024)[:,1]
                # Subtract the source image mean from BOTH times, preserving
                # temporal changes while reducing common scene components.
                c=v[0].mean(0);a,b=normalize(v[0]-c),normalize(v[1]-c)
                q=normalize(a[src].mean(0));scores=b@q
                raw=normalize(v[1])@normalize(normalize(v[0])[src].mean(0))
                k=int(dst.sum());top=np.argsort(scores)[-k:]
                rows.append(dict(sample=i,role=role,model=n,source_patches=int(src.sum()),future_patches=k,
                    future_auc=auc(scores,dst,~dst),distractor_auc=auc(scores,dst,dist),
                    future_topk_recall=float(dst[top].mean()),
                    future_target_minus_distractor=float(scores[dst].mean()-scores[dist].mean()),
                    raw_future_auc=auc(raw,dst,~dst)))
                ax=axes[j+2];im=ax.imshow(scores.reshape(24,24),cmap='viridis',vmin=-1,vmax=1)
                x0,y0,x1,y1=np.array(ann[role][1])*24/256-.5
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='#D55E00',lw=2))
                ax.set_title(f'{n}: recall {rows[-1]["future_topk_recall"]:.0%}')
            for ax in axes:ax.set_xticks([]);ax.set_yticks([])
            fig.colorbar(im,ax=axes[2:],shrink=.8,label='Centered feature cosine (not attention)')
            label=ann.get(role+'_name',role)
            fig.suptitle(f'{samples[i]["episode"]}: {label}\nCurrent-region prototype searches the future image',fontsize=14)
            name=f'sample_{i:02d}_{role}';fig.savefig(out/f'{name}.png',dpi=150);fig.savefig(out/f'{name}.pdf');plt.close(fig);gallery.append(name)
    (out/'metrics.json').write_text(json.dumps(rows,indent=2))
    summary={}
    for role in ['object','target','gripper']:
        summary[role]={n:{k:float(np.mean([r[k] for r in rows if r['role']==role and r['model']==n]))
                       for k in ['future_auc','distractor_auc','future_topk_recall','future_target_minus_distractor','raw_future_auc']}
                       for n in names}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    html='<!doctype html><meta charset="utf-8"><title>Task-region inspection</title><style>body{max-width:1400px;margin:auto;font:18px sans-serif}img{width:100%}</style><h1>Task-region inspection</h1><p>Three manually inspected examples, approximate visible-region boxes. Same current object / target / gripper ROI across encoders; retrieve matching future patches. Shared cosine scale. No language conditioning; these are correspondence maps, not attention or causal importance. Small/occluded objects have few patches.</p>'
    for n in gallery:html+=f'<h2>{n}</h2><img src="{n}.png">'
    (out/'index.html').write_text(html)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
