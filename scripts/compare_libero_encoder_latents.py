"""Matched LIBERO raw-latent inspection across one encoder training lineage."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def erank(x):
    x = x.float() - x.float().mean(0, keepdim=True)
    e = torch.linalg.eigvalsh(x @ x.T).clamp_min(0)
    if e.sum() < 1e-12:
        return 0.
    p = e / e.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))


def gram(x):
    x = x.float() - x.float().mean(0, keepdim=True)
    return x @ x.T


def cka(a, b):
    a, b = gram(a), gram(b)
    return float((a*b).sum() / (a.norm()*b.norm()).clamp_min(1e-20))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--long-run', action='store_true')
    args = p.parse_args(); root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6); torch.manual_seed(239)
    rng = np.random.default_rng(239)
    cache = root/'data/elaf_v0_grid11k_raw'
    manifest = json.loads((cache/'manifest.json').read_text())
    models = {'official': root/'checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt'}
    for step in [11000, 12000, 14000, 17000, 17358]:
        models[f'grid{step}'] = root/f'experiments/ctp_grid20k_20261008/embodied_vjepa_encoder_{step:06d}.pt'
    if args.long_run:
        models = {k:v for k,v in models.items() if k in ['official','grid11000']}
        for step in [10000,20000,40000,100000]:
            models[f'full{step}'] = root/f'experiments/ctp_full_72h/embodied_vjepa_encoder_{step:06d}.pt'
    samples, images = [], []
    # Four distinct instructions per suite, two interior positions per episode.
    for suite in ['spatial', 'object', 'goal', '10']:
        eps = [e for e in manifest['episodes'] if e['suite']==suite and e['length']>=32]
        groups = {}
        for ep in eps: groups.setdefault(ep['instruction'], []).append(ep)
        names = sorted(groups)
        for ix in rng.choice(len(names), 4, replace=False):
            pool = groups[names[ix]]; ep = pool[int(rng.integers(len(pool)))]
            a = np.load(cache/ep['path']/'images.npy', mmap_mode='r')
            for fraction in [.33, .66]:
                t = max(3, min(ep['length']-9, round((ep['length']-9)*fraction)))
                samples.append(dict(episode=ep['episode_id'],suite=suite,t=t,future=t+8,
                                    instruction=ep['instruction'],path=ep['path']))
                images.append(np.stack([a[t], a[t+8]]))
    dump(out/'samples.json', samples)
    np.save(out/'images.npy', np.stack(images))
    (out/'visual-contract.md').write_text('''# LIBERO encoder latent comparison
Question: how do spatial structure and temporal responses change after grid11k?
Six checkpoints: official and grid 11000/12000/14000/17000/17358 from one run.
32 paired clips, 16 demonstration episodes, four instructions per suite, seed 239.
Same raw RGB, four-frame window, 384px preprocessing, future endpoint t+8.
Raw encoder outputs only; no action head or E-LAF resampler. Grid is T,H,W=2,24,24,
confirmed from Conv3d flatten(2).transpose(1,2); maps show the last tubelet.
Shared PCA fitted on balanced L2-normalized tokens from all models and both times;
shared RGB percentiles. PCA colors are coordinates, not object labels.
Delta maps share a global scale. Similarity maps use the same reference patch,
chosen by maximum pixel change, and a fixed cosine scale.
Per-sample statistics retained. Pixel-change correlation is descriptive, not an
object mask or a measure of physical understanding. No success-rate causality claim.
Outputs: overview PNG/PDF, case PNG/PDF, quantitative JSON/CSV, HTML gallery.
''')
    from elaf.v0_cache import encode_windows, sha256
    src = root/'repos/RoboActualizer/src/Roboactualizer/models/roboactualizer/vjepa.py'
    spec = importlib.util.spec_from_file_location('latent_factory', src)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    identities = {}
    for name, checkpoint in models.items():
        dump(out/'status.json', dict(state='encoding',model=name,pid=os.getpid()))
        identities[name] = dict(path=str(checkpoint),sha256=sha256(checkpoint))
        dest = out/f'{name}.npy'
        if dest.exists(): continue
        teacher = module.build_vjepa_encoder(ckpt_path=str(checkpoint),model_size='large',
                    image_size=(384,384),torch_hub_dir=str(root/'repos/vjepa2'))
        teacher.eval().requires_grad_(False).to('cuda:0')
        values = []
        for i, s in enumerate(samples):
            a = np.load(cache/s['path']/'images.npy',mmap_mode='r')
            bits = encode_windows(teacher,a,[s['t'],s['future']],'cuda:0')
            values.append(torch.from_numpy(bits).view(torch.bfloat16).float().numpy())
            if i % 8 == 0: print(name,i,len(samples),flush=True)
        np.save(dest, np.stack(values).astype(np.float16))
        del teacher, values; gc.collect(); torch.cuda.empty_cache()
    dump(out/'identities.json',identities)
    dump(out/'status.json',dict(state='analysis',pid=os.getpid()))
    features = {k:torch.from_numpy(np.load(out/f'{k}.npy')).float() for k in models}
    # Fit one basis and one RGB scale, balanced across checkpoints and timestamps.
    train=[]
    for v in features.values():
        v=F.normalize(v.reshape(-1,1024),dim=-1)
        train.append(v[torch.from_numpy(rng.choice(len(v),2048,replace=False))])
    train=torch.cat(train);mean=train.mean(0);_,sv,basis=torch.pca_lowrank(train-mean,q=8,center=False,niter=4)
    basis=basis[:,:3];projected=(train-mean)@basis
    low=torch.quantile(projected,.01,dim=0);high=torch.quantile(projected,.99,dim=0)
    np.savez(out/'shared_pca.npz',mean=mean.numpy(),basis=basis.numpy(),low=low.numpy(),high=high.numpy())
    pca={};change={};records=[];affinity={}
    pixel=[]
    for im in images:
        diff=np.abs(im[1].astype(float)-im[0].astype(float)).mean(-1)/255
        pixel.append(F.adaptive_avg_pool2d(torch.tensor(diff)[None,None],(24,24))[0,0].numpy())
    for name,v in features.items():
        grid=v.reshape(len(samples),2,2,24,24,1024)
        last=grid[:,:,1]
        pca[name]=(((F.normalize(last,dim=-1)-mean)@basis-low)/(high-low)).clamp(0,1).numpy()
        delta=last[:,1]-last[:,0]
        change[name]=delta.square().mean(-1).sqrt().numpy()
        affinity[name]=[]
        for i,s in enumerate(samples):
            x,y=last[i,0].reshape(576,1024),last[i,1].reshape(576,1024)
            norm=F.normalize(x,dim=-1);ref=int(np.argmax(pixel[i]));affinity[name].append((norm@norm[ref]).reshape(24,24).numpy())
            spatial=x-x.mean(0)
            corr=float(spearmanr(pixel[i].ravel(),change[name][i].ravel()).statistic)
            order=np.argsort(pixel[i].ravel(),kind='stable')
            energy=change[name][i].ravel()**2
            records.append(dict(model=name,sample=i,episode=s['episode'],suite=s['suite'],
                rms=float(v[i,0].square().mean().sqrt()),delta_rms=float((v[i,1]-v[i,0]).square().mean().sqrt()),
                relative_delta=float((v[i,1]-v[i,0]).norm()/v[i,0].norm()),
                spatial_var=float(spatial.square().mean()),spatial_erank=erank(x[::3]),
                spatial_mean_cosine=float(((norm.sum(0).square().sum()-576)/(576*575))),
                temporal_cosine=float(F.cosine_similarity(x,y,dim=-1).mean()),
                top_pixel_quartile_delta_energy_fraction=float(energy[order[-144:]].sum()/energy.sum()),
                bottom_pixel_half_delta_rms=float(np.sqrt(energy[order[:288]].mean())),
                pixel_delta_spearman=corr if np.isfinite(corr) else None,
                cka_to_official=cka(x[::3],features['official'][i,0].reshape(2,576,1024)[1,::3]),
                cka_to_grid11k=cka(x[::3],features['grid11000'][i,0].reshape(2,576,1024)[1,::3])))
    dump(out/'per_sample_metrics.json',records)
    import csv
    with (out/'per_sample_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    keys=[k for k in records[0] if k not in ['model','sample','episode','suite']]
    summary={name:{k:float(np.median([x[k] for x in records if x['model']==name and x[k] is not None])) for k in keys} for name in models}
    dump(out/'summary.json',dict(n_pairs=len(samples),n_episodes=len(set(s['episode'] for s in samples)),medians=summary,
        caveat='Descriptive demonstration subset; no object labels, no causal attribution to control performance.'))
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    vmax=float(np.quantile(np.concatenate([x.ravel() for x in change.values()]),.99))
    names=list(models)
    # Overview: four preselected cases, no result-dependent cherry-picking.
    fig,axs=plt.subplots(4,7,figsize=(17,10),layout='constrained')
    for row,i in enumerate([0,8,16,24]):
        axs[row,0].imshow(images[i][0]);axs[row,0].set_ylabel(samples[i]['suite'])
        for j,name in enumerate(names):axs[row,j+1].imshow(pca[name][i,0],interpolation='nearest')
    for j,title in enumerate(['Current RGB']+names):axs[0,j].set_title(title)
    for ax in axs.flat:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Same LIBERO input, shared PCA coordinates (last temporal tubelet)',fontsize=16)
    fig.savefig(out/'overview.png',dpi=160);fig.savefig(out/'overview.pdf');plt.close(fig)
    cases=[]
    for i in [0,4,8,12,16,20,24,28]:
        fig,axs=plt.subplots(7,4,figsize=(11,17),layout='constrained')
        axs[0,0].imshow(images[i][0]);axs[0,1].imshow(images[i][1]);axs[0,2].imshow(pixel[i],cmap='magma');axs[0,3].axis('off')
        for ax,title in zip(axs[0],['Current RGB','Future RGB (t+8)','Pixel change','']):ax.set_title(title)
        ref=np.unravel_index(int(np.argmax(pixel[i])),(24,24));axs[0,2].plot(ref[1],ref[0],'+',color='cyan',markersize=10)
        for j,name in enumerate(names):
            axs[j+1,0].imshow(pca[name][i,0]);axs[j+1,1].imshow(pca[name][i,1]);axs[j+1,0].set_ylabel(name)
            dm=axs[j+1,2].imshow(change[name][i],cmap='magma',vmin=0,vmax=vmax)
            sm=axs[j+1,3].imshow(affinity[name][i],cmap='viridis',vmin=-1,vmax=1)
            axs[j+1,3].plot(ref[1],ref[0],'+',color='cyan',markersize=10)
        for ax in axs.flat:ax.set_xticks([]);ax.set_yticks([])
        for ax,title in zip(axs[1],['Current shared PCA','Future shared PCA','Latent delta RMS','Reference-patch cosine']):ax.set_title(title)
        fig.colorbar(dm,ax=list(axs[1:,2]),shrink=.5,label='Delta RMS (shared scale)')
        fig.colorbar(sm,ax=list(axs[1:,3]),shrink=.5,label='Cosine (not attention)')
        import textwrap
        fig.suptitle(f"{samples[i]['episode']} / frame {samples[i]['t']}\n"+textwrap.fill(samples[i]['instruction'],85),fontsize=13)
        name=f'case_{i:02d}';fig.savefig(out/f'{name}.png',dpi=140);fig.savefig(out/f'{name}.pdf');plt.close(fig);cases.append(name)
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    chartkeys=['delta_rms','spatial_var','spatial_erank','spatial_mean_cosine','pixel_delta_spearman','cka_to_grid11k']
    for ax,k in zip(axs.flat,chartkeys):
        vals=[[x[k] for x in records if x['model']==n] for n in names]
        ax.boxplot(vals,showfliers=False);ax.set_xticks(range(1,7),names,rotation=25)
        ax.set_title(k.replace('_',' '));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Matched LIBERO pairs: descriptive distribution, not control scores',fontsize=14)
    fig.savefig(out/'metrics.png',dpi=160);fig.savefig(out/'metrics.pdf');plt.close(fig)
    html='<!doctype html><meta charset="utf-8"><title>LIBERO latent comparison</title><style>body{max-width:1400px;margin:auto;font:17px sans-serif}img{width:100%}p{line-height:1.6}</style><h1>LIBERO latent comparison</h1><p>32 paired windows / 16 episodes. Shared PCA basis and scale across all six encoders. Colors are not semantic labels. Reference-patch cosine is not attention. Delta maps use the same scale. Training demonstration subset, not closed-loop trajectories.</p>'
    for name in ['overview','metrics']+cases:html+=f'<h2>{name}</h2><a href="{name}.pdf">PDF</a><img src="{name}.png">'
    (out/'index.html').write_text(html)
    dump(out/'status.json',dict(state='completed',pid=os.getpid(),time=time.time()))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
