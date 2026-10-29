"""Build a separate warm-start experiment and test its real input/output paths."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from Roboactualizer.utils import misc


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); r, out = args.root, args.output
    source = r/'experiments/action_only/ctp_grid20k_search/encoder_011000/train_action_only_011000'
    cache = out/'cache_build/cache_final_011000_exclude_goal82'
    out.mkdir(exist_ok=True)
    cfg = OmegaConf.load(source/'config.yaml')
    cfg.output_dir = str(out/'train')
    cfg.batch_size = 40; cfg.gradient_accumulation_steps = 1; cfg.num_workers = 4
    cfg.max_steps = 20000; cfg.save_every = 5000; cfg.log_every = 10
    cfg.model.action_only = False; cfg.model.future_size = 4; cfg.model.image_predict_weight = 1.
    cfg.model.predictive_context = True; cfg.model.predictor_checkpointing = False
    cfg.model.eval_encode_fp32 = True
    cfg.data.train.action_only = False; cfg.data.train.strict_read = True
    cfg.data.train.rgb_latent_cache_dir = str(cache)
    cfg.data.train.pretrained_norm_stats = str(source/'dataset_stats.json')
    # Exact original per-channel statistics keep the warm-start input distribution.
    cfg.model.latent_norm_stats_path = str(source.parent/'cache_retry1/cache_final_011000_exclude_goal82/latent_norm_stats.json')
    cfg.resume = str(out/'initialization.pt')
    cfg.run_name = 'grid11k-action20k-predictive-context-wm'
    cfg.wandb.project = 'eb_vjepa'; cfg.wandb.name = cfg.run_name
    cfg.wandb.workspace = 'zjc010100000-xi-an-jiaotong-university-'
    cfg.wandb.group = 'grid11k-predictive-context'; cfg.wandb.enabled = True
    cfg.model.warm_start_source = str(source/'checkpoints/weights/step_020000.pt')
    cfg.model.world_objective = 'normalized_frozen_latent_mse'
    OmegaConf.save(cfg, out/'train_config.yaml')

    # The cache workers write sharded metadata; equivalent old metadata permits
    # a real-batch preflight while remaining episode payloads are still encoding.
    old_cache = Path(str(cfg.model.latent_norm_stats_path)).parent
    for p in old_cache.glob('*/latents_meta.json'):
        target = cache/p.parent.name/p.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    torch.set_num_threads(2); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model = instantiate(cfg.model, model_dtype=torch.float32, device='cuda:0')
    receipt = model.warm_start_action_only(cfg.model.warm_start_source)
    model.metadata.update(experiment='grid11k_predictive_context', warm_start=receipt)
    model.save_checkpoint(str(out/'initialization.pt'), step=0)
    misc.register_work_dir(out/'preflight')
    dataset = instantiate(cfg.data.train)
    sample = dataset[0]
    torch.save(sample, out/'preflight/real_sample.pt')
    sample = {k: v.unsqueeze(0).to('cuda') if torch.is_tensor(v) else v for k, v in sample.items()}
    batch = dict(z0=sample['video_latent'][:, 0].float(),
        z_future=sample['video_latent'][:, 1:].float(),
        action_chunk=sample['action'][:, :16].float(), proprio=sample['proprio'][:, 0].float(),
        text_context=sample['context'], text_mask=sample['context_mask'].bool(),
        action_is_pad=sample['action_is_pad'][:, :16].bool(), image_is_pad=sample['image_is_pad'].bool())
    report = dict(warm_start=receipt, shapes={k: list(v.shape) for k,v in batch.items()},
        target_offsets=[4,8,12,16], source_action_updates=20000, new_updates=20000,
        world_size=6, global_batch=240, original_global_batch=256)
    noise = torch.randn_like(batch['action_chunk']); sigma = torch.full((1,1,1),.5,device='cuda')
    timestep = torch.full((1,),500.,device='cuda')
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        a = model.prediction_and_targets(batch, noise, sigma, timestep)
        changed = dict(batch, z_future=batch['z_future'] * 7 + 5)
        b = model.prediction_and_targets(changed, noise, sigma, timestep)
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
    report['future_target_leakage'] = False
    model.train(); model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        velocity, target, _, _ = model.prediction_and_targets(batch, noise, sigma, timestep)
        loss = (velocity.float() - target.float()).square().mean()
    loss.backward()
    query_grad = model.image_token_encoder.future_query.grad.norm().item()
    assert query_grad > 0 and torch.isfinite(model.image_token_encoder.future_query.grad).all()
    report['action_to_future_query_grad_norm'] = query_grad
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-6)
    report['updates'] = []
    for _ in range(2):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, logs = model(batch)
        assert torch.isfinite(loss)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        report['updates'].append(dict(loss=float(loss.detach()), grad_norm=float(grad),
                                     **{k:float(v) for k,v in logs.items()}))
    # These two updates test plumbing only; production starts from the untouched
    # initialization.pt saved before the smoke optimizer was constructed.
    report['peak_memory_gib'] = torch.cuda.max_memory_allocated()/2**30
    model.eval()
    generated = []
    with patch.object(model.backbone, 'encode_clip_last_group', return_value=batch['z0']), \
            torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for use_cache, use_graph in [(False, False), (True, False), (True, True)]:
            torch.manual_seed(239); torch.cuda.manual_seed_all(239)
            actions = model.generate(batch['z0'], batch['text_context'], batch['text_mask'],
                batch['proprio'], num_inference_steps=4, use_kv_cache=use_cache,
                use_cuda_graph=use_graph)
            generated.append(actions.clone())
    report['inference_path_max_abs_diffs'] = [float((generated[0]-x).abs().max()) for x in generated[1:]]
    for actions in generated[1:]:
        torch.testing.assert_close(generated[0], actions, rtol=.02, atol=.02)
    report['status'] = 'passed'
    (out/'preflight/results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
