"""Opt-in diagnostics for the E-LAF stability recovery experiment."""
import torch


def parameter_partition(policy):
    groups = {}
    for name, p in policy.named_parameters():
        if p.requires_grad:
            key = 'lora' if name.startswith('backbone.') else name.split('.')[0]
            groups.setdefault(key, []).append(p)
    return groups


def gradient_norms(groups):
    result = {}
    for name, params in groups.items():
        norms = [p.grad.detach().float().norm().square() for p in params if p.grad is not None]
        if norms:
            result['train/grad_preclip/' + name] = torch.stack(norms).sum().sqrt().item()
    return result


def parameter_snapshot(groups):
    return {name: [p.detach().clone() for p in params] for name, params in groups.items()}


def update_metrics(groups, before):
    result = {}
    for name, params in groups.items():
        size = torch.stack([v.float().square().sum() for v in before[name]]).sum().sqrt()
        change = torch.stack([(p.detach().float() - v.float()).square().sum()
                              for p, v in zip(params, before[name])]).sum().sqrt()
        result['train/update_norm/' + name] = change.item()
        result['train/update_relative/' + name] = (change / size.clamp_min(1e-12)).item()
    return result


def flow_metrics(result, batch):
    """Local rank debug batch; deliberately separate from globally reduced losses."""
    metrics = {}
    for stream, valid in [('action', batch['action_valid']), ('delta', batch['future_valid'][:, None])]:
        pred = result[stream + '_velocity'].detach().float()
        target = result[stream + '_velocity_target'].detach().float()
        mask = valid[..., None].expand_as(pred)
        pred, target = pred[mask], target[mask]
        if not target.numel():
            continue
        root = 'debug/rank0/' + stream
        metrics[root + '/prediction_rms'] = pred.square().mean().sqrt().item()
        metrics[root + '/target_rms'] = target.square().mean().sqrt().item()
        metrics[root + '/zero_velocity_mse'] = target.square().mean().item()
        metrics[root + '/mse'] = (pred-target).square().mean().item()
        metrics[root + '/cosine'] = (torch.dot(pred, target) / (pred.norm()*target.norm()).clamp_min(1e-12)).item()
    return metrics
