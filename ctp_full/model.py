"""Checkpoint-compatible upstream models and the official video JEPA loss."""
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from app.vjepa_2_1.models.predictor import vit_predictor
from app.vjepa_2_1.models.vision_transformer import vit_large
from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
from app.vjepa_2_1.models.utils.modules import Lambda_LinearWarmupHold
from src.masks.utils import apply_masks

HEAD_KEYS = {
    "predictor_proj.weight", "predictor_proj.bias",
    "predictor_proj_context.weight", "predictor_proj_context.bias",
}


def clean_state(state):
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in cleaned:
            raise ValueError(f"Duplicate normalized checkpoint key: {key}")
        cleaned[key] = value
    return cleaned


def load_report(module, state):
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatches = {
        key: {"expected": list(expected[key].shape), "checkpoint": list(state[key].shape)}
        for key in set(state) & set(expected)
        if expected[key].shape != state[key].shape
    }
    loaded = [key for key in expected if key in state and key not in mismatches]
    denominator = sum(v.numel() for v in expected.values())
    return {
        "missing_keys": missing, "unexpected_keys": unexpected,
        "shape_mismatches": mismatches, "loaded_keys": sorted(loaded),
        "loaded_tensor_pct": 100 * len(loaded) / len(expected),
        "loaded_numel_pct": 100 * sum(expected[k].numel() for k in loaded) / denominator,
    }


def make_backbones(activation_checkpointing, predictor_output=1664):
    common = dict(
        img_size=256, num_frames=16, patch_size=16, tubelet_size=2,
        use_rope=True, use_sdpa=True, uniform_power=True,
        img_temporal_dim_size=1, modality_embedding=True,
        use_activation_checkpointing=activation_checkpointing,
        n_output_distillation=1,
    )
    encoder = vit_large(**common, interpolate_rope=True)
    predictor = vit_predictor(
        **common, embed_dim=1024, predictor_embed_dim=384,
        depth=12, num_heads=12, num_mask_tokens=8, use_mask_tokens=True,
        return_all_tokens=True, teacher_embed_dim=predictor_output,
        # Match the released hub's predictor RoPE setting (encoder interpolates).
        interpolate_rope=False,
    )
    return encoder, predictor


def initialize(config, report_path):
    checkpoint_path = Path(config["root"]) / config["checkpoint"]
    if checkpoint_path.stat().st_size != config["checkpoint_bytes"]:
        raise RuntimeError("Released checkpoint is incomplete; refusing random initialization")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    encoder, predictor = make_backbones(config["activation_checkpointing"])
    source = config["encoder_source"]
    if source != "ema_encoder":
        raise ValueError("CTP-v1 alignment must use ema_encoder")
    enc_state = clean_state(checkpoint[source])
    pred_state = clean_state(checkpoint["predictor"])
    report = {
        "checkpoint": str(checkpoint_path), "alignment_source": source,
        "encoder": load_report(encoder, enc_state),
        "predictor_released": load_report(predictor, pred_state),
        "predictor_depth": 12, "num_mask_tokens": 8,
        "target_representation": "ema_encoder final normalized 1024-dim layer",
        "loss_recipe": "upstream L1 prediction plus distance-weighted progressive context loss",
    }
    Path(report_path).write_text(json.dumps(report, indent=2))
    for name in ("encoder", "predictor_released"):
        item = report[name]
        if item["missing_keys"] or item["unexpected_keys"] or item["shape_mismatches"]:
            raise RuntimeError(f"Unexpected checkpoint mismatch in {name}; see {report_path}")
    encoder.load_state_dict(enc_state, strict=True)
    predictor.load_state_dict(pred_state, strict=True)
    target = copy.deepcopy(encoder).requires_grad_(False)
    target.use_activation_checkpointing = False
    target.eval()
    report["target_encoder"] = load_report(target, enc_state)
    if predictor.predictor_proj.out_features != encoder.embed_dim:
        if config["predictor_head_policy"] != "reinitialize_native":
            raise RuntimeError("Teacher-space heads require explicit reinitialize_native policy")
        for name in ("predictor_proj", "predictor_proj_context"):
            replacement = nn.Linear(predictor.predictor_embed.out_features, encoder.embed_dim)
            predictor._init_weights(replacement)
            setattr(predictor, name, replacement)
    reused = sum(v.numel() for k, v in pred_state.items() if k not in HEAD_KEYS)
    report["head_adaptation"] = {
        "policy": config["predictor_head_policy"],
        "reinitialized_keys": sorted(HEAD_KEYS),
        "source_output_dim": 1664, "target_output_dim": encoder.embed_dim,
        "retained_checkpoint_numel_pct": 100 * reused / sum(v.numel() for v in pred_state.values()),
        "trunk_loaded_pct": 100.0,
        "note": "Only two teacher-space output heads replaced; not a random predictor.",
    }
    report["unused_video_only_parameters"] = [
        f"encoder.{n}" for n, _ in encoder.named_parameters()
        if n.startswith(("patch_embed_img.", "img_mod_embed", "norms_block.0.",
                         "norms_block.1.", "norms_block.2."))
    ] + ["predictor.img_mod_embed"] + [f"predictor.mask_tokens.{i}" for i in range(2, 8)]
    Path(report_path).write_text(json.dumps(report, indent=2))
    print(json.dumps({"load_report": str(report_path), "head_adaptation": report["head_adaptation"]}), flush=True)
    del checkpoint
    return Student(encoder, predictor), target, report


class Student(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.context_schedule = Lambda_LinearWarmupHold(lambda_value=0.5)

    def forward(self, clips, target, masks_enc, masks_pred, step):
        # Same L1 target/context objective as upstream, one released student level.
        context_weight = self.context_schedule.value(step)
        distance = compute_mask_distance([masks_pred], [masks_enc], 16, False)[0]
        components = []
        for i, (me, mp) in enumerate(zip(masks_enc, masks_pred)):
            encoded = self.encoder(clips, masks=me)
            pred, context = self.predictor(encoded, me, mp, mod="video", mask_index=i)
            ht = apply_masks(target, [mp])
            hc = apply_masks(target, [me])
            pred_loss = (pred.float() - ht.float()).abs().mean()
            weights = distance[i].reshape(me.shape).unsqueeze(-1)
            context_loss = ((context.float() - hc.float()).abs() / weights).mean()
            components.append(torch.stack((pred_loss, context_loss)))
        components = torch.stack(components)
        total = (components[:, 0] + context_weight * components[:, 1]).mean()
        return total, components.detach()


@torch.no_grad()
def ema_update(student, target, momentum):
    online = dict(student.named_parameters())
    for name, parameter in target.named_parameters():
        parameter.mul_(momentum).add_(online[name], alpha=1 - momentum)
    for name, buffer in target.named_buffers():
        buffer.copy_(dict(student.named_buffers())[name])


def target_forward(target, clips):
    with torch.no_grad():
        features = target(clips)
        return F.layer_norm(features, (features.shape[-1],))
