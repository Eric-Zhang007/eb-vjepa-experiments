"""E-LAF training with a native Dream decoder, explicit resume, and LoRA."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from elaf.data import ELAFDataset, collate_fn
from elaf.model import ELAFConfig, ELAFPolicy


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_backbone(cfg):
    sys.path.insert(0, str(Path(cfg["dream_repo"]) / "vla"))
    from dreamvla.configuration_dreamvl import DreamVLAConfig
    from dreamvla.modeling_dreamvl import DreamVLAForActionPrediction

    if cfg.get("tiny", False):
        config = DreamVLAConfig(
            vocab_size=152064, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            n_action_bins=256, norm_stats={}, projector_hidden_act="gelu", mrope_section=[2,3,3],
            vision_config=dict(depth=2, embed_dim=32, hidden_size=64,
                               num_heads=4, patch_size=14, spatial_merge_size=2,
                               temporal_patch_size=2),
        )
        config._attn_implementation = "sdpa"
        backbone = DreamVLAForActionPrediction(config)
    else:
        checkpoint = Path(cfg["checkpoint"])
        receipt = checkpoint / "download_receipt.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"Verified download receipt required: {receipt}")
        backbone = DreamVLAForActionPrediction.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32,
            attn_implementation="sdpa", local_files_only=True,
        )
    backbone.requires_grad_(False)
    if cfg["backbone_training"] == "lora":
        from peft import LoraConfig, get_peft_model
        backbone.model = get_peft_model(backbone.model, LoraConfig(
            r=cfg["lora_rank"], lora_alpha=cfg["lora_rank"] * 2,
            lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
    elif cfg["backbone_training"] == "full":
        backbone.model.requires_grad_(True)
    elif cfg["backbone_training"] != "frozen":
        raise ValueError("backbone_training must be lora, full or frozen")
    if cfg.get("gradient_checkpointing", False):
        backbone.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return backbone


def validate(cfg, output, resume):
    for key in ("manifest", "dream_repo", "checkpoint"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"{key}: {cfg[key]}")
    for key in ("batch_size", "max_steps", "save_every", "gradient_accumulation"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["precision"] not in ("bf16", "no"):
        raise ValueError("precision must be bf16 or no")
    if cfg["device"] not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if cfg["precision"] == "bf16" and (cfg["device"] != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 requires a supported CUDA device")
    if resume and not Path(resume).is_dir():
        raise FileNotFoundError(resume)
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated output: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tiny", action="store_true", help="Random tiny native Dream model: plumbing test only")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.tiny:
        cfg["tiny"] = True
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    validate(cfg, args.output, args.resume)
    accelerator = Accelerator(cpu=cfg["device"] == "cpu", mixed_precision=cfg["precision"],
                              gradient_accumulation_steps=cfg["gradient_accumulation"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    set_seed(cfg["seed"])
    dataset = ELAFDataset(cfg["manifest"])
    if dataset.manifest.get("smoke_only", False) and not (cfg.get("allow_smoke_data", False) and cfg["max_steps"] <= 2):
        raise ValueError("Smoke samples require allow_smoke_data=true and max_steps<=2")
    if len(dataset) < cfg["batch_size"] * accelerator.num_processes:
        raise ValueError("Dataset smaller than one global batch")
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                        collate_fn=collate_fn, num_workers=cfg["workers"], drop_last=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    policy = ELAFPolicy(build_backbone(cfg), ELAFConfig(**cfg["model"]))
    policy.to(dtype=torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, cfg["warmup_steps"]))
    )
    # Step the scheduler once per optimizer update, independent of GPU count.
    policy, optimizer, loader = accelerator.prepare(policy, optimizer, loader)
    accelerator.register_for_checkpointing(scheduler)
    contract = dict(config=cfg, manifest_sha256=digest(cfg["manifest"]),
                    world_size=accelerator.num_processes)
    step, epoch, batch_offset = 0, 0, 0
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = json.loads((args.resume / "progress.json").read_text())
        if state["contract"] != contract:
            raise ValueError("Resume contract mismatch; use identical data/config/world size")
        accelerator.load_state(str(args.resume))
        step, epoch, batch_offset = state["step"], state["epoch"], state["batch_offset"]
    if accelerator.is_main_process:
        (args.output / "config.json").write_text(json.dumps(contract, indent=2))
        logging.info("Model %s | data %d | trainable %s | device %s | precision %s | output %s",
                     "TINY RANDOM SMOKE" if cfg.get("tiny") else "Dream-VLA E-LAF",
                     len(dataset), sum(p.numel() for p in trainable), accelerator.device,
                     cfg["precision"], args.output)

    def save(next_epoch, next_offset):
        destination = args.output / "checkpoints" / f"step_{step:06d}"
        accelerator.save_state(str(destination))
        if accelerator.is_main_process:
            (destination / "progress.json").write_text(json.dumps(dict(
                contract=contract, step=step, epoch=next_epoch, batch_offset=next_offset), indent=2))
        accelerator.wait_for_everyone()

    while step < cfg["max_steps"]:
        policy.train()
        for index, batch in enumerate(loader):
            if index < batch_offset:
                continue
            with accelerator.accumulate(policy):
                result = policy(batch)
                if not torch.isfinite(result["loss"]):
                    raise FloatingPointError("Non-finite training loss")
                accelerator.backward(result["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, cfg["max_grad_norm"])
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            step += 1
            record = {"step": step, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}
            for key in ("loss", "action_loss", "latent_loss"):
                record[key] = accelerator.reduce(result[key].detach(), reduction="mean").item()
            if accelerator.is_main_process:
                logging.info("step %d loss %.5f action %.5f latent %.5f", step, record["loss"], record["action_loss"], record["latent_loss"])
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            if step % cfg["save_every"] == 0 or step == cfg["max_steps"]:
                save(epoch, index + 1)
            if step == cfg["max_steps"]:
                break
        epoch += 1
        batch_offset = 0
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (args.output / "complete.json").write_text(json.dumps({"step": step, "tiny": cfg.get("tiny", False)}))


if __name__ == "__main__":
    main()
