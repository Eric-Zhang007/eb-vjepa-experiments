"""Official Dream-VLA continuous-flow baseline, isolated from E-LAF.

Checklist: official modules/LoRA; preserve shifted action positions; verify
batched flow and trainable-only roundtrip; real checkpoint one-step smoke.
"""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn


@dataclass
class DreamBaselineConfig:
    checkpoint: str
    dream_repo: str
    lora_rank: int = 32
    lora_alpha: int = 16
    gradient_checkpointing: bool = True
    attention_implementation: str = "sdpa"


def _official_modules(repo):
    """Load official source without unrelated Prismatic registry imports.

Only remove its platform constants import, supplying explicit LIBERO constants;
all official class/function definitions execute unchanged.
"""
    result = []
    for name in ("action_heads", "projectors"):
        path = Path(repo) / "vla/prismatic/models" / (name + ".py")
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "prismatic.vla.constants")]
        module = types.ModuleType("dream_baseline.official_" + name)
        module.__dict__.update(ACTION_DIM=7, NUM_ACTIONS_CHUNK=8, PROPRIO_DIM=8,
            ACTION_TOKEN_BEGIN_IDX=151666, IGNORE_INDEX=-100, STOP_INDEX=2)
        exec(compile(tree, str(path), "exec"), module.__dict__)
        result.append(module)
    return result


class DreamBaselinePolicy(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.config = config
        self.backbone = backbone
        heads, projectors = _official_modules(config.dream_repo)
        dim = backbone.config.hidden_size
        self.action_head = heads.FlowMatchingActionHead(input_dim=dim, hidden_dim=dim, action_dim=7)
        self.noisy_action_projector = projectors.NoisyActionProjector(llm_dim=dim)
        self.proprio_projector = projectors.ProprioProjector(llm_dim=dim, proprio_dim=8)

    @classmethod
    def from_pretrained(cls, config, device="cuda:0"):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor
        sys.path.insert(0, str(Path(config.dream_repo) / "vla"))
        from dreamvla.configuration_dreamvl import DreamVLAConfig
        from dreamvla.modeling_dreamvl import DreamVLAForActionPrediction
        native_config = DreamVLAConfig.from_pretrained(config.checkpoint, n_action_bins=256, norm_stats={})
        native_config._attn_implementation = config.attention_implementation
        native_config.vision_config._attn_implementation = config.attention_implementation
        backbone = DreamVLAForActionPrediction.from_pretrained(config.checkpoint, config=native_config,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
        backbone = get_peft_model(backbone, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=0.0, target_modules="all-linear", init_lora_weights="gaussian", bias="none"))
        if config.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Official PEFT keeps LoRA/master weights FP32; action modules are BF16,
        # while the proprio projector is FP32 and runs under BF16 autocast.
        model = cls(backbone, config).to(device=device)
        model.action_head.to(dtype=torch.bfloat16)
        model.noisy_action_projector.to(dtype=torch.bfloat16)
        model.processor = AutoProcessor.from_pretrained(config.checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False)
        model.processor.image_processor.size["min_pixels"] = 64 * 28 * 28
        model.processor.image_processor.min_pixels = 64 * 28 * 28
        model.processor.tokenizer.add_tokens([f"<|action_{n}|>" for n in range(256)])
        return model

    def _action_mask(self, batch):
        mask = batch.get("action_mask")
        if mask is None:
            labels = batch["labels"]
            mask = labels > 151666
        if mask.shape != batch["input_ids"].shape or not (mask.sum(1) == 56).all():
            raise ValueError("exactly 56 action positions per sample required")
        if mask[:, 0].any():
            raise ValueError("shifted action positions require a preceding prompt token")
        return mask.bool()

    def velocity(self, batch, noisy_actions, timestep_embeddings):
        if noisy_actions.shape != (batch["input_ids"].shape[0], 8, 7):
            raise ValueError("actions must be [B,8,7]")
        if batch["proprio"].shape != (noisy_actions.shape[0], 8):
            raise ValueError("proprio must be [B,8]")
        if batch["image_grid_thw"].shape != (2 * noisy_actions.shape[0], 3):
            raise ValueError("official baseline requires two camera images per sample")
        mask = self._action_mask(batch)
        features = self.noisy_action_projector(noisy_actions.reshape(noisy_actions.shape[0], 56, 1))
        output = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
            proprio=batch["proprio"], proprio_projector=self.proprio_projector,
            noisy_action_features=features, all_actions_mask=mask, timestep_embeddings=timestep_embeddings,
            return_dict=True, use_cache=False, num_logits_to_keep=1)
        # Exact official convention: outputs one token before action placeholders.
        states = output.hidden_states[:, :-1][mask[:, 1:]].reshape(noisy_actions.shape[0], 56, -1)
        return self.action_head.predict_velocity(states)

    def loss(self, batch):
        actions = batch["actions"].to(dtype=next(self.action_head.parameters()).dtype)
        if actions.shape != (batch["input_ids"].shape[0], 8, 7) or not torch.isfinite(actions).all():
            raise ValueError("finite actions [B,8,7] required")
        if "action_valid" in batch and not batch["action_valid"].bool().all():
            raise ValueError("baseline loss requires complete eight-action chunks")
        sampled = self.action_head.sample_noisy_actions(actions)
        prediction = self.velocity(batch, sampled["noisy_actions"], sampled["timestep_embeddings"])
        loss = (prediction.float() - (actions - sampled["noise"]).float()).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite baseline flow loss")
        return {"loss": loss, "action_loss": loss}

    def forward(self, batch):
        return self.loss(batch)

    @torch.inference_mode()
    def predict(self, batch, num_steps=4, noise=None):
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be positive integer")
        b = batch["input_ids"].shape[0]
        actions = self.action_head.sample_noise((b, 8, 7), batch["input_ids"].device) if noise is None else noise.clone()
        for step in range(num_steps):
            time = torch.full((b,), step / num_steps, device=actions.device, dtype=actions.dtype)
            embedding = self.action_head.time_encoder(time).to(actions.dtype).unsqueeze(1)
            actions = actions + self.velocity(batch, actions, embedding) / num_steps
        if not torch.isfinite(actions).all():
            raise FloatingPointError("nonfinite baseline prediction")
        return actions

    def save_trainable(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {name: value.detach().cpu() for name, value in self.named_parameters() if value.requires_grad}
        torch.save(state, directory / "adapter.pt")
        (directory / "model_config.json").write_text(json.dumps(asdict(self.config), indent=2))
        source_paths = [Path(self.config.dream_repo) / "vla/prismatic/models" / (name + ".py") for name in ("action_heads", "projectors")]
        (directory / "official_source_hashes.json").write_text(json.dumps({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}, indent=2))
        return {"trainable_tensors": len(state), "adapter_sha256": hashlib.sha256((directory / "adapter.pt").read_bytes()).hexdigest()}

    def load_trainable(self, directory):
        state = torch.load(Path(directory) / "adapter.pt", map_location="cpu", weights_only=True)
        expected = {name: value for name, value in self.named_parameters() if value.requires_grad}
        if state.keys() != expected.keys():
            raise ValueError("trainable checkpoint parameter names do not match model")
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != expected[name].shape:
                    raise ValueError(f"checkpoint shape mismatch: {name}")
                expected[name].copy_(value)
