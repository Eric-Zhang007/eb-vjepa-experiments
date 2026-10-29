"""Batched policies returning raw training coordinates, gripper 0=close/1=open.

Images must already have simulator orientation corrected. Each observation has
image (uint8 RGB), proprio (8,), instruction, and an explicit noise_seed. RoboActualizer action-only
also requires history=[{image,wrist_image}, ...], updated on every environment
step. One instance loads one model, shared by all environments in its process.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def validate_observations(observations, proprio_dim=8):
    if not observations:
        raise ValueError("empty inference batch")
    for obs in observations:
        validate_image(obs["image"])
        proprio = np.asarray(obs["proprio"])
        if proprio.shape != (proprio_dim,) or not np.isfinite(proprio).all():
            raise ValueError(f"proprio must be finite [{proprio_dim}]")
        if not isinstance(obs["instruction"], str) or not obs["instruction"].strip():
            raise ValueError("instruction must be nonempty")
        if not isinstance(obs["noise_seed"], int) or not 0 <= obs["noise_seed"] < 2**63:
            raise ValueError("noise_seed must be integer in [0,2**63)")


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must be uint8 HWC RGB")


def initial_noise(observations, shape, device, dtype, stream=0):
    # Each trajectory's draw is independent of batching, ordering and global RNG.
    tensors = []
    for obs in observations:
        seed = (obs["noise_seed"] + stream * 6364136223846793005) % (2**63)
        generator = torch.Generator(device=device).manual_seed(seed)
        tensors.append(torch.randn(shape, generator=generator, device=device, dtype=dtype))
    return torch.stack(tensors)


def _result(tensor, batch_size, action_dim=7):
    result = tensor.detach().float().cpu().numpy()
    if result.ndim != 3 or result.shape[0] != batch_size or result.shape[2] != action_dim or not np.isfinite(result).all():
        raise ValueError(f"invalid batched policy output {result.shape}")
    return result


class RoboActualizerActionOnlyPolicy:
    def __init__(self, config):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        repo = Path(config["repo"])
        sys.path.insert(0, str(repo / "src"))
        from Roboactualizer.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        self.prompt_template = DEFAULT_PROMPT
        self.device = torch.device(config.get("device", "cuda:0"))
        self.steps = int(config.get("num_steps", 4))
        if self.steps < 1 or self.device.type != "cuda":
            raise ValueError("RoboActualizer action-only evaluator requires CUDA and positive num_steps")
        run = Path(config["run"])
        cfg = OmegaConf.load(run / "config.yaml")
        if not cfg.model.action_only:
            raise ValueError("this adapter requires the action-only policy")
        cfg.model.eval_encode_fp32 = True
        self.model = instantiate(cfg.model, model_dtype=torch.float32, device=str(self.device)).eval()
        checkpoint = run / "checkpoints" / "weights" / f"step_{int(config['step']):06d}.pt"
        self.model.load_checkpoint(str(checkpoint))
        self.processor = instantiate(cfg.data.train.processor).eval()
        stats_path = run / "dataset_stats.json"
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        self.clip_len, self.interval = int(cfg.model.clip_len), int(cfg.model.frame_interval)
        self.history_length = (self.clip_len - 1) * self.interval + 1
        self.text_cache = Path(config["text_cache"])
        self.context_len = int(cfg.model.tokenizer_max_len)
        self.texts = {}
        self.identity = {"type": "roboactualizer_action_only", "policy_family": "RoboActualizer action-only derivative", "training_mode": "action_only", "architecture": "VJEPA encoder plus action DiT", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "encoder": str(cfg.model.backbone.ckpt_path), "encoder_sha256": sha256(cfg.model.backbone.ckpt_path),
            "stats_sha256": sha256(stats_path), "latent_stats_sha256": sha256(cfg.model.latent_norm_stats_path),
            "config_sha256": sha256(run / "config.yaml"), "text_cache": str(self.text_cache),
            "num_steps": self.steps, "action_coordinates": "raw_dataset_gripper_0close_1open",
            "history_length": self.history_length, "noise": "per_observation_seed", "eval_encode_fp32": True,
            "cache": "z0_and_cross_attention", "device": str(self.device)}
        self.identity.update(action_dim=int(cfg.model.raw_action_dim), action_horizon=int(cfg.model.chunk_size),
            proprio_dim=int(cfg.model.proprio_dim), action_space="libero_delta_ee", recommended_execute_steps=10)

    def fingerprint(self):
        return dict(self.identity, text_files={k: v[2] for k, v in self.texts.items()})

    def _text(self, instruction):
        prompt = self.prompt_template.format(task=instruction)
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in self.texts:
            path = self.text_cache / f"{key}.t5_len{self.context_len}.wan22ti2v5b.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("prompt", prompt) != prompt:
                raise ValueError("cached prompt mismatch")
            context, mask = payload["context"].clone(), payload["mask"].bool()
            if context.shape != (self.context_len, 4096) or mask.shape != (self.context_len,) or not mask.any() or not torch.isfinite(context).all():
                raise ValueError("invalid cached text embedding")
            context[~mask] = 0
            self.texts[key] = (context.to(self.device, torch.bfloat16), mask.to(self.device), sha256(path))
        return self.texts[key][:2]

    def _frame(self, obs):
        images = []
        for key, meta in zip(("image", "wrist_image"), self.processor.shape_meta["images"]):
            image = obs[key]
            validate_image(image)
            height, width = int(meta["shape"][1]), int(meta["shape"][2])
            source = Image.fromarray(image)
            scale = max(width / source.width, height / source.height)
            source = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.BILINEAR)
            left, top = (source.width - width) // 2, (source.height - height) // 2
            images.append(np.asarray(source.crop((left, top, left + width, top + height))))
        return torch.from_numpy(np.concatenate(images, axis=1).copy()).permute(2, 0, 1).float().mul_(2 / 255).sub_(1)

    @torch.inference_mode()
    def predict(self, observations):
        validate_observations(observations)
        clips = []
        for obs in observations:
            history = obs["history"]
            if not history:
                raise ValueError("RoboActualizer action-only requires nonempty every-step frame history")
            indices = [max(0, len(history) - 1 - i * self.interval) for i in reversed(range(self.clip_len))]
            clips.append(torch.stack([self._frame(history[i]) for i in indices]))
        pixels = torch.stack(clips).to(self.device)
        texts = [self._text(obs["instruction"]) for obs in observations]
        context = torch.stack([x[0] for x in texts])
        text_mask = torch.stack([x[1] for x in texts])
        state_key = self.processor.shape_meta["state"][0]["key"]
        states = {"state": {state_key: torch.as_tensor(np.stack([x["proprio"] for x in observations]), dtype=torch.float32)}}
        states = self.processor.normalizer.forward(self.processor.action_state_transform(states))
        proprio = states["state"][state_key].to(self.device)
        model = self.model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.autocast("cuda", enabled=False):
                z0 = model.backbone.encode_clip_last_group(pixels.float())
            z0 = model._normalize_latent(z0.to(model.rgb_latent_dtype).float())
            image_context, mask = model.image_context_builder(context, text_mask, proprio)
            action_context, _ = model.action_context_builder(context, text_mask, proprio)
            model.action_scheduler.set_inference_steps(self.steps)
            actions = initial_noise(observations, (model.chunk_size, model.action_dim), self.device, torch.float32)
            zero = torch.zeros(len(observations), device=self.device)
            cache = model._prefill_z0_cache(z0, image_context, mask, zero)
            cross = model._action_cross_kv(action_context)
            for step in range(self.steps):
                time = model.action_scheduler.scale_for_model_input(step, self.device).expand(len(observations))
                sigma = model.action_scheduler.sigma_for_step(step, self.device).view(1, 1, 1)
                prediction = model._action_only_forward_cached(cache, actions, action_context, mask, time, None, cross)
                velocity = model.action_scheduler.x_pred_to_velocity(prediction, actions, sigma)
                actions = actions + velocity * model.action_scheduler.dt[step].to(self.device)
        action_key = self.processor.shape_meta["action"][0]["key"]
        raw = self.processor.normalizer.normalizers["action"][action_key].backward(actions.float().cpu())
        return _result(raw, len(observations))


class ELAFInferencePolicy:
    def __init__(self, config):
        from elaf.model import ELAFConfig, ELAFPolicy
        from elaf.train import build_backbone
        from transformers import AutoProcessor
        from safetensors.torch import load_file
        checkpoint = Path(config["checkpoint"])
        progress_path = checkpoint / "progress.json"
        progress = json.loads(progress_path.read_text())
        cfg = progress["contract"]["config"]
        manifest_path = Path(cfg["manifest"])
        if sha256(manifest_path) != progress["contract"]["manifest_sha256"]:
            raise ValueError("training manifest identity changed")
        manifest = json.loads(manifest_path.read_text())
        stats_path = manifest_path.parent / manifest["stats_path"]
        if sha256(stats_path) != manifest["stats_sha256"]:
            raise ValueError("training statistics identity changed")
        self.stats = json.loads(stats_path.read_text())
        action_dim = int(cfg["model"].get("action_dim", 7))
        proprio_dim = int(cfg["model"].get("proprio_dim", 8))
        action_space = cfg.get("action_space")
        if action_space is None:
            if (action_dim, proprio_dim) != (7, 8):
                raise ValueError("Non-LIBERO model dimensions require explicit training action_space")
            action_space = "libero_delta_ee"
        if action_space not in ("libero_delta_ee", "xpolicy_joint_absolute", "xpolicy_ee_absolute_wxyz"):
            raise ValueError(f"Unsupported training action_space: {action_space}")
        if action_space == "libero_delta_ee" and (action_dim, proprio_dim) != (7, 8):
            raise ValueError("libero_delta_ee requires action_dim=7 and proprio_dim=8")
        for key, dimension in (("actions", action_dim), ("proprio", proprio_dim)):
            for statistic in ("mean", "std"):
                values = np.asarray(self.stats[key][statistic])
                if values.shape != (dimension,) or not np.isfinite(values).all():
                    raise ValueError(f"{key}.{statistic} must be finite [{dimension}]")
                if statistic == "std" and np.any(values <= 0):
                    raise ValueError(f"{key}.std must be positive")
        self.device = torch.device(config.get("device", "cuda:0"))
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = ELAFPolicy(build_backbone(cfg), ELAFConfig(**cfg["model"]))
        state_path = checkpoint / "model.safetensors"
        self.model.load_state_dict(load_file(str(state_path)), strict=True)
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.processor = AutoProcessor.from_pretrained(manifest["processor_path"], trust_remote_code=True, local_files_only=True, use_fast=False)
        self.steps, self.mode = int(config.get("num_steps", 4)), config.get("mode", "joint")
        if self.steps < 1 or self.mode not in ("joint", "fixed_latent_noise", "action_only"):
            raise ValueError("invalid inference steps or mode")
        if self.model.config.action_only and self.mode != "action_only":
            raise ValueError("action-only checkpoint requires action_only mode")
        if not self.model.config.action_only and self.mode != "joint" and not config.get("ablation", False):
            raise ValueError("Changing joint checkpoint inference mode requires ablation=true")
        self.identity = {"type": "elaf", "policy_family": "E-LAF", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(state_path),
            "progress_sha256": sha256(progress_path), "manifest_sha256": sha256(manifest_path), "stats_sha256": sha256(stats_path),
            "teacher": manifest.get("target"), "teacher_path": manifest.get("teacher_path"),
            "teacher_sha256": manifest.get("teacher_sha256"), "processor_revision": manifest.get("processor_revision"),
            "num_steps": self.steps, "mode": self.mode, "noise": "per_observation_seed", "device": str(self.device),
            "tiny": cfg.get("tiny", False), "action_coordinates": "raw_dataset_gripper_0close_1open" if action_space == "libero_delta_ee" else "raw_training_coordinates_no_conversion"}
        self.identity.update(action_dim=self.model.config.action_dim, action_horizon=self.model.config.action_horizon,
            proprio_dim=self.model.config.proprio_dim, action_space=action_space,
            recommended_execute_steps=self.model.config.action_horizon, history_length=1,
            training_mode="action_only" if self.model.config.action_only else "joint_latent_action",
            architecture="Dream-VLA native decoder with continuous action and latent streams", ablation=config.get("ablation", False))

    def fingerprint(self):
        return dict(self.identity)

    @torch.inference_mode()
    def predict(self, observations):
        validate_observations(observations, proprio_dim=self.model.config.proprio_dim)
        images, texts = [], []
        for obs in observations:
            images.append(Image.fromarray(obs["image"]).resize((224, 224)))
            conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"What action should the robot take to {obs['instruction']}?"}]}]
            texts.append(self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = {key: value.to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype) for key, value in inputs.items()}
        proprio = np.stack([x["proprio"] for x in observations])
        inputs["proprio"] = torch.as_tensor((proprio - np.asarray(self.stats["proprio"]["mean"])) / np.asarray(self.stats["proprio"]["std"]), device=self.device, dtype=self.dtype)
        condition, mask = self.model._condition(inputs)
        cfg = self.model.config
        actions = initial_noise(observations, (cfg.action_horizon, cfg.action_dim), self.device, self.dtype)
        latents = None if self.mode == "action_only" else initial_noise(observations, (cfg.latent_tokens, cfg.latent_dim), self.device, self.dtype, stream=1)
        for step in range(self.steps):
            s = torch.full((len(observations),), step / self.steps, device=self.device)
            u = torch.zeros_like(s) if self.mode == "fixed_latent_noise" else s
            va, vz = self.model.velocity(condition, mask, actions, latents, s, u)
            actions = actions + va / self.steps
            if self.mode == "joint":
                latents = latents + vz / self.steps
        mean = torch.as_tensor(self.stats["actions"]["mean"], device=self.device)
        std = torch.as_tensor(self.stats["actions"]["std"], device=self.device)
        return _result(actions.float() * std + mean, len(observations), action_dim=cfg.action_dim)



def dream_eval_image(image, center_crop=True):
    """Official Dream/OpenVLA eval JPEG -> TF Lanczos3 -> optional 90% area crop."""
    validate_image(image)
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError("TensorFlow was initialized on GPU; eval image preprocessing must stay on CPU")
    with tf.device("/CPU:0"):
        pixels = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
        pixels = tf.cast(tf.clip_by_value(tf.round(tf.image.resize(pixels, (224, 224), method="lanczos3", antialias=True)), 0, 255), tf.uint8)
        if center_crop:
            value = tf.image.convert_image_dtype(pixels, tf.float32)
            side = tf.sqrt(tf.constant(0.9, tf.float32))
            offset = (1 - side) / 2
            box = tf.reshape(tf.stack([offset, offset, offset + side, offset + side]), (1, 4))
            value = tf.image.crop_and_resize(value[None], box, [0], (224, 224))[0]
            pixels = tf.image.convert_image_dtype(tf.clip_by_value(value, 0, 1), tf.uint8, saturate=True)
        return Image.fromarray(pixels.numpy()).convert("RGB")


class DreamVLABaselinePolicy:
    """Deprecated custom trainer checkpoint adapter; NOT official finetune output."""
    def __init__(self, config):
        from dream_baseline.model import DreamBaselineConfig, DreamBaselinePolicy
        checkpoint = Path(config["checkpoint"])
        contract_path = checkpoint / "training_contract.json"
        contract = json.loads(contract_path.read_text())
        if contract.get("model_family") != "dream_vla_baseline":
            raise ValueError("Expected original Dream-VLA baseline training contract")
        training = contract["config"]
        stats_path = checkpoint / "dataset_statistics.json"
        self.unnorm_key = config["unnorm_key"]
        stats_payload = json.loads(stats_path.read_text())
        if self.unnorm_key not in stats_payload:
            raise ValueError("Explicit unnorm_key absent from checkpoint statistics")
        training_key = training.get("unnorm_key", training.get("norm_key", training.get("suite")))
        if training_key != self.unnorm_key:
            raise ValueError(f"Eval norm key {self.unnorm_key!r} differs from trained suite {training_key!r}")
        self.stats = stats_payload[self.unnorm_key]
        for key, dim in (("action", 7), ("proprio", 8)):
            for quantile in ("q01", "q99"):
                arr = np.asarray(self.stats[key][quantile])
                if arr.shape != (dim,) or not np.isfinite(arr).all():
                    raise ValueError(f"Invalid {key}.{quantile}")
            if np.any(np.asarray(self.stats[key]["q99"]) < np.asarray(self.stats[key]["q01"])):
                raise ValueError("Invalid statistics bounds")
        for path_key, expected_key in (("dataset_statistics", "dataset_statistics_sha256"),):
            expected = contract.get("statistics_sha256") or contract.get(expected_key) or training.get(expected_key)
            if expected is not None and sha256(stats_path) != expected:
                raise ValueError("Checkpoint normalization identity mismatch")
            source = Path(training[path_key])
            if source.is_file() and sha256(source) != sha256(stats_path):
                raise ValueError("Checkpoint and training normalization files differ")
        model_path = checkpoint / "model_config.json"
        model_settings = json.loads(model_path.read_text())
        if model_settings != training["model"]:
            raise ValueError("Saved model configuration differs from training contract")
        model_cfg = DreamBaselineConfig(**model_settings)
        for name in ("model.py", "data.py"):
            expected_source = contract.get("source_sha256", {}).get(name)
            source = Path(__file__).resolve().parents[2] / "dream_baseline" / name
            if expected_source and sha256(source) != expected_source:
                raise ValueError(f"Dream baseline training source changed: {name}")
        for path, expected in json.loads((checkpoint / "official_source_hashes.json").read_text()).items():
            if sha256(path) != expected:
                raise ValueError(f"Official Dream implementation changed: {path}")
        self.device = torch.device(config.get("device", "cuda:0"))
        self.steps = int(config.get("num_steps", 4))
        if self.steps < 1:
            raise ValueError("num_steps must be positive")
        self.center_crop = bool(config.get("center_crop", True))
        # Fail before loading model weights if the official CPU image pipeline is unavailable.
        dream_eval_image(np.zeros((256, 256, 3), dtype=np.uint8), self.center_crop)
        self.model = DreamBaselinePolicy.from_pretrained(model_cfg, device=str(self.device))
        self.model.load_trainable(checkpoint)
        self.model.eval()
        self.processor = self.model.processor
        self.identity = dict(type="dream_vla_baseline", policy_family="Dream-VLA original flow baseline",
            training_mode="native_action_flow_lora", architecture="Dream-VLA + official FlowMatchingActionHead; no physical latent stream",
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint / "adapter.pt"),
            model_config_sha256=sha256(model_path), training_contract_sha256=sha256(contract_path),
            stats_sha256=sha256(stats_path), unnorm_key=self.unnorm_key, base_checkpoint=model_cfg.checkpoint,
            official_source_hashes=json.loads((checkpoint / "official_source_hashes.json").read_text()),
            action_dim=7, proprio_dim=8, action_horizon=8, recommended_execute_steps=8,
            action_space="libero_delta_ee", action_coordinates="raw_dataset_gripper_0close_1open",
            history_length=1, num_steps=self.steps, noise="per_observation_seed", num_images=2,
            image_preprocessing="official_tf_jpeg_lanczos3_224", center_crop=self.center_crop,
            image_tokens_per_view=64, device=str(self.device))

    def fingerprint(self):
        return dict(self.identity)

    @torch.inference_mode()
    def predict(self, observations):
        from dream_baseline.data import build_sample, collate_fn, normalize
        validate_observations(observations)
        samples = []
        for obs in observations:
            wrist = obs.get("wrist_image")
            if wrist is None:
                raise ValueError("Original Dream-VLA baseline requires wrist_image")
            images = [dream_eval_image(obs["image"], self.center_crop), dream_eval_image(wrist, self.center_crop)]
            proprio = normalize(np.asarray(obs["proprio"], np.float32), self.stats["proprio"])
            samples.append(build_sample(self.processor, images, obs["instruction"], np.zeros((8, 7), np.float32), proprio))
        batch = collate_fn(samples, pad_token_id=self.processor.tokenizer.pad_token_id)
        batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        noise = initial_noise(observations, (8, 7), self.device, torch.bfloat16)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            normalized = self.model.predict(batch, num_steps=self.steps, noise=noise).float()
        low = torch.as_tensor(self.stats["action"]["q01"], device=self.device)
        high = torch.as_tensor(self.stats["action"]["q99"], device=self.device)
        mask = torch.as_tensor(self.stats["action"].get("mask", [True] * 7), device=self.device, dtype=torch.bool)
        raw = torch.where(mask, 0.5 * (normalized + 1) * (high - low + 1e-8) + low, normalized)
        return _result(raw, len(observations))


def build_policy(config):
    if config["type"] == "dream_jepa_dualview":
        from elaf.evaluation.dream_jepa import DreamJEPAPolicy
        return DreamJEPAPolicy(config)
    if config["type"] == "elaf_v2":
        from elaf.evaluation.elaf_v2 import ELAFV2InferencePolicy
        return ELAFV2InferencePolicy(config)
    if config["type"] == "dream_vla_official":
        from elaf.evaluation.official_dream import OfficialDreamPolicy
        return OfficialDreamPolicy(config)
    if config["type"] == "roboactualizer_action_only":
        return RoboActualizerActionOnlyPolicy(config)
    if config["type"] == "dream_vla_baseline":
        raise ValueError("Use official Dream-VLX evaluator for official trainer checkpoints; custom adapter is deprecated")
    if config["type"] == "dream_vla_custom_deprecated":
        return DreamVLABaselinePolicy(config)
    if config["type"] == "elaf":
        return ELAFInferencePolicy(config)
    raise ValueError("policy type must be roboactualizer_action_only, dream_vla_baseline, or elaf")

