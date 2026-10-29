"""CPU contract tests without external checkpoints or simulator data."""
import tempfile
from pathlib import Path

import torch
from Roboactualizer.models.roboactualizer.backbone import Backbone
from Roboactualizer.models.roboactualizer.policy import FlowPolicy


class TinyBackbone(Backbone):
    embed_dim, num_patches, patch_grid = 16, 4, (2, 2)

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 16)

    def encode_clip_last_group(self, clips):
        return self.proj(clips.mean((1, 3, 4))).unsqueeze(1).expand(-1, 4, -1)


def make_model():
    return FlowPolicy(
        TinyBackbone(), chunk_size=3, future_size=0, video_freq_ratio=4,
        action_dim=2, text_dim=8, proprio_dim=3, d_model=32, num_layers=2,
        num_heads=2, dim_head=16, mlp_dim=64, freq_dim=16,
        action_only=True, image_predict_weight=0,
    )


def main():
    torch.set_num_threads(2)
    torch.manual_seed(239)
    model = make_model().train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    batch = dict(
        z0=torch.randn(2, 4, 16), text_context=torch.randn(2, 5, 8),
        text_mask=torch.ones(2, 5, dtype=torch.bool), proprio=torch.randn(2, 3),
        action_chunk=torch.randn(2, 3, 2),
        action_is_pad=torch.tensor([[False, False, True], [False, False, False]]),
    )
    for _ in range(2):
        optimizer.zero_grad()
        loss, logs = model.training_loss(batch)
        assert torch.isfinite(loss) and logs["image_loss"] == 0
        loss.backward()
        assert all(p.grad is None for p in model.backbone.parameters())
        assert model.action_context_builder.text_embedding.net[0].weight.grad is not None
        optimizer.step()
    assert not any(k.startswith("image_decoder.") for k in model.state_dict())
    assert model.image_scheduler is None
    try:
        model.training_loss({**batch, "z_future": torch.empty(0)})
    except ValueError:
        pass
    else:
        raise AssertionError("Future latents accepted")
    pixels = torch.randn(2, 4, 3, 8, 8)
    online_batch = {k: v for k, v in batch.items() if k != "z0"}
    online_loss, _ = model.training_loss({**online_batch, "pixels": pixels[:, None]})
    assert torch.isfinite(online_loss)
    model.eval()
    kwargs = dict(
        obs_pixels=pixels, text_context=batch["text_context"], text_mask=batch["text_mask"],
        proprio=batch["proprio"], num_inference_steps=2, use_cuda_graph=False,
        use_adaln_cache=False, use_cross_kv_cache=False,
    )
    torch.manual_seed(5)
    plain = model.generate(**kwargs, use_kv_cache=False)
    torch.manual_seed(5)
    cached = model.generate(**kwargs, use_kv_cache=True)
    torch.testing.assert_close(plain, cached, atol=1e-5, rtol=1e-4)
    with tempfile.TemporaryDirectory() as directory:
        policy_path = Path(directory) / "policy.pt"
        model.save_checkpoint(str(policy_path), step=2)
        make_model().load_checkpoint(str(policy_path))
        from Roboactualizer.utils.encoder_identity import checkpoint_sha256
        identity_fixture = Path(directory) / "encoder_identity_fixture.bin"
        identity_fixture.write_bytes(b"unit-test-identity-only")
        model.metadata = {"backbone": {"encoder_checkpoint_sha256": checkpoint_sha256(identity_fixture)}}
        model.save_checkpoint(str(policy_path), step=2)
        guarded = make_model()
        guarded.backbone.ckpt_path = str(identity_fixture)
        guarded.load_checkpoint(str(policy_path))
        identity_fixture.write_bytes(b"wrong-unit-test-identity")
        try:
            guarded.load_checkpoint(str(policy_path))
        except ValueError as exc:
            assert "Encoder identity mismatch" in str(exc)
        else:
            raise AssertionError("Wrong encoder accepted for action-only checkpoint")
        path = Path(directory) / "resume.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng": torch.get_rng_state(), "step": 2}, path)
        state = torch.load(path, weights_only=True)
        restored = make_model().eval()
        restored.load_state_dict(state["model"], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.trainable_parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        assert state["step"] == 2
        torch.manual_seed(5)
        torch.testing.assert_close(plain, restored.generate(**kwargs, use_kv_cache=False))
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=libero_action_only"])
        assert cfg.model.action_only and cfg.model.future_size == 0
        assert cfg.data.train.action_only
    print("PASS: action-only backward, frozen encoder, language gradients, no future inputs/head, "
          "pixel path, cached inference parity, strict state/optimizer roundtrip")


if __name__ == "__main__":
    main()
