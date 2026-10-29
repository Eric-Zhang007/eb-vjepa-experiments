"""CPU checkpoint plumbing unit test; not a real V-JEPA training acceptance run."""
import json
from pathlib import Path
import tempfile

from accelerate import Accelerator
from omegaconf import OmegaConf
import torch

from Roboactualizer.trainer_jointfm import JointPredictFMTrainer
from Roboactualizer.utils.samplers import ResumableEpochSampler


class UnitModel(torch.nn.Linear):
    action_only = True
    metadata = {"backbone": {"encoder_checkpoint_sha256": "unit-only"}}
    latent_norm_stats_meta = {"sha256": "unit-only"}

    def save_checkpoint(self, path, optimizer=None, step=None):
        torch.save({"model": self.state_dict(), "step": step}, path)


def main():
    torch.manual_seed(239)
    with tempfile.TemporaryDirectory() as temporary:
        trainer = JointPredictFMTrainer.__new__(JointPredictFMTrainer)
        trainer.accelerator = Accelerator(cpu=True)
        model = UnitModel(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        trainer.model, trainer.optimizer, trainer.scheduler = trainer.accelerator.prepare(model, optimizer, scheduler)
        trainer.cfg = OmegaConf.create({"model": {"action_only": True},
                                       "data": {"train": {"unit_fixture": True}}, "lr_scheduler_type": "unit_step"})
        trainer.batch_size = 2
        trainer.gradient_accumulation_steps = 1
        trainer.seed = 239
        trainer.max_steps = 10
        trainer.learning_rate = 0.001
        trainer.weight_decay = 0.01
        trainer.mixed_precision = "no"
        trainer._clip_meta = {}
        trainer.wandb_identity = {"id": "unit-id", "project": "unit", "workspace": None}
        trainer.state_dir = str(Path(temporary) / "state")
        trainer.weights_dir = str(Path(temporary) / "weights")
        Path(trainer.weights_dir).mkdir()
        trainer.train_sampler = ResumableEpochSampler(range(10), 239, 2, 1)
        loss = trainer.model(torch.ones(1, 2)).square().mean()
        trainer.accelerator.backward(loss)
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.optimizer.zero_grad()
        expected_parameters = {key: value.clone() for key, value in trainer.model.state_dict().items()}
        expected_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.global_step, trainer.epoch, trainer.batch_in_epoch = 1, 0, 1
        trainer.save_checkpoint()
        trainer.global_step, trainer.batch_in_epoch = 2, 2
        result = trainer.save_checkpoint()
        assert (Path(trainer.state_dir) / "previous/trainer_state.json").exists()
        expected_random = torch.rand(5)
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.zero_()
        trainer.optimizer.param_groups[0]["lr"] = 1.
        trainer.optimizer.state.clear()
        trainer.scheduler.scheduler.last_epoch = 99
        trainer.wandb_identity = None
        trainer.load_training_state(result["state_path"])
        assert trainer.wandb_identity["id"] == "unit-id"
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, expected_parameters[key])
        torch.testing.assert_close(torch.rand(5), expected_random)
        assert trainer.optimizer.param_groups[0]["lr"] == expected_lr
        assert trainer.optimizer.state and trainer.scheduler.scheduler.last_epoch == 1
        expected_indices = list(ResumableEpochSampler(range(10), 239, 2, 1))[4:]
        assert list(trainer.train_sampler) == expected_indices
        assert (trainer.global_step, trainer.batch_in_epoch) == (2, 2)
        committed = (Path(result["state_path"]) / "trainer_state.json").read_bytes()
        original_save = trainer.accelerator.save_state
        def failed_save(**kwargs):
            raise OSError("unit-test injected write failure")
        trainer.accelerator.save_state = failed_save
        try:
            trainer.save_checkpoint()
        except OSError:
            pass
        else:
            raise AssertionError("Injected write failure was swallowed")
        finally:
            trainer.accelerator.save_state = original_save
        assert (Path(result["state_path"]) / "trainer_state.json").read_bytes() == committed
        trainer.seed = 240
        try:
            trainer.load_training_state(result["state_path"])
        except ValueError as exc:
            assert "contract mismatch" in str(exc)
        else:
            raise AssertionError("Incompatible resume accepted")
        try:
            trainer.load_training_state(str(Path(temporary) / "incomplete"))
        except ValueError as exc:
            assert "complete trainer_state" in str(exc)
        else:
            raise AssertionError("Incomplete resume accepted")
    print(json.dumps({"trainer_state_unit": "PASS", "real_pipeline_acceptance": False}))


if __name__ == "__main__":
    main()
