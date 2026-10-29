"""CPU regression guards with explicit unit fixtures, not pipeline acceptance."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
import torch

from Roboactualizer.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from Roboactualizer.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from Roboactualizer.models.wan22.helpers import loader
from Roboactualizer.trainer_jointfm import JointPredictFMTrainer


class Guards(unittest.TestCase):
    def test_pro_limits(self):
        # Extract the pure function without importing simulator/GPU entrypoint side effects.
        path = Path(__file__).resolve().parents[1] / "experiments/libero/eval_libero_single.py"
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_get_max_steps")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        limit = namespace["_get_max_steps"]
        for base, expected in (("libero_spatial", 400), ("libero_object", 400),
                               ("libero_goal", 400), ("libero_10", 700)):
            self.assertEqual(limit(base), expected)
            for suffix in ("lan", "object", "swap", "task"):
                self.assertEqual(limit(f"{base}_{suffix}"), expected)
        self.assertEqual(limit("libero_90"), 700)
        with self.assertRaises(ValueError):
            limit("libero_spatial_unknown")

    def test_outer_dataset_never_substitutes(self):
        dataset = RobotVideoDataset.__new__(RobotVideoDataset)
        dataset.action_only = True
        dataset._get = Mock(side_effect=FileNotFoundError("unit missing cache"))
        with patch("numpy.random.randint", side_effect=AssertionError("Random replacement")):
            with self.assertRaises(FileNotFoundError):
                dataset[7]
        dataset._get.assert_called_once_with(7)

    def test_inner_dataset_never_substitutes(self):
        dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
        dataset.strict_read = True
        class Broken:
            num_frames = 10
            def __getitem__(self, index):
                raise OSError(f"unit decode error {index}")
        dataset.multi_dataset = Broken()
        with patch("numpy.random.randint", side_effect=AssertionError("Random replacement")):
            with self.assertRaisesRegex(RuntimeError, "requested sample 7") as caught:
                dataset[7]
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_text_load_is_strict(self):
        model = torch.nn.Linear(2, 2)
        state = model.state_dict()
        registry = [{"model_hash": "unit", "model_name": "unit",
                     "model_class": lambda: torch.nn.Linear(2, 2)}]
        with patch.object(loader, "WAN22_MODEL_REGISTRY", registry), \
             patch.object(loader, "hash_model_file", return_value="unit"):
            with patch.object(loader, "load_state_dict", return_value=state):
                actual = loader._load_registered_model("unit", "unit", torch.float32, "cpu")
                torch.testing.assert_close(actual.weight, model.weight)
            for bad in ({"weight": state["weight"]}, {**state, "extra": torch.zeros(1)},
                        {**state, "weight": torch.zeros(1, 2)}):
                with patch.object(loader, "load_state_dict", return_value=bad):
                    with self.assertRaises(RuntimeError):
                        loader._load_registered_model("unit", "unit", torch.float32, "cpu")

    def test_wandb_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = JointPredictFMTrainer.__new__(JointPredictFMTrainer)
            trainer.output_dir = temporary
            trainer.resume = None
            trainer.wandb_identity = None
            trainer.wandb_enabled = True
            trainer.accelerator = SimpleNamespace(is_main_process=True)
            trainer.cfg = OmegaConf.create({"wandb": {"workspace": None, "project": "unit",
                                                       "name": "unit", "group": None, "mode": "online"}})
            fake = SimpleNamespace(util=SimpleNamespace(generate_id=lambda: "unit-id"), init=Mock())
            with patch.dict(sys.modules, {"wandb": fake}), \
                 patch("Roboactualizer.trainer_jointfm.uuid.uuid4", return_value=SimpleNamespace(hex="unit-id0")):
                trainer._init_wandb()
                self.assertEqual(fake.init.call_args.kwargs["resume"], "never")
                identity = json.loads((Path(temporary) / "wandb_run.json").read_text())
                self.assertEqual(identity["id"], "unit-id0")
                trainer.resume = temporary
                trainer.wandb_identity = identity
                trainer._init_wandb()
                self.assertEqual(fake.init.call_args.kwargs["resume"], "must")
                self.assertEqual(fake.init.call_args.kwargs["id"], "unit-id0")
                trainer.cfg.wandb.project = "wrong"
                with self.assertRaisesRegex(ValueError, "project/workspace"):
                    trainer._init_wandb()
                trainer.cfg.wandb.project = "unit"
                trainer.wandb_identity = None
                with self.assertRaisesRegex(ValueError, "requires run identity"):
                    trainer._init_wandb()
                trainer.resume = None
                with self.assertRaisesRegex(ValueError, "already has"):
                    trainer._init_wandb()


if __name__ == "__main__":
    unittest.main()
