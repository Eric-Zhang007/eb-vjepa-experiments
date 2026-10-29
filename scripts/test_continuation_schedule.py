import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'RoboActualizer/src/Roboactualizer/utils/continuation_schedule.py'
if not path.exists():
    path = Path(__file__).resolve().parents[1] / 'repos/RoboActualizer/src/Roboactualizer/utils/continuation_schedule.py'
spec = importlib.util.spec_from_file_location('schedule', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ScheduleTest(unittest.TestCase):
    def test_extension_and_restarted_step_agree(self):
        cfg = dict(start_step=21360, warmup_steps=200, initial_lr=2e-6, peak_lr=2e-5, final_lr=2e-6)
        f = lambda s: module.continuation_lr(s, 40000, cfg)
        self.assertAlmostEqual(f(21360), 2e-6)
        self.assertAlmostEqual(f(21560), 2e-5)
        self.assertAlmostEqual(f(40000), 2e-6)
        uninterrupted = [f(s) for s in range(21360, 40001)]
        resumed = [f(s) for s in range(30000, 40001)]
        self.assertEqual(uninterrupted[30000-21360:], resumed)
        self.assertTrue(all(a >= b for a, b in zip(resumed, resumed[1:])))


if __name__ == '__main__':
    unittest.main()
