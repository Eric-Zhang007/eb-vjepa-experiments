"""Unit checks; ELAF_POLICY_TEST_CONFIG enables actual checkpoint batch checks."""
import json
import os
import unittest

import numpy as np
import torch

from elaf.evaluation.policy import build_policy, initial_noise, validate_observations


class PolicyTests(unittest.TestCase):
    def observations(self):
        rng = np.random.default_rng(19)
        result = []
        for i in range(2):
            image = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            wrist = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            result.append(dict(image=image, proprio=np.zeros(8, dtype=np.float32),
                instruction="put the white mug on the left plate and put the yellow and white mug on the right plate",
                noise_seed=239 + i, history=[dict(image=image, wrist_image=wrist)]))
        return result

    def test_noise_batch_independence(self):
        observations = self.observations()
        state = torch.get_rng_state().clone()
        together = initial_noise(observations, (8, 7), "cpu", torch.float32)
        apart = torch.cat([initial_noise([x], (8, 7), "cpu", torch.float32) for x in observations])
        torch.testing.assert_close(together, apart, rtol=0, atol=0)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(together.shape, (2, 8, 7))

    def test_validation(self):
        observations = self.observations()
        validate_observations(observations)
        observations[0]["image"] = observations[0]["image"].astype(np.float32)
        with self.assertRaises(ValueError):
            validate_observations(observations)

    @unittest.skipUnless(os.getenv("ELAF_POLICY_TEST_CONFIG"), "requires explicit ready checkpoint config")
    def test_ready_checkpoint_batched_inference(self):
        with open(os.environ["ELAF_POLICY_TEST_CONFIG"]) as stream:
            config = json.load(stream)
        policy = build_policy(config)
        observations = self.observations()
        together = policy.predict(observations)
        apart = np.concatenate([policy.predict([x]) for x in observations])
        self.assertEqual(together.shape[0], 2)
        self.assertEqual(together.shape[-1], 7)
        self.assertTrue(np.isfinite(together).all())
        # BF16 GEMM is allowed small batch-shape rounding differences.
        np.testing.assert_allclose(together, apart, rtol=0.03, atol=0.02)
        repeat = policy.predict(observations)
        np.testing.assert_array_equal(together, repeat)
        print("POLICY_FINGERPRINT " + json.dumps(policy.fingerprint(), sort_keys=True), flush=True)
        print("BATCH_SINGLE_MAX_ABS " + str(float(np.max(np.abs(together - apart)))), flush=True)


if __name__ == "__main__":
    unittest.main()
