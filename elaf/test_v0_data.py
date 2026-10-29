"""Temporal and raw-cache contracts; no GPU, encoder, or dataset dependency."""

import copy
import unittest

from elaf.v0_data import make_temporal_pair, validate_raw_feature_pair, validate_temporal_pair


class TemporalPairTests(unittest.TestCase):
    def test_eight_actions_end_at_t_plus_eight(self):
        pair = make_temporal_pair("suite:episode", 5, 20, 19)
        self.assertEqual(pair["action_indices"], list(range(5, 13)))
        self.assertEqual(pair["current_frame_indices"], [2, 3, 4, 5])
        self.assertEqual(pair["future_frame_indices"], [10, 11, 12, 13])
        self.assertEqual(pair["future_end"], 13)
        self.assertTrue(pair["future_valid"])
        validate_temporal_pair(pair)

    def test_start_padding_repeats_only_first_frame(self):
        pair = make_temporal_pair("ep", 0, 9, 8)
        self.assertEqual(pair["current_frame_indices"], [0, 0, 0, 0])
        self.assertEqual(pair["current_frame_valid"], [False, False, False, True])
        self.assertEqual(pair["future_frame_indices"], [5, 6, 7, 8])
        self.assertTrue(pair["future_valid"])

    def test_valid_full_chunk_without_terminal_observation(self):
        pair = make_temporal_pair("ep", 0, 8, 8)
        self.assertTrue(all(pair["action_valid"]))
        self.assertFalse(pair["future_valid"])
        self.assertEqual(pair["requested_future_end"], 8)
        self.assertEqual(pair["future_frame_indices"], [4, 5, 6, 7])

    def test_rlds_dummy_action_is_excluded_by_explicit_count(self):
        pair = make_temporal_pair("ep", 5, 9, 8)
        self.assertEqual(pair["action_valid"], [True] * 3 + [False] * 5)
        self.assertEqual(pair["action_indices"], [5, 6, 7, 7, 7, 7, 7, 7])
        self.assertFalse(pair["future_valid"])
        self.assertEqual(pair["future_end"], 8)
        with self.assertRaises(ValueError):
            make_temporal_pair("ep", 8, 9, 8)

    def test_one_frame_episode_has_no_cross_episode_indices(self):
        pair = make_temporal_pair("ep", 0, 1, 1)
        self.assertEqual(pair["future_frame_indices"], [0] * 4)
        self.assertEqual(pair["action_indices"], [0] * 8)
        self.assertEqual(pair["action_valid"], [True] + [False] * 7)
        self.assertFalse(pair["future_valid"])

    def test_extra_observations_cannot_replace_missing_controls(self):
        pair = make_temporal_pair("ep", 0, 20, 2)
        self.assertFalse(pair["future_valid"])

    def test_false_future_claim_and_out_of_bounds_indices_rejected(self):
        pair = make_temporal_pair("ep", 0, 8, 8)
        for key, value in (("future_valid", True), ("future_frame_indices", [5, 6, 7, 8]),
                           ("action_valid", [True] * 7 + [False])):
            altered = {**pair, key: value}
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                validate_temporal_pair(altered)

    def test_invalid_counts_and_coordinates_rejected(self):
        for kwargs in (dict(t=-1), dict(t=8), dict(t=True), dict(num_actions=0),
                       dict(num_actions=10), dict(num_observations=0),
                       dict(horizon=0), dict(window=0), dict(episode_id="")):
            args = dict(episode_id="ep", t=0, num_observations=9, num_actions=8)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_temporal_pair(**(args | kwargs))


class RawFeatureIdentityTests(unittest.TestCase):
    def setUp(self):
        self.pair = make_temporal_pair("ep", 0, 9, 8)
        common = dict(episode_id="ep", encoder_sha256="a" * 64,
                      preprocessing_sha256="b" * 64, feature_stage="encoder_raw",
                      spatial_pooling="none", resampler_applied=False,
                      feature_shape=[392, 1024])
        self.current = dict(common, frame_indices=self.pair["current_frame_indices"])
        self.future = dict(common, frame_indices=self.pair["future_frame_indices"])

    def test_matching_raw_pair_accepted(self):
        validate_raw_feature_pair(self.pair, self.current, self.future)

    def test_mismatched_identity_window_and_processed_features_rejected(self):
        for key, value in (("encoder_sha256", "c" * 64),
                           ("preprocessing_sha256", "c" * 64),
                           ("episode_id", "other"), ("frame_indices", [4, 5, 6, 7]),
                           ("feature_stage", "resampler_output"),
                           ("spatial_pooling", "average"), ("resampler_applied", True),
                           ("feature_shape", [16, 1024]), ("encoder_sha256", "path.pt")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_raw_feature_pair(self.pair, self.current, self.future | {key: value})

    def test_missing_cache_provenance_rejected(self):
        altered = copy.deepcopy(self.future)
        del altered["resampler_applied"]
        with self.assertRaises(ValueError):
            validate_raw_feature_pair(self.pair, self.current, altered)


if __name__ == "__main__":
    unittest.main()
