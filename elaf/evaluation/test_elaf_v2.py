import numpy as np
import torch

from elaf.evaluation.elaf_v2 import history_clips, unnormalize_action
from elaf.v0_cache import normalization


def test_history_is_causal_and_pads_only_episode_start():
    frames = [{'image':np.full((2,2,3), i, np.uint8)} for i in range(6)]
    short = dict(image=frames[1]['image'], history=frames[:2])
    full = dict(image=frames[-1]['image'], history=frames)
    clips = history_clips([short, full])
    assert clips[:, :, 0, 0, 0].tolist() == [[0,0,0,1], [2,3,4,5]]


def test_training_normalization_round_trip_and_gripper():
    stats = dict(q01=[-2]*6+[0], q99=[2]*6+[1], min=[-3]*6+[0],
                 max=[3]*6+[1], mask=[True]*6+[False])
    raw = np.array([[.2,-.4,.8,1.,-.8,.1,0], [.1]*6+[1]], np.float32)
    restored = unnormalize_action(torch.from_numpy(normalization(raw, stats)), stats)
    np.testing.assert_allclose(restored.numpy(), raw, atol=1e-6)


def test_constant_training_dimensions_are_restored():
    stats = dict(q01=[2]*7, q99=[2]*7, min=[2]*7, max=[2]*7)
    assert torch.equal(unnormalize_action(torch.randn(8,7), stats), torch.full((8,7), 2.))
