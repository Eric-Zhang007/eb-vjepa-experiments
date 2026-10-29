"""Episode-local temporal contracts for E-LAF v0, with no encoder or data I/O.

Checklist: explicit valid-action count; t+H future endpoint; bounded padding;
raw-feature identity validation; boundary and tampering tests.
"""

from collections.abc import Mapping
import re


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def make_temporal_pair(episode_id, t, num_observations, num_actions,
                       horizon=8, window=4):
    """Return indices and validity, never indices into a neighboring episode.

    ``num_actions`` counts valid controls, not RLDS rows: callers must exclude a
    terminal dummy action explicitly. Observation i is the state before action
    i; executing actions [t, t+H) reaches observation t+H. A missing endpoint
    gets a same-episode filler window and ``future_valid=False``. This filler
    must not enter a future-target loss or representation diagnostics.

    Invalid action positions repeat the last valid action for safe gathering;
    only ``action_valid`` determines which positions contribute to the loss.
    Window masks distinguish real history from repeated frame-zero padding.
    """
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a nonempty string")
    for name, value, minimum in (("t", t, 0),
                                 ("num_observations", num_observations, 1),
                                 ("num_actions", num_actions, 1),
                                 ("horizon", horizon, 1), ("window", window, 1)):
        _integer(name, value, minimum)
    if num_actions > num_observations:
        raise ValueError("Every valid action must have its starting observation")
    if t >= num_actions:
        raise ValueError("t must have at least one valid action")

    def frame_window(end):
        indices = list(range(end - window + 1, end + 1))
        return [max(i, 0) for i in indices], [i >= 0 for i in indices]

    requested_end = t + horizon
    future_valid = requested_end < num_observations and requested_end <= num_actions
    # An endpoint beyond available controls is not a valid H-action transition,
    # even if additional observations happen to exist.
    future_end = requested_end if future_valid else num_observations - 1
    current_indices, current_mask = frame_window(t)
    future_indices, future_mask = frame_window(future_end)
    return {
        "episode_id": episode_id,
        "frame_index": t,
        "num_observations": num_observations,
        "num_actions": num_actions,
        "horizon": horizon,
        "window": window,
        "current_end": t,
        "requested_future_end": requested_end,
        "future_end": future_end,
        "current_frame_indices": current_indices,
        "current_frame_valid": current_mask,
        "future_frame_indices": future_indices,
        "future_frame_valid": future_mask,
        "future_valid": future_valid,
        "action_indices": [min(i, num_actions - 1) for i in range(t, requested_end)],
        "action_valid": [i < num_actions for i in range(t, requested_end)],
    }


def validate_temporal_pair(pair):
    """Reject altered provenance, including falsely claiming a padded future."""
    if not isinstance(pair, Mapping):
        raise ValueError("temporal pair must be a mapping")
    required = ("episode_id", "frame_index", "num_observations", "num_actions",
                "horizon", "window")
    if any(key not in pair for key in required):
        raise ValueError("temporal pair is missing source counts or window identity")
    expected = make_temporal_pair(*(pair[key] for key in required))
    for key, value in expected.items():
        if key not in pair or pair[key] != value:
            raise ValueError(f"Temporal provenance mismatch: {key}")
    return pair


def validate_raw_feature_pair(pair, current, future):
    """Validate cache descriptors, not tensor content, before trainable resampling.

    Each descriptor must declare encoder/checkpoint SHA256, preprocessing
    SHA256 (including the *resolved shared augmentation parameters*), raw stage,
    no pooling/resampling, [N,D] shape, episode and exact window frame indices.
    This function cannot prove those declarations match an encoder forward;
    cache preparation must independently verify and hash its actual artifacts.
    """
    validate_temporal_pair(pair)
    for name, descriptor in (("current", current), ("future", future)):
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"{name} feature descriptor must be a mapping")
        if descriptor.get("feature_stage") != "encoder_raw":
            raise ValueError(f"{name} must contain raw encoder features")
        if descriptor.get("spatial_pooling") != "none" or descriptor.get("resampler_applied") is not False:
            raise ValueError(f"{name} must not contain pooled or resampled features")
        for key in ("encoder_sha256", "preprocessing_sha256"):
            identity = descriptor.get(key)
            if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
                raise ValueError(f"{name} requires a SHA256 {key}")
        if descriptor.get("episode_id") != pair["episode_id"]:
            raise ValueError(f"{name} features belong to another episode")
        if descriptor.get("frame_indices") != pair[f"{name}_frame_indices"]:
            raise ValueError(f"{name} feature frame indices do not match the window")
        shape = descriptor.get("feature_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"{name} feature_shape must be [N,D]")
        for value in shape:
            _integer(f"{name} feature dimension", value, 1)
    for key in ("encoder_sha256", "preprocessing_sha256"):
        if current[key] != future[key]:
            raise ValueError(f"Current/future {key} mismatch")
    if tuple(current["feature_shape"]) != tuple(future["feature_shape"]):
        raise ValueError("Current/future raw feature shapes must match")
    return pair
