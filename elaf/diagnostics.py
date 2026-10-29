"""Fixed-set representation diagnostics; no universal collapse decision thresholds.

Inputs are *unscaled* shared-resampler outputs [samples, slots, channels].
All reductions use FP32. Variance is population variance across samples.
Effective rank uses covariance eigenvalues (squared singular values).
The caller must preserve sample identities, preprocessing, and initial RMS.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


def _validate(z: Tensor) -> Tensor:
    if z.ndim != 3 or z.shape[0] < 2 or min(z.shape[1:]) < 1:
        raise ValueError("Expected [K, M, D] with K >= 2 and positive M, D")
    z = z.detach().float()
    if not torch.isfinite(z).all():
        raise ValueError("Representation contains nonfinite values")
    return z


def _rank(centered: Tensor) -> Tensor:
    # Use the smaller Gram, never a flattened MD-by-MD covariance matrix.
    k, d = centered.shape
    scale = centered.abs().max()
    if scale == 0:
        return centered.new_zeros(())
    x = centered / scale
    gram = x @ x.T if k <= d else x.T @ x
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    # Remove only eigensolver roundoff relative to the leading eigenvalue.
    tolerance = eigenvalues.max() * torch.finfo(torch.float32).eps * min(k, d)
    eigenvalues = eigenvalues[eigenvalues > tolerance]
    if eigenvalues.numel() == 0:
        return centered.new_zeros(())
    p = eigenvalues / eigenvalues.sum()
    return (-(p * p.log()).sum()).exp()


def _center(z: Tensor) -> Tensor:
    # Subtract an anchor first so identical non-binary floats center to exact 0.
    shifted = z - z[:1]
    return shifted - shifted.mean(dim=0, keepdim=True)


def _summary(z: Tensor, representation_rank_cap: int | None) -> dict[str, Tensor]:
    centered = _center(z)
    variance = centered.square().mean(dim=(0, 2))
    ranks = torch.stack([_rank(centered[:, m]) for m in range(z.shape[1])])
    global_rank = _rank(centered.flatten(1))
    result = {
        "slot_variance": variance,
        "slot_erank": ranks,
        "slot_erank_norm": ranks / min(z.shape[0] - 1, z.shape[2]),
        "global_erank": global_rank,
        "global_erank_norm": global_rank / min(z.shape[0] - 1, z.shape[1] * z.shape[2]),
    }
    if representation_rank_cap is not None:
        cap = min(z.shape[0] - 1, z.shape[2], representation_rank_cap)
        result["slot_structural_rank_cap"] = z.new_tensor(cap)
        result["slot_erank_structural_norm"] = ranks / cap
    return result


@torch.no_grad()
def slot_linear_cka(z: Tensor, *, variance_floor: float = 0.0) -> tuple[Tensor, Tensor]:
    """Return M-by-M CKA and valid-slot mask; invalid rows/columns are NaN.

    variance_floor is an explicitly recorded absolute numerical validity floor
    in squared latent units, not a universal collapse threshold. Default rejects
    zero variance; set a fixed positive floor to also reject near-zero slots.
    Grams are normalized before pairwise products to avoid scale underflow.
    """
    z = _validate(z)
    if not math.isfinite(variance_floor) or variance_floor < 0:
        raise ValueError("variance_floor must be finite and nonnegative")
    x = _center(z)
    valid = x.square().mean(dim=(0, 2)) > variance_floor
    grams = []
    for m in range(z.shape[1]):
        if valid[m]:
            y = x[:, m] / x[:, m].abs().max()
            gram = y @ y.T
            grams.append((gram / torch.linalg.vector_norm(gram)).flatten())
        else:
            grams.append(z.new_zeros(z.shape[0] ** 2))
    grams = torch.stack(grams)
    cka = (grams @ grams.T).clamp(0, 1)
    cka[~(valid[:, None] & valid[None, :])] = float("nan")
    return cka, valid


@torch.no_grad()
def representation_diagnostics(
    current: Tensor,
    future: Tensor,
    *,
    initial_delta_rms: float | None = None,
    variance_floor: float = 0.0,
    representation_rank_cap: int | None = 256,
) -> dict:
    """Compute standard diagnostics, scalar W&B metrics, and per-slot tensors.

    None initial_delta_rms means step-zero calibration; persist the returned
    initial_delta_rms and pass it unchanged subsequently. A zero baseline or
    zero current-state energy produces NaN ratios instead of false reassurance.
    CKA summaries exclude invalid slots and diagonal entries.
    """
    if representation_rank_cap is not None and (
        isinstance(representation_rank_cap, bool)
        or not isinstance(representation_rank_cap, int)
        or representation_rank_cap < 1
    ):
        raise ValueError("representation_rank_cap must be a positive integer or None")
    current, future = _validate(current), _validate(future)
    if current.shape != future.shape or current.device != future.device:
        raise ValueError("Current and future must have identical shape and device")
    delta = future - current
    if not torch.isfinite(delta).all():
        raise ValueError("Delta contains nonfinite values")
    z_stats, f_stats, d_stats = (_summary(x, representation_rank_cap) for x in (current, future, delta))
    cka, valid = slot_linear_cka(current, variance_floor=variance_floor)
    off_diagonal = ~torch.eye(current.shape[1], dtype=torch.bool, device=current.device)
    cka_values = cka[off_diagonal & torch.isfinite(cka)]
    rms = delta.square().mean().sqrt().item()
    initial = rms if initial_delta_rms is None else float(initial_delta_rms)
    if not math.isfinite(initial) or initial < 0:
        raise ValueError("initial_delta_rms must be finite and nonnegative")
    energy = current.square().mean().item()
    metrics = {
        "monitor/delta/rms_raw": rms,
        "monitor/delta/rms_vs_initial": rms / initial if initial > 0 else float("nan"),
        "monitor/delta/energy_relative": rms ** 2 / energy if energy > 0 else float("nan"),
        "monitor/z/slot_cka_mean": cka_values.mean().item() if cka_values.numel() else float("nan"),
        "monitor/z/slot_cka_p90": torch.quantile(cka_values, .9).item() if cka_values.numel() else float("nan"),
        "monitor/z/cka_valid_slots": valid.sum().item(),
    }
    for prefix, stats in (("z", z_stats), ("future_z", f_stats), ("delta", d_stats)):
        for name in ("global_erank", "global_erank_norm"):
            metrics[f"monitor/{prefix}/{name}"] = stats[name].item()
        for name, values in (("slot_var", stats["slot_variance"]), ("slot_erank", stats["slot_erank"])):
            metrics[f"monitor/{prefix}/{name}_mean"] = values.mean().item()
            metrics[f"monitor/{prefix}/{name}_min"] = values.min().item()
            metrics[f"monitor/{prefix}/{name}_median"] = torch.quantile(values, .5).item()
        if representation_rank_cap is not None:
            metrics[f"monitor/{prefix}/slot_structural_rank_cap"] = stats["slot_structural_rank_cap"].item()
            values = stats["slot_erank_structural_norm"]
            for suffix, value in (("mean", values.mean()), ("min", values.min()), ("median", torch.quantile(values, .5))):
                metrics[f"monitor/{prefix}/slot_erank_structural_norm_{suffix}"] = value.item()
    return {
        "metrics": metrics,
        "current": z_stats, "future": f_stats, "delta": d_stats,
        "slot_cka": cka, "cka_valid_slots": valid,
        "initial_delta_rms": initial, "variance_floor": variance_floor,
        "representation_rank_cap": representation_rank_cap,
        "num_samples": current.shape[0],
        "effective_rank_definition": "entropy of normalized covariance eigenvalues",
    }
