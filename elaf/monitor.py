"""Offline fixed-set E-LAF diagnostics. Never initializes a W&B run.

Cache schema v1: current_features/future_features [K,N,D], unique sample_ids,
and manifest with encoder_checkpoint_sha256 and preprocessing_sha256. Cache
contains frozen encoder outputs, never learned resampler outputs. Snapshot
identity binds the entire cache, including feature bytes, to the training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor, nn

from elaf.diagnostics import representation_diagnostics


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sha(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected lowercase SHA256 identity")


def monitor_manifest_sha256(cache: dict) -> str:
    """Verify fixed-cache schema and hash metadata, ordering, and raw features."""
    if cache.get("schema_version") != 1:
        raise ValueError("Unsupported fixed-cache schema")
    metadata = cache.get("manifest", {})
    for key in ("encoder_checkpoint_sha256", "preprocessing_sha256"):
        _check_sha(metadata.get(key))
    ids = cache.get("sample_ids")
    if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique strings for at least two pairs")
    identity = {"schema_version": 1, "manifest": metadata, "sample_ids": ids}
    shape = None
    for key in ("current_features", "future_features"):
        value = cache.get(key)
        if not isinstance(value, Tensor) or value.ndim != 3 or value.shape[0] != len(ids) or min(value.shape[1:]) < 1:
            raise ValueError(f"Invalid {key} shape")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {key} values")
        if shape is not None and value.shape != shape:
            raise ValueError("Current/future feature shapes differ")
        shape = value.shape
        raw = value.detach().cpu().contiguous()
        digest = hashlib.sha256(memoryview(raw.view(torch.uint8).numpy())).hexdigest()
        identity[key] = {"shape": list(raw.shape), "dtype": str(raw.dtype), "sha256": digest}
    return hashlib.sha256(_canonical(identity)).hexdigest()


def _atomic_write(path: Path, writer, *, immutable: bool = False) -> None:
    """Publish only closed, fsynced files; exclusive hard-link for snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temporary, path)  # Atomic exclusive publication; never overwrite.
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_resampler_snapshot(
    resampler: nn.Module,
    directory: str | Path,
    *,
    training_step: int,
    config: dict,
    parent_run_id: str,
    monitor_manifest_sha256: str,
) -> Path:
    """Training hook; call at step 0 and the configured diagnostic intervals."""
    if isinstance(training_step, bool) or not isinstance(training_step, int) or training_step < 0:
        raise ValueError("training_step must be a nonnegative integer")
    if not parent_run_id:
        raise ValueError("parent_run_id is required")
    _check_sha(monitor_manifest_sha256)
    _canonical(config)
    payload = {
        "schema_version": 1, "training_step": training_step, "config": config,
        "parent_run_id": parent_run_id, "monitor_manifest_sha256": monitor_manifest_sha256,
        "state_dict": {name: value.detach().cpu().clone() for name, value in resampler.state_dict().items()},
    }
    path = Path(directory) / f"resampler-step-{training_step:09d}.pt"
    _atomic_write(path, lambda stream: torch.save(payload, stream), immutable=True)
    return path


def discover_snapshots(directory: str | Path) -> list[Path]:
    """Temporary files and unfinished writers are deliberately invisible."""
    return sorted(path for path in Path(directory).glob("resampler-step-*.pt") if path.is_file())


def _json_metrics(metrics: dict) -> dict:
    invalid = [key for key, value in metrics.items() if not math.isfinite(value)]
    return {"metrics": {key: None if key in invalid else value for key, value in metrics.items()},
            "invalid_metric_fields": invalid}


@torch.no_grad()
def run_monitor_snapshot(
    resampler: nn.Module,
    cache: dict,
    snapshot: str | Path,
    output_directory: str | Path,
    *,
    batch_size: int = 32,
    variance_floor: float = 0.0,
    representation_rank_cap: int | None = 256,
    wandb_run=None,
) -> dict:
    """Load one complete snapshot, use persistent step-zero calibration, save.

    output_directory belongs to exactly one monitor set and parent training run.
    The caller owns this inference resampler; it is converted to FP32/eval mode.
    A supplied W&B run must be a separate diagnostics run, never the live trainer.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    snapshot, output_directory = Path(snapshot), Path(output_directory)
    if snapshot not in discover_snapshots(snapshot.parent):
        raise ValueError("Snapshot has not been atomically published")
    payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot schema")
    step = payload["training_step"]
    if not isinstance(step, int) or step < 0 or snapshot.name != f"resampler-step-{step:09d}.pt":
        raise ValueError("Snapshot filename/step mismatch")
    manifest_sha = monitor_manifest_sha256(cache)
    if payload["monitor_manifest_sha256"] != manifest_sha:
        raise ValueError("Fixed monitor set, encoder, or preprocessing identity changed")
    snapshot_sha = _sha256_file(snapshot)
    baseline_path = output_directory / "baseline.json"
    identity = {
        "monitor_manifest_sha256": manifest_sha,
        "parent_run_id": payload["parent_run_id"],
        "config_sha256": hashlib.sha256(_canonical(payload["config"])).hexdigest(),
        "variance_floor": variance_floor, "representation_rank_cap": representation_rank_cap,
    }
    baseline = None
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        if any(baseline.get(key) != value for key, value in identity.items()):
            raise ValueError("Step-zero baseline identity changed")
        if step == 0 and baseline["snapshot_sha256"] != snapshot_sha:
            raise ValueError("Step-zero snapshot changed")
    elif step != 0:
        raise ValueError("Run step-zero snapshot first to establish fixed baseline")
    resampler.load_state_dict(payload["state_dict"], strict=True)
    resampler.float().eval()
    device = next(resampler.parameters(), torch.empty(0)).device
    states = []
    for key in ("current_features", "future_features"):
        batches = []
        for start in range(0, len(cache["sample_ids"]), batch_size):
            features = cache[key][start:start + batch_size].to(device=device, dtype=torch.float32)
            batches.append(resampler(features).detach().float().cpu())
        states.append(torch.cat(batches, dim=0))
    result = representation_diagnostics(
        *states,
        initial_delta_rms=None if baseline is None else baseline["initial_delta_rms"],
        variance_floor=variance_floor, representation_rank_cap=representation_rank_cap,
    )
    if baseline is None:
        baseline = {**identity, "schema_version": 1, "training_step": 0,
                    "snapshot_sha256": snapshot_sha, "initial_delta_rms": result["initial_delta_rms"]}
        _atomic_write(baseline_path, lambda stream: stream.write(_canonical(baseline)), immutable=True)
    report = {
        "schema_version": 1, "training_step": step, **identity,
        "snapshot_sha256": snapshot_sha, "initial_delta_rms": baseline["initial_delta_rms"],
        "num_samples": result["num_samples"], **_json_metrics(result["metrics"]),
    }
    stem = f"diagnostics-step-{step:09d}"
    _atomic_write(output_directory / f"{stem}.pt", lambda stream: torch.save(result, stream))
    # JSON is the completion marker, published after the tensor artifact.
    _atomic_write(output_directory / f"{stem}.json", lambda stream: stream.write(_canonical(report)))
    if wandb_run is not None:
        log_to_wandb(wandb_run, result, training_step=step, parent_run_id=payload["parent_run_id"])
    return report


def log_to_wandb(run, result: dict, *, training_step: int, parent_run_id: str) -> None:
    """Explicit opt-in logging to caller-created separate diagnostics run."""
    if getattr(run, "id", None) == parent_run_id:
        raise ValueError("Use a separate diagnostics run; do not write to the live trainer run")
    import wandb
    run.define_metric("training_step")
    run.define_metric("monitor/*", step_metric="training_step")
    payload = {"training_step": training_step, **_json_metrics(result["metrics"])["metrics"]}
    payload["monitor/z/slot_var_histogram"] = wandb.Histogram(result["current"]["slot_variance"].cpu().numpy())
    cka = result["slot_cka"]
    payload["monitor/z/slot_cka_table"] = wandb.Table(
        columns=["slot_i", "slot_j", "cka"],
        data=[[i, j, cka[i, j].item() if torch.isfinite(cka[i, j]) else None]
              for i in range(cka.shape[0]) for j in range(cka.shape[1])],
    )
    run.log(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--variance-floor", type=float, default=0.0)
    args = parser.parse_args()
    from elaf.v0_model import ELAFV0Config, SharedResampler
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    config = ELAFV0Config(**snapshot["config"])
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    report = run_monitor_snapshot(
        SharedResampler(config).to(args.device), cache, args.snapshot, args.output_directory,
        batch_size=args.batch_size, variance_floor=args.variance_floor,
        representation_rank_cap=config.resampler_dim,
    )
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
