"""Grid-only archive provenance and bounded RAM/disk cache placement."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_archived_source(audit_path, encoder, step, source):
    audit = json.loads(audit_path.read_text())
    receipt = json.loads(audit_path.with_name(f"archive_receipt_{step:06d}.json").read_text())
    if (audit.get("status") != "PASS" or audit.get("step") != step
            or not audit.get("export_equals_source_ema") or not audit.get("finite")
            or audit.get("tensors") != 302
            or Path(audit["encoder"]).resolve() != encoder.resolve()
            or Path(audit["source_checkpoint"]).resolve() != source.resolve()
            or digest(encoder) != audit["encoder_sha256"]
            or receipt.get("phase") != "complete" or receipt.get("step") != step
            or receipt.get("local_sha256_verified") is not True
            or receipt.get("source_sha256") != audit.get("source_sha256")):
        raise RuntimeError("Archived checkpoint provenance or EMA identity mismatch")
    return audit


def prepare_hybrid_storage(root, cache, step):
    expected = root / f"experiments/action_only/ctp_grid20k_search/encoder_{step:06d}/cache_retry1/cache_final_{step:06d}_exclude_goal82"
    prior10k = root / "experiments/action_only/ctp_full_010000_handoff/cache_retry1/cache_final_010000_exclude_goal82"
    rebuilding_prior = step == 10000 and cache == prior10k
    if rebuilding_prior:
        expected = prior10k
    if cache != expected or cache.resolve() != expected:
        raise RuntimeError("Hybrid cache is restricted to the new grid search")
    reference = root / "experiments/action_only/ctp_full_050000_handoff/cache_retry1/cache_final_050000_exclude_goal82"
    counts = {p.parent.name: sum(json.loads(p.read_text())["episodes"].values())
              for p in reference.glob("*/latents_meta.json")}
    suites = [f"libero_{s}_no_noops_1.0.0_lerobot" for s in ("spatial", "object", "goal")]
    if len(counts) != 4 or sum(counts.values()) != 273336 or any(s not in counts for s in suites):
        raise RuntimeError("Missing measured storage reference metadata")
    # Previous identical-shape float16 cache measured 219.45 GB; keep a 10% margin.
    estimate = int(220_000_000_000 * sum(counts[s] for s in suites) / 273336)
    reservation = int(estimate * 1.1)
    ram = Path(f"/dev/shm/eb_vjepa_grid_{os.getuid()}/encoder_{step:06d}")
    if rebuilding_prior:
        ram = Path(f"/dev/shm/eb_vjepa_extend_{os.getuid()}/encoder_010000")
    if ram.resolve() != ram:
        raise RuntimeError("Refusing redirected RAM cache")
    existing_ram = sum(p.stat().st_size for p in ram.glob("*/episode_*.pt"))
    needed = max(0, reservation - existing_ram)
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(meminfo["MemAvailable"].split()[0]) * 1024
    if shutil.disk_usage("/dev/shm").free < needed + 40_000_000_000 or available < needed + 100_000_000_000:
        raise RuntimeError("Insufficient RAM/tmpfs headroom for bounded hybrid cache")
    cache.mkdir(parents=True, exist_ok=True)
    ram.mkdir(parents=True, exist_ok=True, mode=0o700)
    for suite in suites:
        target = ram / suite
        target.mkdir(exist_ok=True)
        link = cache / suite
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError("Unexpected cache symlink")
        elif link.exists():
            if not rebuilding_prior or any(link.glob("episode_*.pt")):
                raise RuntimeError("Expected retired metadata-only directory")
            preserved = cache.parent / f"preserved_before_extension_{suite}"
            if preserved.exists():
                raise RuntimeError("Prior metadata backup already exists; inspect before retry")
            for meta in link.glob("*.json"):
                shutil.copy2(meta, target / meta.name)
            link.rename(preserved)
            link.symlink_to(target, target_is_directory=True)
        else:
            link.symlink_to(target, target_is_directory=True)
    (cache / "storage_plan.json").write_text(json.dumps({
        "ram_root": str(ram), "suites": suites, "ram_expected_bytes": estimate,
        "ram_reserved_bytes": reservation, "reference": str(reference),
        "semantics": "Same float16 payloads and metadata; placement only. Rebuild after reboot."}, indent=2))
    return estimate


def retire_grid_payloads(run, step, cancelled_replacement=False):
    state = json.loads((run / "continuation_status.json").read_text())
    if cancelled_replacement:
        if step != 10000 or state["phase"] not in ("failed", "cancelled", "cancelled_by_user"):
            raise RuntimeError("Only the cancelled 10k replacement may be retired")
        for process in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                if str(run).encode() in process.read_bytes():
                    raise RuntimeError("Cancelled cache still has an active consumer")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    elif state["phase"] != "complete":
        raise RuntimeError("Cannot retire unfinished trial cache")
    marker = run / "cache_retirement.json"
    if marker.exists() and json.loads(marker.read_text())["phase"] == "complete":
        return
    cache = run / f"cache_retry1/cache_final_{step:06d}_exclude_goal82"
    plan = json.loads((cache / "storage_plan.json").read_text())
    ram = Path(plan["ram_root"])
    expected_ram = Path(f"/dev/shm/eb_vjepa_grid_{os.getuid()}/encoder_{step:06d}")
    if (ram != expected_ram or ram.resolve() != ram or run.resolve() != run
            or cache.resolve() != cache or run.name != f"encoder_{step:06d}"
            or run.parent.name != "ctp_grid20k_search"):
        raise RuntimeError("Unexpected grid cache ownership")
    files = list(cache.glob("*/episode_*.pt"))
    for path in files:
        resolved = path.resolve()
        allowed_ram = path.parent.name in plan["suites"] and resolved.parent == ram / path.parent.name
        allowed_disk = not path.parent.is_symlink() and resolved.is_relative_to(cache.resolve())
        if path.is_symlink() or not re.fullmatch(r"episode_\d+\.pt", path.name) or not (allowed_ram or allowed_disk):
            raise RuntimeError("Unsafe cache payload")
    for suite in plan["suites"]:
        destination = run / "preserved_ram_cache_metadata" / suite
        destination.mkdir(parents=True, exist_ok=True)
        for meta in (ram / suite).glob("*.json"):
            shutil.copy2(meta, destination / meta.name)
    report = {"paths": [str(p) for p in files], "bytes": sum(p.stat().st_size for p in files),
              "scope": "Only generated grid trial payloads; all models and metadata retained"}
    marker.write_text(json.dumps({**report, "phase": "removing"}, indent=2))
    for path in files:
        path.unlink()
    marker.write_text(json.dumps({**report, "phase": "complete"}, indent=2))
