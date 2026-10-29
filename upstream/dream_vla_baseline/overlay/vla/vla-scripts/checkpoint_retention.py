"""Transactional latest/best retention around the unchanged official serializer."""
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import uuid


def should_save(update, interval, maximum):
    return update > 0 and (update % interval == 0 or update >= maximum)


def publish(staging, root, metadata):
    staging, root = Path(staging), Path(root)
    if not math.isfinite(metadata["selection_loss"]):
        raise ValueError("checkpoint selection loss must be finite")
    latest, best = root / "latest", root / "best"
    previous = set()
    for pointer in (latest, best):
        if pointer.exists():
            if not pointer.is_symlink():
                raise ValueError(f"Refusing to replace unmanaged directory {pointer}")
            previous.add(pointer.resolve())
    best_loss = float("inf")
    if best.exists():
        best_loss = json.loads((best / "checkpoint_metadata.json").read_text())["selection_loss"]
    improved = metadata.get("best_candidate", True) and metadata["selection_loss"] < best_loss
    metadata = dict(metadata, best_at_save=improved, best_selection_loss=metadata["selection_loss"] if improved else (best_loss if math.isfinite(best_loss) else None),
                    selection_criterion="minimum cross-rank mean training loss over microbatches since previous save; candidates only at save points")
    (staging / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2))
    generation = root / ("generation_" + str(metadata["optimizer_update"]) + "_" + uuid.uuid4().hex)
    staging.rename(generation)
    for pointer in ([best, latest] if improved else [latest]):
        temporary = root / (".pointer_" + uuid.uuid4().hex)
        temporary.symlink_to(generation.name, target_is_directory=True)
        temporary.replace(pointer)
    retained = {p.resolve() for p in (latest, best) if p.exists()}
    for old in previous - retained:
        if old.parent != root.resolve() or not old.name.startswith("generation_"):
            raise ValueError(f"Refusing to remove unmanaged checkpoint {old}")
        shutil.rmtree(old)
    return metadata


def save_retained(save_function, cfg, run_dir, update, window_sum_count, window_start, dist, distributed_state, training_state=None, best_candidate=True, **kwargs):
    # Every rank receives identical reduced values and participates in official barriers.
    totals = window_sum_count.clone()
    dist.all_reduce(totals)
    mean = (totals[0] / totals[1]).item()
    root = Path(run_dir) / "checkpoints"
    staging_names = [f".saving_{update}_{uuid.uuid4().hex}" if distributed_state.is_main_process else None]
    dist.broadcast_object_list(staging_names, src=0)
    staging = root / staging_names[0]
    if distributed_state.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            raise FileExistsError(staging)
    dist.barrier()
    save_function(cfg=replace(cfg, save_latest_checkpoint_only=True), run_dir=staging,
                  log_step=update, distributed_state=distributed_state, **kwargs)
    if training_state is not None:
        training_state(staging)
    dist.barrier()
    status = [None]
    if distributed_state.is_main_process:
        try:
            status[0] = {"metadata": publish(staging, root, dict(optimizer_update=update,
                selection_loss=mean, window_start_update=window_start, window_end_update=update,
                global_microbatch_count=int(totals[1].item()), world_size=dist.get_world_size(),
                best_candidate=best_candidate,
                resume_semantics="full optimizer scheduler per-rank RNG and iterator state" if training_state is not None else "official weight checkpoint; optimizer and RNG not saved"))}
        except Exception as error:
            status[0] = {"error": repr(error)}
    dist.broadcast_object_list(status, src=0)
    if "error" in status[0]:
        raise RuntimeError(status[0]["error"])
    return status[0]["metadata"]
