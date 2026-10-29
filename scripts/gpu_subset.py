"""Explicit physical GPU ownership shared by downstream launchers."""
import argparse


def parse_gpus(value):
    try:
        ids = tuple(int(x) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated GPU indices") from exc
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i > 7 for i in ids):
        raise argparse.ArgumentTypeError("GPU IDs must be unique indices in 0..7")
    return ids


def gpu_string(ids):
    return ",".join(map(str, ids))
