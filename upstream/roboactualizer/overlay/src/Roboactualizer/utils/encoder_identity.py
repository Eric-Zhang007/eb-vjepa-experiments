"""Content identity for frozen-encoder caches and normalization statistics."""
import hashlib
from pathlib import Path


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_encoder_identity(metadata, expected, source):
    actual = metadata.get("encoder_checkpoint_sha256")
    if not expected or actual != expected:
        raise ValueError(f"Encoder identity mismatch in {source}: {actual!r} != {expected!r}")
