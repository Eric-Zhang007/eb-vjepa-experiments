# Official Dream-VLA mixed LIBERO iterator verification

Verified on Autolab, 2026-10-16. No GPUs were used.

The authoritative test explicitly passed `--dataset libero_mix4` and asserted action chunk 8, proprio 8, and bounds_q99 normalization. Initial tests in `rlds_resume100k` selected the official BRIDGE default (5-action chunks); those results are superseded by the LIBERO results below.

- Mixture `libero_mix4` contains Spatial, Object, Goal and LIBERO-10 with probability 0.25 each. Dataset-size reweighting is disabled for this mixture.
- Official RLDS action chunks, image transformations and augmentation ranges are retained. Augmentation seeds are now stateless `(seed + rank, stream_index)`, so parallel scheduling does not consume an uncheckpointed random generator. The stream index is checkpointed. Unused random goal relabeling is disabled for this mixture.
- Native TensorFlow iterator checkpoints retain the shuffle buffer, interleave cursors and transformation pipeline. No external-state ignore policy is used. PyTorch DataLoader must use `num_workers=0`.
- Production shuffle buffer 100,000: save succeeded, then an independently started Python process restored and matched all fields of the next **128 samples exactly**, including both augmented images, actions, proprioception and language.
- Rank 1 independently produced a different first sample from rank 0, with the same data statistics. Each rank uses seed `239 + rank`.
- One rank's iterator snapshot occupied **6,338,429,232 bytes**; saving took **25.19 s**, restoring plus consuming 128 samples took **34.53 s**. Six ranks require approximately **38.03 GB** of iterator state per saved checkpoint, excluding model and optimizer state. Concurrent six-rank save throughput has not been measured by this test.
- Canonical JSON statistics before/after the official statistics writer, and independent rank 1 construction, all hash to `c2bf045a1d214bb352c73d0beb905dd0008e63443b7d18f9df111ef122a3ca45`. The full trainer resume contract issue is tracked separately by the training-state owner.

Test: `scripts/test_dream_rlds_resume.py`, modes `save`, `restore`, `rank`. Whole-object pickle hashes are not used because cross-process object aliasing can change pickle bytes without changing any tensor value; each actual field is hashed canonically instead.

Remote evidence: `experiments/dream_vla_mix4/rlds_resume100k_libero_{save,restore,rank}.log`, directory `experiments/dream_vla_mix4/rlds_resume100k_libero/`. Source implementation: `Dream-VLX/vla/prismatic/vla/datasets/datasets.py`, `rlds/dataset.py`, `rlds/oxe/mixtures.py`.

This verifies the real data stream only. Full model, optimizer, scheduler and distributed state equivalence are covered by the separate official trainer interruption test.
