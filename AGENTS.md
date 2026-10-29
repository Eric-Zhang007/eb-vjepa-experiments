# New-cluster handoff

Read README.md, docs/RESTORE.md and provenance/training_handoff.json first.

## Model identities
- Continued-pretraining grid11k encoder is the only retained encoder. It is NOT full72h step10k or final103233.
- Baseline is the final 200K four-suite mixed LIBERO Dream-VLA fine-tune, not four official suite-specific models.
- Dream-JEPA final is optimizer step156500, distinct from old E-LAF 72-token Delta Flow and the small RoboActualizer-derived latent WM.
- Retired intermediate weights cannot be recreated just from their metrics. Preserve recorded results and label historical evidence.

## Work order
1. Use `python handoff.py init --workspace /absolute/scratch/eb-runtime --data-root /absolute/datasets/rlds`.
2. Run `sources`, inspect source receipts, then `env`. Use the allocated Linux GPU node, not a login-node GPU or another user's allocation.
3. Run `assets` for required resources. No embedded token is required for public assets. Credentials belong in normal CLI login/environment, never in source/config.
4. Run `doctor --gpu`; verify checkpoint files exist, not just empty HF repos. Missing uploaded files are an external dependency, not a reason to substitute random weights.
5. First evaluate the baseline with `--episodes 2 --batch-size 1`, then the final Dream-JEPA. Preserve 4 denoising steps, execute8, image preprocessing, normalization and instruction policy.
6. Before fresh Dream-JEPA training regenerate the cache with `cache`; source/feature cache must match grid11k and paired camera preprocessing. A full cache is large (~hundreds of GiB).

## Exact resume at 156500
The portable source intentionally rewrites absolute paths and makes the training-cache index lazy for evaluation. This changes the original source/config contract. Do NOT disable resume validation or claim exact resume after passing the old checkpoint to the portable trainer. See docs/RESTORE.md. Prefer an explicit audited relocation or old-path container/bind-mount layout; preserve optimizer/scheduler, six-rank RNG/iterator states, data order and recorded source identities. Rebuilt cache manifests may have different paths/timestamps; compare contents and keep original provenance.

## Boundaries
- `sources` reconstructs runtime trees from pinned upstream commits and patches; never edit original research snapshots to hide a mismatch.
- No automatic training, GPU watchers, deletion or visibility changes during setup.
- Raw metrics stay on HF/local backup; GitHub contains configs, summaries and source.
- CPU handoff tests are not GPU, simulator or model-quality tests. Report the actual validation level.
