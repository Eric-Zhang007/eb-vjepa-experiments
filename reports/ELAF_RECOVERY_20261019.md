# E-LAF recovery trial, 2026-10-19

Authorized: safely pause original run and perform a short recovery experiment.

Original run `elaf_v1_grid11k_72token` paused at step 19287 with complete model,
optimizer, scheduler and six rank RNG/cursor files. Its latest and best checkpoint
remain in its own directory. Source before edits is archived remotely under
`experiments/elaf_recovery_source_before_20261019/elaf`.

Step 583 was not present in the original checkpoint directory or the searched
experiment backup directories. Earliest available full state was step 5000:
`generation_005000_f819a8cb3c004288b5d38d218eb7dd9f`.

New run: `experiments/elaf_recovery_from5000_20261019`, W&B `ifzfip50` in
`eb_vjepa`. This tests recovery from an already degraded model, not prevention of
the original step-1220 instability. New update axis starts at zero; data cursor
and per-rank RNG continue from parent step 5000. AdamW moments are reset and the
scheduler is new. The fork receipt records parent identity and state handling.

Configuration: `elaf/configs/recovery_5000.json`. GPU 0–5, batch 48, 1000 updates;
LoRA LR 2e-5, Resampler 5e-5, new modules 1e-4; linear warmup 200 updates then
constant LR; independent logical-group gradient clipping at 1.0; full FP32
Resampler and delta construction; backbone BF16. Loss weights, slots, heads,
data and augmentation unchanged. No EMA or SIGReg. Save at 500 and 1000.

Every ten updates log module-level preclip gradients, absolute and relative
parameter updates, rank-0 valid-element velocity RMS/target RMS/cosine/MSE and
zero-velocity MSE. Rank-0 debug batch metrics are not global validation metrics.
Fixed-set representation monitoring at 0/100/300/500/1000 remains enabled.

Validation: model/training tests 14 passed, 1 skipped. Diagnostic valid-mask
check passed. Six-GPU real-data two-update run completed and saved complete
checkpoint; exact resume launched to finish the remaining 998 updates.
Launch receipts and logs: `experiments/elaf_recovery_launch_20261019/`.

This multi-change trial cannot isolate the causal contribution of LR, clipping,
precision or resetting optimizer moments. New run does not modify old results.
