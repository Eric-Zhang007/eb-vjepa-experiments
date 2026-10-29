# Infrastructure update — verified 2026-10-19 12:33 CST

Production resumed from step583, same W&B run gtnxs7ir, same optimizer state,
sampling cursor, Linear heads, batch48, learning rates and schedule.

Changes: bounded ordered CPU prefetch with separate augmentation RNG and consumed
state restoration; rank-specific pinned memory; aggregated GPU gradient norms;
DDP gradient bucket views; disabled activation checkpoint recomputation.
Original source is preserved remotely at experiments/elaf_infra_before_20261019.
Explicit source/config migration receipt: experiments/elaf_infra_resume_20261019/migration.json.

Eight relevant unit tests passed. Actual augmented cached samples matched the
synchronous reader bitwise, including consumed CPU RNG state. Single-GPU batch8
model-only benchmark: checkpoint on1.0093s, off0.7377s; off peak allocated33.17GB.
Disabling recomputation is not claimed bitwise equivalent under BF16.

Live six-GPU training reached6630 updates. Mean resumed update time1.180s versus
previous2.038s, approximately1.73x throughput. Last100 rank0 data waits averaged
0.000332s. A GPU snapshot showed91-100% utilization; this is not a sustained
utilization average. Latest/best complete checkpoint both at5000. Diagnostics
continued through6000 in the existing diagnostic run mtd2kaxm.

Representation concern remains separate from throughput: at6000 raw Delta RMS
0.01512 (0.5001x initial), slot CKA mean0.9640, state global effective rank6.50.
These are warning signals, not a standalone proof of collapse. No architecture
or objective change was made in response.

## Follow-up performance investigation, 12:49 CST

Checklist completed: measure real data waiting, compare optimizer on identical
inputs, compare single/DDP on matched mixed samples, sweep communication bucket
sizes, and confirm production continues and spare GPUs are released.

Production continued past step7450 at approximately1.180s/update. Recent data
waiting remains below1ms. Short utilization sampling confirmed dips, including
roughly50-60% on individual GPUs while other ranks were busy; utilization is not
uniformly100%. GPU0-3 and GPU4-7 are on separate NUMA nodes, without NVLink.

Isolated GPU6/7 tests used batch8, recomputation disabled, 3warmup/10timed
updates. Fixed-input AdamW versus fused AdamW:0.74078 versus0.73489s/update
(0.8%); optimizer segment14.82 versus8.18ms. Separate BF16 trajectories were not
bitwise equal, so this is not an equivalence claim.

For mixed sample_step7000, single GPU and DDP rank0 used identical sample indices
and padded input shape[8,172]. Single:0.74269s, forward0.42282s,
backward0.30173s. Two GPUs:0.83747s, rank0 forward0.42226s,
backward0.39811s. The additional time is concentrated in backward for this test.
DDP rank1 padded shape was[8,174]. These are model-only timing tests using rank
mean losses; they do not reproduce production's global valid-element weighting,
data transfers, telemetry, or six-GPU cross-NUMA topology.

Two-GPU bucket cap25MB(default),4MB,100MB: approximately0.83747,0.83072,
1.21517s/update. No tested alternative justified a production restart. The
remaining six-GPU overhead cannot be assigned entirely to communication from
these tests; a production six-rank trace is needed for that attribution.
Production config, optimizer and run remained unchanged during this follow-up.
GPU6/7 were confirmed released after testing (5/7MiB idle contexts).

Reports are under remote experiments/elaf_infra_benchmark_20261019:
default_segment.json, fused_segment.json, single_mixed_segment.json,
ddp_mixed_segment.rank*.json, ddp_bucket{4,100}_segment.rank*.json.
