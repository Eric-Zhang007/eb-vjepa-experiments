# E-LAF v0: 72-token joint delta-action flow

User specification accepted 2026-10-16. This supersedes the earlier E-LAF
16-future-token prototype for new experiments. It does not change the running
official Dream-VLA mixed-suite baseline or RoboActualizer/V-JEPA policy runs.

## Implementation and acceptance checklist

- [ ] Shared trainable resampler, 32 current + 32 delta + 8 vector-action tokens.
- [ ] Both element-mean flow losses reach the resampler; frozen raw encoder
  features do not acquire gradients. No EMA update, SIGReg or inverse loss.
- [ ] With u=s=0, changing real future observations cannot change model inputs
  or predictions under fixed noise. Targets/loss may change.
- [ ] Future observation is after actions t through t+7, at t+8. Pair windows
  remain in one episode, with explicit action/future validity.
- [ ] Delta RMS, per-slot variance, slot CKA and covariance-spectrum effective
  rank have executable degeneracy tests and persistent step-zero references.
- [ ] Fixed monitor identity and atomic resampler snapshots are checked before
  diagnostics, with training step as the W&B horizontal axis.
- [ ] Native Dream decoder integration tested separately from toy tests.
- [ ] Actual encoder selected, raw feature cache and monitor set constructed.
- [ ] Official trainer integration, full-state resume and distributed smoke
  validated before any E-LAF production run.

## Fixed architecture

Frozen V-JEPA encoder Phi, shared trainable cross-attention resampler g, Dream
pretrained backbone with LoRA, delta flow and action flow. Cache only raw frozen
encoder outputs h_t and h_f. Never cache the trainable z_t/g(h_t) as training
targets. An encoder exported from an EMA training branch is allowed as a fixed
initial weight source; no EMA teacher/update is introduced in this experiment.

Resampler: 32 learned queries, internal width256, eight heads, input/output1024,
one cross-attention layer, query residual and output LayerNorm then a linear
256-to1024 projection. No FFN or reconstruction objective. Current and future
call the same instance. Feature token count N is checked from a real forward.

The downloaded Dream-VLA-7B config was read on the server: language hidden_size
**3584**, 28 layers, 28 heads, full_attn_mask=true. The constructor default4096
does not describe these weights. Always read the loaded config; no hardcoded D.

Use d*=g(h_f)-g(h_t), optionally divided by one recorded, frozen calibration
constant. Independent uniform u,s; independent Gaussian noises; velocity
targets d*-epsilon_d and A*-epsilon_a. Preserve autograd through targets and
noisy inputs into g. Heads are linear. Action vectors are projected7-toD, giving
8 action tokens, unlike the official scalar56-token action representation.

Concatenate VL/proprio conditions,32 current tokens,32 delta tokens,8 action
tokens. Position/type and separate time embeddings are added to these tokens;
time embeddings are not additional tokens. Total sequence length is L_C+72.
Both streams must see each other through bidirectional attention.

Action and delta losses are FP32 element means, each coefficient1. Invalid
action positions do not enter the action denominator. Samples without an
observed t+8 endpoint do not supply a delta target. At inference only current
features are available; integrate both flows jointly with u=s over4 Euler steps.

## Proposed optimization and comparison identity

LoRA rank32, LR5e-5; resampler LR1e-4; new projections, linear heads and time
embeddings LR2e-4. AdamW, weight decay0.01 for matrix weights, excluding bias,
normalization and positional/query embeddings. Cosine,3% warmup, gradient
clipping1, BF16, gradient checkpointing. Proposed global batch60 is
6 GPUs x microbatch1 x accumulation10 (or2 x5).

The active baseline uses global batch48, LR1e-4 and milestone decay, and its
checked repository evaluation default is one flow step. Therefore batch60,
the new learning-rate schedule and four inference steps are explicitly
different experimental settings. Do not attribute any result solely to joint
flow while concealing these changes. Equal-update counts are not equal sample
counts. Total E-LAF update budget and final comparison protocol must be recorded
in the launch configuration, not inferred from the architecture specification.

## Standard diagnostics

Fixed1024 temporal pairs, suite-balanced and episode-diverse; ordered pair IDs,
encoder hash, exact preprocessing identity and data selection are immutable.
The primary monitor is a fixed training diagnostic, not automatically a held-out
validation set. Any validation/probe split must be episode-disjoint and its
training exposure recorded. At step0 save resampler weights and all statistics;
repeat at100,300,1000, then every2000 updates (full models every5000).

1. Raw delta RMS, ratio to fixed initial RMS, delta/current energy ratio.
2. Across-sample variance for each slot: min, mean, median and distribution.
3. Centered linear CKA between slots: off-diagonal mean/p90 and matrix;
   invalid constant slots remain flagged, never assigned reassuring values.
4. Entropy effective rank of covariance eigenvalues for each slot, flattened
   state and flattened delta. Center across samples first. Use sample Gram
   matrices for flattened states, never a32768-square covariance.

Record current, future and delta statistics. The linear256-to1024 output gives
each slot and each delta slot a structural rank cap256. With K1024, the requested
normalization by1023 has maximum256/1023, not1. Preserve that definition and also
log structural rank cap and structural-normalized rank. Global flatten rank
does not inherit the single-slot256 cap. Zero-energy ratios/CKA are undefined,
reported as missing/invalid, rather than as healthy values. No universal alarm
thresholds are claimed. Do not adaptively normalize away falling raw delta RMS.

W&B project remains eb_vjepa. An asynchronous monitor uses its own run grouped
with the E-LAF training run and an explicit training_step axis; two processes
must not concurrently resume/write one W&B run. Keep local full metric records,
per-slot arrays and occasional heatmaps. Never attach E-LAF diagnostics to the
currently running vanilla Dream baseline.

## Data and launch boundary

The old elaf/prepare_data.py is a smoke-only LeRobot exporter with one camera,
mean/std normalization and pooled16-token future features. It is not valid
input for this v0 specification. The active baseline uses verified official
RLDS, two cameras and suite-specific normalization. A v0 exporter must preserve
and record those choices if comparison is intended; the old Goal82 exclusion
must not silently transfer from a damaged video conversion to official RLDS.

Current/future video windows share encoder preprocessing. Cache generation
must use a recorded deterministic crop or a common pairwise transform, never
independent random crops. Beginning-of-episode history may repeat the first
observation, with padding recorded. Missing future observations must be masked
or excluded; never cross episode boundaries or pretend a repeated final frame
is an observed t+8 state.

The selected encoder is the prior grid11k EMA export:
`experiments/ctp_grid20k_20261008/embodied_vjepa_encoder_011000.pt`, SHA256
`468d7af3a5bdae9fb2205718ab0f86a13f5395afcc53700500439d993baf987a`. This is
the audited downstream encoder export, not the old10k checkpoint and not a claim
that it is identical to every official V-JEPA2.1 release. Feature-cache manifests
must record this exact path/hash and the real preprocessing hash.

## Implemented cache and queued launch (2026-10-16)

`elaf/v0_cache.py` exports the four official RLDS suites, then caches one frozen
raw encoder output per observation endpoint. Actual grid11k CPU forward verified
four-frame 384px windows produce 1152 x 1024 features. BF16 features are stored
as uint16 bits; current/future samples reuse the same endpoint arrays. Endpoints
without an observed t+8 frame have an invalid future mask. RLDS no-noops is the
retained dataset clock; it is not an assertion of eight original simulator ticks.

JEPA preprocessing uses deterministic bilinear-antialiased 384px resize and
ImageNet normalization. Native Dream receives both cameras with TensorFlow
Lanczos3 224px resize. This first v0 cache/loader uses no random image augmentation;
the active baseline uses image augmentation. Together with global batch60 versus
baseline48 and new optimizer rates, this is an explicit comparison difference.

`elaf/v0_train.py` is the separate E-LAF research trainer; the running official
Dream baseline is unchanged. It uses equal-probability suite sampling, 200000
updates, six GPUs, micro-batch1 and accumulation10. Every5000 updates it retains
latest and lowest-window-training-loss complete states, including optimizer,
scheduler, per-rank RNG and sampler cursor. Diagnostics run separately on CPU,
so GPU6/7 availability does not block the four required representation metrics.

Full-cache completion and checksum verification precede training. The watcher
requires GPUs0-5 to remain empty for30 seconds, then runs a full-model two-GPU
preflight comparing three updates with two updates plus resume. Only a passing
receipt permits formal training after another30 seconds of idle GPUs0-5.
CPU/unit checks are completed before arming; full-model GPU validation remains
queued while the cards are occupied. No completed GPU validation is implied.
