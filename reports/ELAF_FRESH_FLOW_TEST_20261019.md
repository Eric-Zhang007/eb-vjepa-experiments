# Fresh E-LAF optimization trial

2026-10-19. This is a diagnostic run, not a full 200k training run.

## Same-batch analytic test

Checkpoint: recovery run `elaf_recovery_from5000_20261019`, update 1000.
Four valid training samples, one per suite, augmentation disabled, u=s=0.5,
fixed Gaussian noise and identical target/mask/reduction.

| Delta measure | Value |
|---|---:|
| Model MSE | 0.985690 |
| Zero velocity MSE | 0.998078 |
| Analytic -2*d_u MSE | 0.001259791 |
| 4*mean(delta_target^2) | 0.001259791 |
| Delta target RMS | 0.0177468 |

The analytic identity holds. This model fails to approach the simple denoising
baseline at u=.5; it is not merely a uniform-time averaging effect.

Delta projection input/output RMS: .499553/.366164. Delta output-head input RMS
(Transformer delta hidden): 5.260995; predicted velocity RMS: .066335.
Flipping only delta noise changes velocity by RMS .031302. Changing u to .25
while retaining noise changes velocity by .022995; changing only the time input
while holding noisy tokens fixed changes it by .022479. The path responds, but
these measurements do not rule out routing or optimization defects.

## Fresh fixed-batch test

Restored fresh LoRA/resampler/new-module initialization after the checkpoint
diagnostic; retained pretrained frozen Dream backbone. Used the same four
samples, u=s=.5 and fixed noise for 60 optimization updates, no image augmentation
and no warmup. Five independent clipping groups: LoRA, Resampler, Action,
Delta, Current (including proprio). LR 2e-5/5e-5/1e-4 respectively.

| Metric | Update 0 | Update 60 |
|---|---:|---:|
| Action MSE | 1.816097 | .012743 |
| Delta MSE | 1.661134 | .393323 |
| Delta analytic MSE | .017317 | .015278 |

Both objectives can decrease. Delta remains much worse than the analytic
baseline. Fixed samples/noise can be memorized; this is not evidence of dynamics
learning or generalization. Targets remain learnable in this test.

## Fresh real-data run

Output: `experiments/elaf_fresh_flowclip_2k_20261019`.
Config: `elaf/configs/fresh_flowclip_2k.json`.
No resume/fork, no weights imported from either failed/recovery model or the
fixed-batch test. Pretrained Dream + frozen grid11k retained; adapters and new
modules freshly initialized. GPU0-5, global batch48, 2000 updates then stop.
FP32 resampler/delta, five independent clipping groups at1, 200-update warmup,
LoRA LR2e-5, Resampler5e-5, new modules1e-4. Real-data augmentation enabled,
independent uniform u/s. Loss1:1; no EMA/SIGReg/stop-gradient.

This tests optimization with fresh weights across the old instability interval.
It does not establish that the analytic baseline has already been matched.
Logs: `experiments/elaf_fresh_flowclip_launch_20261019/train.log`.
Detailed diagnostic output: `experiments/elaf_flow_diagnostic_20261019/`.
Model/training tests:20 passed,1 skipped.
