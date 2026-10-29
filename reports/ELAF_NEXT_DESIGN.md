# Next E-LAF revision — decisions 2026-10-19

Status: design pending; does not modify or resume the running v0 experiment.

## User-approved training alignment

- All trainable module learning rates: 1e-4.
- Scheduler: MultiStepLR, milestone100000 optimizer updates, gamma0.1.
- Warmup:0, matching the completed baseline.
- Effective global batch:48. Prefer six GPUs x microbatch8 x accumulation1;
  choose a smaller microbatch only if measured memory requires it, retaining48.
- Native Dream image augmentation: reuse the baseline RLDS augmentation
  implementation and parameters; a boolean flag alone is not equivalence.
- Keep four-suite equal sampling and200000 optimizer updates.
- The frozen JEPA cache remains deterministic. Dream image augmentation and
  paired JEPA preprocessing are separate branches and must be documented.

## Verified official action path

Remote source: repos/Dream-VLX/vla/prismatic/models/action_heads.py,
projectors.py and vla-scripts/finetune_dream.py.

Actual hidden width D=3584. Eight seven-dimensional noisy actions are flattened
into56 scalar tokens. NoisyActionProjector maps1 -> D -> D with GELU.
The native model fuses action tokens, time, visual/language and proprio conditions.
The56 output hidden vectors are regrouped as [B,8,7D].
FlowMatchingActionHead is a shared per-action-time MLPResNet:
LN(7D), Linear(7D,D), ReLU, two residual blocks each LN(D),Linear(D,D),ReLU,
then LN(D),Linear(D,7). It has115713031 parameters at D3584.
Temporal interaction between the eight actions occurs in the shared Transformer;
this head is not a separate temporal Transformer.

## Candidate preserving official action modules (not yet selected)

Retain56 scalar action tokens, official action input projector, action time
conditioning, proprio projector, and official MLPResNet action velocity head.
Append32 current-state and32 noisy-delta tokens to shared bidirectional backbone.
Action output remains[B,8,7]; Delta output[B,32,1024]. All streams interact in
the backbone. There are120 world/action tokens, plus native condition/time tokens;
this is not the earlier72-token architecture.

Delta head candidate: per-slot LayerNorm, Linear(D,1024), activation, two
width1024 residual MLP blocks, LayerNorm, Linear(1024,1024). Shared across32 slots.
This is a proposed nonlinear reader, not a proven improvement over Linear.

Still to decide: this120-token design versus retaining8 vector action tokens
with a redesigned nonlinear head; LoRA scope (official whole-VLA scope needed
for a controlled baseline comparison); official flow time sampling/sign versus
the v0 uniform noise-to-data convention. These must be coherent with integration.
Do not label modified inputs/heads an unchanged official action module.
