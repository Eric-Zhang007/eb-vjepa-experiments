# E-LAF architecture audit

Date: 2026-10-15. Inspected upstream Dream-VLX commit `a72fa4106eae97a923b531342fb692742809782f` and `E-LAF路线初步规划.md`. This document records implementation findings, not experimental evidence. No model code was changed in this audit.

## Checkpoint and architecture

The official release is `Dream-org/Dream-VLA-7B`. Parent task verified HF revision `0cd0e25c1771c5b468c74585b5cf35c4a4aa315c`: four safetensors shards, 16,634,226,144 bytes of shard files, 8,317,072,896 BF16 parameters, no separate action-head or noisy-action-projector checkpoint. The public base checkpoint must not be described as an already LIBERO-finetuned continuous-flow policy. Official `vla-scripts/run-dreamvla.sh` initializes from this release and trains continuous flow modules separately.

The repo's `vl/train/dreamvl/config.json` has hidden size 3584, 28 decoder layers, 28 attention heads, 4 KV heads; vision depth 32, embed dimension 1280, patch size 14, temporal patch size 2, spatial merge 2. These values describe the bundled config; use downloaded checkpoint config as runtime authority rather than Python class defaults (which differ). Native modules are `DreamVLModel.visual`, `.projector`, `.model`, `.lm_head`. `DreamVLAForActionPrediction` subclasses this model and adds action normalization and generation helpers.

`DreamVLAttention`/`DreamVLFlashAttention2`/`DreamVLSdpaAttention` explicitly use noncausal attention (`modeling_dreamvl.py:342,522,628`). This is shared self-attention over the concatenated sequence, not a separate cross-attention block. Padding must remain masked. Clean condition *input embeddings* stay unchanged across flow steps; their internal hidden states can change because they attend to noisy action/latent tokens. Cached input visual embeddings are valid; cached transformer condition K/V would generally not be equivalent.

## Exact input and output contracts

`DreamVLModel.get_image_features` produces one sequence per image, length `prod(image_grid_thw) // spatial_merge_size**2`. `forward` projects these through `.projector` and replaces image placeholders in the text embedding sequence using `masked_scatter` (`modeling_dreamvl.py:1630,1717`). Thus 256 image tokens is not a universal constant: it depends on resize/grid and camera count. Official finetuning shell uses two images and proprioception.

For LIBERO, official constants are action chunk 8, action dimension 7, proprio dimension 8. The actual action input is **56 scalar tokens**, not 8 vector tokens. `NoisyActionProjector` is `Linear(1,D) -> GELU -> Linear(D,D)` (`prismatic/models/projectors.py:32`); `[B,8,7]` is flattened to `[B,56,1]` then projected to `[B,56,D]`. A sinusoidal time token `[B,1,D]` is prepended. Proprio is projected to `[B,1,D]` and prepended separately. `FlowMatchingActionHead.predict_velocity` reshapes action hidden states from `[B,56,D]` to `[B,8,7*D]`, then predicts `[B,8,7]` with a residual MLP (`action_heads.py:296`).

`DreamVLModel.forward` accepts `inputs_embeds`, `attention_mask`, `noisy_action_features`, `all_actions_mask`, and `timestep_embeddings`. It overwrites supplied position IDs with `attention_mask.cumsum(1)` in this VLA path. It strips prepended proprio/time tokens from returned `hidden_states`, which is a final-state tensor, not the usual Hugging Face tuple. It always computes vocabulary logits. For a continuous-only objective, direct use of `.model(inputs_embeds=..., attention_mask=..., position_ids=..., use_cache=False, return_dict=True)` avoids those unneeded logits and returns decoder output `.last_hidden_state`.

## Training and inference pitfalls

- Official training forms `x_t=(1-t)*noise+t*clean`; the actual training target in `finetune_dream.py:477` is `clean-noise`, and inference adds `velocity*dt`. The `flow` entry returned by `FlowMatchingActionHead.sample_noisy_actions` has the opposite sign and is unused by that trainer. Do not reuse that entry.
- Official time sampling is a custom function named `sample_beta`, implemented via powers of uniform samples, not PyTorch's Beta distribution. The new proposal explicitly uses independent uniform stream times, which should be implemented as stated and logged as a changed training choice.
- Official hidden-state selection uses `labels[:,1:]` against `hidden_states[:,:-1]` (`finetune_dream.py:447-461`). Action predictions are therefore shifted relative to inserted action tokens. Preserve this when reproducing the official baseline; use explicit same-position stream slices for the new in-place flow model and treat that as a design change.
- `constants.py` guesses platform from `sys.argv`; its default is actually BRIDGE, chunk 5, despite the comment claiming LIBERO. New code should carry explicit dimensions in its own config and never depend on command-line spelling.
- `_flow_matching_prediction(...,steps=1)` defaults to one step (`modeling_dreamvl.py:2120` onward), and uses a floating-point while loop. Four steps is an experimental choice, not a universal architectural guarantee. Use an integer loop with exactly K evaluations and `t=k/K` in new code.
- Official inference finally reshapes to `[chunk,action_dim]`, so batch sizes above one are not supported there. Keep `[B,H,A]` in the new policy API.
- Saving official continuous finetuning stores the action head and noisy projector as separate `.pt` files (`finetune_dream.py:656-662`). Downloading only the base weights cannot recreate those learned modules.

## Minimal safe joint extension

Keep upstream checkout intact. Build a wrapper around its native decoder and visual embedding construction. Assemble an explicit sequence containing clean prompt/visual/proprio embeddings, distinct action-time and latent-time embeddings, latent tokens `[B,M,D]`, and scalar action tokens `[B,H*A,D]`. Retain explicit indices and valid masks for each stream; never infer them from numeric token-ID thresholds. Supply padding-aware positions and bidirectional attention. Avoid future target features in condition inputs.

Use a separate latent input projector `d_z -> D`, latent output projector `D -> d_z`, stream identifiers, and independent times `u,s`. Action output can retain official scalar-token aggregation to minimize unrelated changes. Both heads must consume the same native decoder pass; an independent auxiliary head with no influence on action computation does not implement the requested joint model.

Loss should be written consistently as `L_action + lambda_latent * L_latent`, with per-stream masked mean squared error before weighting. The planning equation weights action while its experiment table weights latent: resolve this naming inconsistency in code. Record latent normalization statistics and preserve teacher identity/checksum. Padding masks must exclude nonexistent future frames/actions near episode ends.

Frozen V-JEPA targets require an explicit temporal clip definition. A single future RGB frame is not automatically a dynamic representation. Specify teacher checkpoint, frame cadence, clip length, target horizon, token pooling, camera handling and augmentation; preserve episode and timestamp alignment. The current official RLDS transform supplies current observations and future actions but does not already supply the aligned JEPA future clip. Its random goal relabeling is not a replacement for fixed-horizon future supervision.

Train with independent times and sample jointly along `u=s=t` for the first implementation. A fixed-noise control must hold `u=0` and the same initial latent noise throughout the action solve. This is not cached/amortized future prediction, and uniform continuous time training gives the exact boundary zero probability: include an explicit boundary training mixture if this control matters. It still incurs latent-token attention unless those tokens are removed. Removing tokens changes the model input distribution and requires its own control. Neither comparison alone proves a unique mechanism.

## Minimum meaningful verification

1. Instantiate a tiny native DreamVL decoder and run two unequal-length padded batches; verify action/latent shapes, finite losses and gradients through both heads and shared decoder, with the teacher frozen.
2. Perturb latent noisy inputs at fixed condition/action/time and confirm action predictions change. Compare autograd action sensitivity to latent inputs: this tests actual cross-stream coupling.
3. Perturb padded condition and latent tokens and verify valid predictions remain unchanged; test end-of-episode target masks and zero valid target rejection.
4. With a constant analytic velocity, K-step Euler must recover the exact endpoint, use K evaluations, and preserve batch dimension. Fixed-noise mode must not update latent state.
5. Verify checkpoint save/load reproducibility, action-only mode, parameter freezing, finite backward, and a real official-checkpoint multimodal forward after download. A tiny synthetic pass does not establish 8.3B memory feasibility or readiness for training.

Planning claims such as 40% compute savings, total sequence length 350, superior disturbance robustness, or comparisons with other papers remain unverified hypotheses. No such claim is an implementation acceptance criterion.
