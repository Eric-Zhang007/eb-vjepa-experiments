# Dream-JEPA information-flow probe (2026-10-26)

## Setup

- Checkpoint: Dream-JEPA dual-view 140k pin.
- Evaluation: 8 fixed LIBERO episodes, GPUs 6-7, 4 environments/GPU, action chunk 8, 4-step Euler sampling.
- Current JEPA input: 16 primary-view tokens + 16 wrist-view tokens.
- Future input: 16 primary-view latent tokens.
- The two JEPA views use the same spatially anchored resampler parameters.

## Causal feature swaps

| Condition | Success | Per-episode result |
|---|---:|---|
| Normal | 7/8 | T T T T T T F T |
| Swap primary JEPA features across batch | 7/8 | T T T T T T F T |
| Swap wrist JEPA features across batch | 5/8 | T T F T T F F T |

The primary swap did not reduce this small-sample success count. The wrist swap changed two episodes from success to failure. This is evidence that the primary branch is weakly or unstably used by the action path, while the wrist branch has a modest causal effect. It is not a benchmark-level success-rate estimate.

## Language interventions

| Condition | Success | Action RMS difference from normal over first 16 steps |
|---|---:|---:|
| Swap `stove` and `cabinet` | 5/8 | about 0.06 average over the eight episodes |
| Remove function words (`the`, `in`, `on`, `to`, `and`, ...) | 0/8 | about 0.13 average over the first 16 steps |

The noun swap changes behavior on cabinet/stove-dependent instructions, but the normal control is noisy (the same fixed set has been 6/8 and 7/8 in separate runs). The function-word intervention produces a much larger action change and 0/8 success. This shows strong sensitivity to the instruction surface form, but it does not yet prove that articles or prepositions are represented as task semantics: the resulting prompts are out-of-distribution and grammatically damaged.

## Q/K projection probe

The probe records projected Q/K RMS by token group at layers 0, 7, 14, 24, and 27. The instruction group includes non-image, non-action base tokens (special/proprio/time tokens are therefore mixed in); it is not a pure word-level attention score.

| Layer | Q image | Q instruction/special | Q current JEPA | Q future JEPA | K image | K instruction/special | K current JEPA | K future JEPA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.36 | 3.75 | 2.93 | 11.26 | 20.84 | 20.89 | 20.85 | 20.86 |
| 7 | 1.21 | 1.20 | 1.37 | 1.37 | 1.53 | 1.37 | 1.58 | 1.43 |
| 14 | 1.40 | 1.42 | 1.57 | 1.66 | 1.34 | 1.28 | 1.28 | 1.00 |
| 24 | 1.49 | 1.55 | 1.48 | 1.74 | 1.39 | 1.29 | 1.18 | 1.23 |
| 27 | 29.00 | 20.40 | 185.09 | 544.52 | 51.75 | 44.08 | 210.28 | 607.13 |

The JEPA tokens are not simply ignored. Their late-layer Q/K scale is much larger than the image and instruction groups, especially for Future tokens. This indicates a calibration/routing problem is plausible: the JEPA branch can strongly perturb the shared transformer, while the action output does not consistently depend on the corresponding primary-view features.

These are Q/K magnitude diagnostics, not attention probabilities. The current joint implementation intentionally uses SDPA and rejects `output_attentions=True`; a separate eager-attention diagnostic is needed for exact token-to-token attention weights.

## Video contrast

The clearest visual contrast is LIBERO spatial task 4 versus task 9:

- Task 4 failure: retrieve the bowl from the top drawer, then place it on the plate.
- Task 9 success: pick up the bowl already exposed on the wooden cabinet, then place it on the plate.

The failure requires drawer opening/retrieval and a more precise contact sequence. The model can handle the broad bowl-to-plate relation when the object is exposed, but fails on the hidden-object and contact-control subtask. This points toward fine visual grounding and temporal control, rather than a complete absence of object-language understanding.

## Current information-flow interpretation

The base Dream-VLA sequence contains the original image and instruction tokens plus action placeholders. The model then appends 32 Current JEPA tokens (primary + wrist), one future-time token, and 16 Future JEPA tokens. Action hidden states are read from the original action-token positions and passed through the official Dream-VLA action MLP; the Future head reads the appended Future positions.

When Future dropout is off (80% of samples), the joint mask permits earlier tokens, including action tokens, to attend to Future tokens. Future tokens can attend to preceding base/current/action tokens. When dropout is on (20%), the earlier-to-Future route is blocked. The flow times `u` and `s` are independently sampled during training; they are not synchronized.

The current evidence is consistent with three simultaneous effects:

1. The primary JEPA branch is weakly coupled to the action decision in this interface.
2. The wrist branch has some influence, but not enough to guarantee correct fine manipulation.
3. The instruction representation is highly surface-form sensitive, while late JEPA activations are badly scale-matched to the pretrained backbone.

Data scarcity may amplify the language sensitivity, but it cannot by itself explain why swapping the primary visual features leaves this probe unchanged while late JEPA Q/K magnitudes are huge. The more immediate suspect is the interface calibration and routing between appended JEPA tokens and the pretrained Dream-VLA layers.

## Artifacts

- Q/K data: `artifacts/dream_jepa_qk_probe/qk.json`
- Video contact sheet: `artifacts/dream_jepa_qk_videos/spatial_failure_success_compare.png`
- Normal videos/records: `artifacts/dream_jepa_qk_videos/`
- Function-word records: `artifacts/lang_probe_jsonl/drop/`
- Keyword-swap records: `artifacts/lang_probe_jsonl/swap/`
