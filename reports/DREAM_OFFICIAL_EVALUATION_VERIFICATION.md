# Official Dream-VLA evaluation verification ? 2026-10-16

The verified checkpoint is the previous **LIBERO-10-only step5000** model, not a mixed-four-suite model and not E-LAF. `dream_vla_official` loads official merged safetensors or the original base with an official LoRA adapter, plus the three official action/proprio/noise modules. It does not import the deleted custom trainer.

## Verified

- All four real model shards loaded with strict auxiliary module weights and SHA256 identity.
- Official `run_libero_eval_dream.py:get_action` compared to adapter batch1 under identical noise: max final action error9.01e-8; captured decoder hidden and predicted velocity were exactly equal.
- Two variable-language observations jointly batched vs single inference: max action difference0.015625 from BF16 batched arithmetic; tested with atol0.015 and rtol0.03. Repeat same batch is bitexact. This is **not** batch-vs-single bitexactness.
- Seven tests passed: official image preprocessing equality, explicit suite-to-normalization mapping and failure checks, deprecated-family rejection, runner coverage/failure tests.
- Actual Plus camera tasks703 and714 completed700 control steps each,0/2 successes,72.08seconds including model load. This is a wiring smoke result, not a model quality estimate.2 environment workers had actual maximum inference batch1.

## Evaluation protocol

- Native upstream default `steps=1` flow integration is used, and8 action steps executed per prediction.
- Existing shared exploratory Plus protocol uses256px source images rotated180 degrees, official JPEG/Lanczos224 plus90% center crop,30 settle steps and700-step horizon for LIBERO-10. Gripper0close/1open is converted to simulator coordinates.
- This differs from the upstream clean LIBERO-10 default10 settle steps and520-step horizon. Do not report this as the upstream clean evaluation protocol.
- Suite-specific action/proprio q01/q99 stats are selected explicitly. A10-only checkpoint refuses other suites. The mixed configuration template maps all four suites to their corresponding saved RLDS statistics.
- Future merged full-state checkpoints preserve the same evaluation payload; optimizer/RNG files are ignored for inference. Unmerged loading is implemented but was not exercised by this merged5k verification.

Artifacts: `dream_official_eval_equivalence.json`, remote `experiments/dream_official_plus_smoke_v1/{summary.json,gpu_4/results/}`, and `dream_official_eval_tests.log`.

## Eight-environment follow-up

Eight Plus LIBERO-10 camera tasks completed all5600controlsteps,0/8successes. Actual inference batch mean3.785 and max4 demonstrate dynamic batching (configuredlimit8,wait5ms). Rollout54.63seconds gives527.15episodes/hour; includingmodel load98.11seconds gives293.54episodes/hour. Different episodecounts mean these are not a controlled8vs2speedup claim. All8 per-episode records include horizon700,settle30,execute8,image rotation180 and gripper convention. Artifact: `experiments/dream_official_plus_batch8_v2/`; `dream_official_plus_batch8_summary.json` copied locally. GPU4 released after completion.

A new reporting bug in the first8worker attempt (reset-only metadata read on step replies) was fixed; that failedattempt is excluded, preserved at `dream_official_plus_batch8/`. The runner tests now cover distinct reset/step metadata and passed4/4 afterfix.

Actual LIBERO-Plus repository registers the base suite names (`libero_spatial/object/goal/10`); perturbations are task IDs and category fields, not renamed suites. All verified Plus tasks use these base names. The mixed configuration maps exactly those four registered suite names.
