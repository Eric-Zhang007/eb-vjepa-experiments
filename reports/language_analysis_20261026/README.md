# Dream-VLA language probes: evidence and corrections

Published 2026-10-30. These are historical exploratory probes, not full benchmark evaluations.

| File | Content / scope |
|---|---|
| [information_flow_probe.md](information_flow_probe.md) | Modified Dream-JEPA 140K: 8-episode language and JEPA swaps, Q/K scales, video observations |
| [behavior_paraphrase_probe.md](behavior_paraphrase_probe.md) | Matched-observation action-velocity sensitivity: 5 cases, 56 comparisons |
| [results.json](results.json) | Pooled instruction-feature distances and nearest-canonical retrieval; descriptive only |
| [pairs.json](pairs.json) | Canonical/paraphrase pairs used for feature probes |
| [base_logit_probe.json](base_logit_probe.json) | Original Dream-VLA base, one prompt and sampled generation position; action-token probability mass |
| [semantic_probe_report.md](semantic_probe_report.md) | Historical Ridge probe table; INVALID pending correction, see below |

## Correction discovered on 2026-10-30

The saved `scripts/probe_instruction_semantics.py::ridge_predict` standardizes training features first, then calculates mean/std from those already standardized features and applies these to raw test features. Thus training and test features use different coordinate systems. The semantic-slot and full-task accuracies in `semantic_probe_report.md` and `semantic_probe_results.json` must NOT be used as evidence until corrected and rerun. Original files are preserved for traceability. This correction does not invalidate the separately computed fixed-observation velocity probe or closed-loop intervention counts.

## Interpretation limits

- Most feature/action/intervention probes concern modified Dream-JEPA 140K, not the LIBERO baseline200K model.
- Deleting function words also removes relations and breaks grammar. It does not isolate article sensitivity.
- Keyword swaps are scored against the original task; a failed original task does not establish correct execution of the modified instruction.
- Q/K magnitudes are not attention probabilities. Claims in the historical information-flow report about routing/calibration are hypotheses, not identified causes.
- Nearest-canonical retrieval and mean-pooled distances are not behavioral task discrimination tests.
- The base logits probe does not establish a universal hard restriction on text generation.

Raw stage/subgoal generations and episode records remain in the complete HF archive at `Isla8899/eb-vjepa-experiment-records`; check upload completion before relying on that archive. These small reports are directly readable here on GitHub.
