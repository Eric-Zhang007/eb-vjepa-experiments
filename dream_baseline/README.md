# Reference utilities only

The custom training entrypoint and its dedicated configuration have been deleted.
This directory is not a training suite.

Baseline training uses scripts/launch_dream_baseline.sh, which invokes the
upstream vla-scripts/finetune_dream.py and official RLDS pipeline.
See experiments/DREAM_VLA_OFFICIAL_TRAINING.md for the training recipe.

Remaining model/data utilities are unused prototype reference code.
