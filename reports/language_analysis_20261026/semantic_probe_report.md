# Instruction semantic probe

This probe tests semantic-slot decodability under paraphrase holdout. It does not measure closed-loop task success.

| representation | full task acc | action | object | source | destination | state |
|---|---:|---:|---:|---:|---:|---:|
| umt5_instruction_only | 0.692 | 0.923 | 0.885 | 0.231 | 0.769 | 0.846 |
| dream_input_embedding | 0.731 | 0.885 | 0.808 | 0.423 | 0.846 | 0.885 |
| dream_scene40_time0.0 | 0.192 | 0.346 | 0.115 | 0.423 | 0.231 | 1.000 |
| dream_scene40_time0.5 | 0.462 | 0.423 | 0.462 | 0.115 | 0.538 | 1.000 |
| dream_scene80_time0.0 | 0.115 | 0.615 | 0.115 | 0.077 | 0.385 | 1.000 |
| dream_scene80_time0.5 | 0.154 | 0.615 | 0.308 | 0.115 | 0.577 | 1.000 |

The prior nearest-canonical score is retained as a descriptive baseline only. A behavioral probe with matched scenes and minimal semantic switches is still required before making a claim about task discrimination.