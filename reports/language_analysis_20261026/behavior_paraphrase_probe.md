# Fixed-observation paraphrase action probe

This uses cached action velocity outputs with identical scene, proprio, noise, and flow state. It tests paraphrase stability only; it is not a semantic-switch or success-rate test.

- Cases: 5; comparisons: 56
- Mean relative velocity change: 0.1208
- Median relative velocity change: 0.1262
- Mean velocity cosine: 0.9922
- Minimum velocity cosine: 0.9794

A real task-discrimination probe still needs matched scenes with two valid competing targets and closed-loop goal predicates.