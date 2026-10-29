# Evaluation interface validation, 2026-10-15

Implemented shared per-GPU model service, independent simulator workers, bounded
ready-observation batching, cross-client dynamic batching, multi-GPU episode sharding,
strict unique-episode aggregation, explicit action execution length, and throughput sweep.

## Evidence

- clean LIBERO, Plus, Pro real reset +30 settling +2 control steps passed.
- Tiny E-LAF completed four full clean episodes through the model service.
- Tiny E-LAF completed four full Plus episodes over two GPU services. These random
  tiny-model rollouts verify infrastructure, not E-LAF policy quality.
- Both actual RoboActualizer action-only and tiny E-LAF checkpoints passed batched
  versus individual inference checks within BF16 tolerance; repeated batches identical.
- Two real concurrent RPC clients were combined into one inference batch.
- RoboTwin SAPIEN headless rendering passed; native XPolicyLab adapter registered and
  interface tests passed for RoboTwin and RoboDojo. Full RoboDojo rollout awaits downloads.
- No official joint RoboActualizer model was evaluated. V-JEPA identifies the frozen
  encoder component; it is never used as an alias for E-LAF.

## Initial throughput measurement

GPU0, RoboActualizer action-only derivative, grid encoder11000, action20000.
Same first16 clean exploratory initial states, four denoising steps, execute10 actions
per replan, full400-step episode limit. One measurement per setting, no video saving.

| Environments | Total seconds including model load | Episodes/hour including model load | Rollout seconds | Mean batch | Successes |
|---:|---:|---:|---:|---:|---:|
|1|192.53|299.2|172.18|1.00|15/16|
|4|87.52|658.1|63.13|1.06|15/16|
|8|76.02|757.7|46.19|1.91|15/16|
|16 follow-up|74.02|778.2|34.57|3.68|16/16|

The 1/4/8 sweep used the same evaluator revision. The16-environment follow-up added
cross-client request coalescing and memory telemetry, so it is not a pure one-variable
comparison with that sweep. Numerical batch rounding changed individual trajectories
and total control steps; one extra success is not evidence of improved model quality.
The sample is small and concentrated in the first four spatial tasks.8 environments
is a measured starting point, not a global optimum or a setting transferable to8.3B E-LAF.
The measured1-to8 total-time ratio is2.53. Measure full E-LAF independently after training.

Remote artifacts: `experiments/elaf/throughput_roboactualizer16/`,
`eval_tiny_clean4/`, `eval_tiny_plus4_multigpu/`.
Local numerical copy: `experiments/ELAF_THROUGHPUT_COMPARISON.json`.
Launch instructions: `elaf/evaluation/README.md`.

Current remaining boundaries: model-specific full E-LAF throughput not tested;
RoboTwin/RoboDojo require checkpoints trained for their action/state layout; no
7D LIBERO-to-dualarm conversion is implied. RoboDojo installation/assets remain
background jobs. Latest observed Dream weights6.07/16.63GB and RoboDojo assets3834/15365
were progressing, not stalled; these are timestamped observations, not completion claims.
