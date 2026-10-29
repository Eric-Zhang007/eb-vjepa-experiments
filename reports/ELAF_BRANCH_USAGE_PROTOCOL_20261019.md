# Step 2000 branch usage diagnostic

No training or checkpoint edits. Source: `elaf_fresh_flowclip_2k_20261019`.
Script: `scripts/diagnose_elaf_branch_usage.py`.

32 deterministic training samples, eight from each suite, interleaved so each
four-sample batch contains all four suites. No image augmentation. Same current
observations, text, action target, action noise and delta noise for all paired
interventions. Each sample is evaluated at ten u bin midpoints .05,.15,...,.95;
s=.5 throughout. These are controlled midpoint evaluations, not empirical
averages over uniformly sampled times within each bin. No u=1 division.

Conditions:

- A full joint velocity forward.
- B shuffle clean delta content by a fixed cyclic derangement across samples;
  retain original per-sample Gaussian noise and original target.
- C disallow Action queries reading Delta keys in every Transformer layer.
- D disallow Action queries reading Current JEPA keys in every layer.
- E disallow ALL non-Delta queries reading Delta keys in every layer, removing
  indirect Delta-to-Action relay routes as well as the direct edge.

All conditions retain sequence positions and token count. C/D do not remove
indirect paths; E does. All are distribution-shifting interventions, not
retrained module removal. They do not measure auxiliary loss training benefits.

The initial SDPA approach failed a 5% unblocked-output drift check and was
stopped without generating interpreted intervention results. Final implementation
uses native FlashAttention throughout: full queries use the original kernel;
direct-edge blocks recompute only Action queries with selected keys masked;
global isolation recomputes all non-Delta queries using non-Delta keys/values.
Q/K projections, rotary positions, dtype and original kernel are retained.
All 28 decoder layers must invoke the wrapper. Unblocked wrapper output must
be bit-identical to the original. Global Delta isolation is additionally checked:
flipping Delta input must leave Action output unchanged (max abs <=1e-5).

Report: valid-element Action/Delta velocity MSE and output changes; zero-velocity
and analytic Delta baseline `-d_u/(1-u)` on the original targets and masks.
The analytic MSE is checked against mean(d_target^2)/(1-u)^2 in saved results.

Artifacts: server run directory `branch_usage_native/` and `branch_usage_native.log`.
No inference Euler trajectories or closed-loop LIBERO evaluation in this test.
Results must not be described as policy success rates or proof of dynamics.
