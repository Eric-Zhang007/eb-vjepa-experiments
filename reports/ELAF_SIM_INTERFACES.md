# E-LAF simulator interface and action contracts

## Transport

The simulator process does not import Dream-VLA/E-LAF weights. A single model service owns weights and serves batched `PolicyClient.predict` requests. The adapter converts XPolicyLab decoded observations to `{image: HWC uint8 RGB, proprio: list[float], instruction: str}`, preserving active `env_idx` order and sending `episode_keys`. Each observation carries deterministic `noise_seed` derived from base seed, episode index, environment index and replan count; batching order does not change per-environment randomness. The service must return finite `float32` actions with shape `[batch, action_horizon, action_dim]` plus metadata during hello.

## XPolicyLab compatibility

Native XPolicyLab expects `reset`, `update_obs`/`update_obs_batch`, `get_action`/`get_action_batch`; a chunk is returned per inference. The E-LAF adapter implements these methods and is registered into both RoboTwin and RoboDojo XPolicyLab checkouts. RoboTwin's environment passes decoded `vision[*].color` RGB arrays. The adapter selects one configured camera and explicit state fields; no implicit channel reorder, image normalization, state normalization, action scaling, gripper inversion, or quaternion convention conversion occurs.

The configured example is dual-arm joint policy (`[6,6] + [1,1] = 14D`) with native ordering `left_arm_joint_state,left_ee_joint_state,right_arm_joint_state,right_ee_joint_state`. EE mode requires 7D `[x,y,z,w,x,y,z]` per arm plus grippers and therefore is 16D for dual arms. RoboTwin `franka.yml` resolves to `dual_franka`; its default simulation is not compatible with a 7D LIBERO action checkpoint. Any mismatch is a hard error at adapter construction. Single-arm LIBERO and dual-arm RoboTwin require separately trained output heads/checkpoints or an explicit model-side conversion implementation.

## Launch

```bash
# Register in a checkout once
bash scripts/robodojo/register_policy.sh /data/users/jiachangzhang/RoboTwin/XPolicyLab
bash scripts/robodojo/register_policy.sh /data/users/jiachangzhang/eb_vjepa/repos/RoboDojo/XPolicyLab
# Start main-owned model RPC server separately, then start one CPU bridge per simulator trial process:
XPL_ROOT=/data/users/jiachangzhang/RoboTwin/XPolicyLab bash scripts/robodojo/launch_bridge.sh
# In a separate shell, launch the selected simulator client:
XPL_ROOT=/data/users/jiachangzhang/RoboTwin/XPolicyLab bash scripts/robodojo/launch_sim_client.sh robotwin <task>
XPL_ROOT=/data/users/jiachangzhang/eb_vjepa/repos/RoboDojo/XPolicyLab bash scripts/robodojo/launch_sim_client.sh robodojo <task>
```

`deploy.yml` is an integration template and must be changed to match the checkpoint's metadata (`action_space`, camera, state field order, dimensions). `launch_bridge.sh` starts the CPU-only XPolicyLab websocket bridge, which connects to the separately started model RPC service and does not load weights.

## Measured status

Unit tests run in the remote RoboTwin environment: 6 passed. Native XPolicyLab import checks for `Model`, batch deploy loop, websocket client/server passed. No simulator launched: SAPIEN had missing system Vulkan warnings in the existing env, while RoboDojo requires IsaacSim 5.1 and assets still downloading. RoboDojo bootstrap is ongoing; torch 2.7 pip download is active (large CUBLAS/CUDNN wheels), not stalled at last check. Assets downloader is active via HF mirror/proxy. No benchmark throughput number is claimed until a real service + simulator run.



## SAPIEN render verification (2026-10-15)

Executed `CUDA_VISIBLE_DEVICES=4 /data/users/jiachangzhang/.venv-robotwin/bin/python ROOT/scripts/robodojo/render_robotwin_smoke.py`. Exit 0: 64x64 RGBA frame, RGB min0/max0.8649784, finite pixels and nontrivial image variation. SAPIEN emitted missing system Vulkan library/ICD warnings but its bundled fallback rendered successfully; no environment or system library modification was necessary. Result: `ROOT/experiments/robodojo_setup/robotwin_render_smoke.json`. This proves basic renderer readiness, not a complete RoboTwin task rollout.

Launch flags checked against actual `RoboTwin/scripts/eval_policy_xpolicylab.py`: `--eval_batch`, `--num_workers`, `--test_num`, `--device_id` are supported. The wrapper now defaults simulator to physical GPU4 using CUDA_VISIBLE_DEVICES and local device0. RoboDojo's shell accepts the listed flags and derives vectorized scene count from its selected sim config; `--eval_batch` is accepted but not forwarded to its Python entrypoint. Neither launch script starts model weights.
