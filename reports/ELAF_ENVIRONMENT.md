# E-LAF server environment inventory

Verified read-only inventory on 2026-10-15 (Asia/Shanghai); RoboDojo preparation subsequently started in isolated paths. ROOT below means `/data/users/jiachangzhang/eb_vjepa`.

## Storage

`df -h /data`: 7.0T total, 5.0T used, 1.7T available (76% used). Earlier 16GB-free handoff is obsolete. No cache/data/model deletion performed.

## RoboTwin

- Repo `/data/users/jiachangzhang/RoboTwin`, commit `30954692d06ba7e89f7a6b76064f4062c488fa81`; XPolicyLab submodule `c37109c500be67d0dea6b36bf7337bbd26e763cd`.
- Environment `/data/users/jiachangzhang/.venv-robotwin/bin/python` (Python 3.10).
- Fresh imports passed: torch 2.4.1+cu121; sapien 3.0.0b1; mplib 0.2.1; curobo 0.0.0; gymnasium 0.29.1; h5py 3.16.0; cv2 4.10.0; open3d 0.18.0; transforms3d 0.4.2; zarr 2.18.3; XPolicyLab.
- Assets occupy 16GB; embodiment assets include Aloha, ARX-X5, Franka, UR5 and Piper. 53 top-level environment Python files. Demo clean/randomized configs exist.
- `data/` is only 8KB, contains no demonstrations. `eval_result/` has task directories but no result files. Existing Reflow server log is not evidence of successful benchmark evaluation.
- SAPIEN import emits missing system Vulkan library/ICD warnings and falls back to bundled components. Rendering is NOT verified. Old install log's failed curobo clone was subsequently repaired: current import works and curobo_install.log records success.
- No remote RoboTwin changes, simulation launches, or package installations performed.

## LIBERO family

Clean source `/data/users/jiachangzhang/DNFT4RP/upstream/libero`; Plus and Pro repos at `ROOT/repos/LIBERO-plus-official` and `ROOT/repos/LIBERO-PRO-official`. Existing Plus/Pro config YAML files present under `ROOT/experiments/action_only/{plus_config,pro_config}`. Simulation not rerun during inventory.

Real demonstration root `/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA`; LeRobot v2.1, 20Hz, two videos per episode:

| Suite | Episodes/parquet files | Frames | MP4 files |
|---|---:|---:|---:|
| spatial | 432 | 52970 | 864 |
| object | 454 | 66984 | 908 |
| goal | 428 | 52042 | 856 |
| 10 | 379 | 101469 | 758 |

Total raw metadata: 1693 episodes / 273465 frames. Preserve existing project selection excluding Goal episode 82 (129 frames): 1692 episodes / 273336 frames. Corruption exclusion comes from skill handoff; it was not re-decoded in this inventory.

## DROID

`ROOT/data/droid_1.0.1`, ~384GB; LeRobot v3.0 metadata, 15Hz. Episode metadata contains 95658 distinct episodes, total 27630375 frames. Metadata references 86 data shards (`file-000` through `file-085`) and all referenced files exist; their actual parquet row count equals 27630375.

IMPORTANT: data directory has 156 parquet shards / 39995463 rows. 70 shards `file-086` through `file-155` are unreferenced by current episode metadata. Do not blindly glob all data shards into training, and do not delete extras. Use episode metadata to choose files and boundaries.

Video inventory: exterior_1_left 312 files; exterior_2_left 310; wrist_left 191. Bytes match saved manifest summary. First-frame PyAV decode of file-000 passed for all three cameras (320x180 AV1). This is a sample decode, not a full corruption audit.

Default `action` is 8D joint position+gripper, whereas `action.original` is 7D Cartesian and gripper; Cartesian position and velocity columns also exist. Choose action semantics explicitly before mixing DROID with LIBERO. Metadata sample had an empty `tasks` string; language_instruction columns exist in frame schema. Language completeness still needs measurement.

## RoboDojo setup (background; NOT ready yet)

Official repository corroborated by RoboTwin README and RoboDojo README:
`https://github.com/RoboDojo-Benchmark/RoboDojo`, commit `09bd5f5e762b949f5858ece31347ce62057b3a5e`.

- Repo cloned to `ROOT/repos/RoboDojo`; recursive IsaacLab/curobo/XPolicyLab submodule bootstrap in progress.
- Isolated Python3.11 environment created at `ROOT/envs/robodojo`.
- Upstream stack: IsaacSim5.1, IsaacLab2.3, torch2.7/cu128. Existing RoboTwin Python3.10/SAPIEN environment is intentionally separate.
- System glibc2.35, NVIDIA570.86.10, CUDA12.8 nvcc present. cmake and ffmpeg commands absent; background script installs env-local cmake and imageio-ffmpeg binary.
- User-owned background installer PID 965771 and asset downloader PID 965770 were confirmed alive once. Bootstrap git PID953697 and torch pip PID959834 still running at that check. PIDs are historical and must be rechecked with `/proc/PID/cmdline`.
- Scripts: `scripts/robodojo/setup_remote.sh`, `download_assets.py`, `render_smoke.py`; copied to corresponding ROOT path. Shell syntax and Python compilation pass.
- Assets download only official dataset `RoboDojo-Benchmark/RoboDojo` `Assets/**`, mirror first then local mihomo HTTP proxy7890 fallback. Resume supported by HuggingFace cache.
- Installer waits for existing bootstrap processes, installs pinned dependencies, checks imports/pip consistency, waits for asset completion, runs upstream doctor (Conda/Isaac checks skipped because venv is used; explicit Isaac imports performed separately), then basic 64x64 RGB headless render. This render does not claim task or policy evaluation.
- No simulator-ready claim until `setup.exitcode == 0`, doctor JSON has no failures, and render_smoke.json says passed. No demonstrations are downloaded by Assets-only job; training data selection remains open.

Remote status and retry commands:

```bash
cd /data/users/jiachangzhang/eb_vjepa
cat experiments/robodojo_setup/{setup,assets}.pid
cat experiments/robodojo_setup/{setup,assets}.exitcode
 tail -n 40 experiments/robodojo_setup/setup.log
 tail -n 40 experiments/robodojo_setup/assets.log
cat experiments/robodojo_setup/doctor.json
cat experiments/robodojo_setup/render_smoke.json
# After diagnosing a failed job, rerun; installer uses flock to avoid duplicates.
bash scripts/robodojo/setup_remote.sh
# Asset retry (resumes existing download):
envs/eb_vjepa_pretrain/bin/python scripts/robodojo/download_assets.py
```

SSH used: `ssh -i C:/Users/zjc/.ssh/dnft4rp_ed25519 -o BatchMode=yes 112.45.47.40`.

No AGENTS.md applied to modified files: root scans found only DNFT4RP/AGENTS.md (outside owned work) and RoboTwin GR00T_N17 subtree instructions (not touched). No credentials or sudo password stored.
