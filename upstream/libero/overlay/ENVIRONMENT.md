# DNFT4RP LIBERO Runtime

The remote runtime is under `/data/users/jiachangzhang/DNFT4RP` and uses uv-managed Python 3.9. The LIBERO source is installed editable from `upstream/libero`; datasets are configured through `.libero-config/config.yaml` and must remain under the project directory.

## Environment variables

```bash
cd /data/users/jiachangzhang/DNFT4RP
source .venv-libero-advisor/bin/activate
export PYTHONPATH=$PWD/upstream/libero:$PYTHONPATH
export LIBERO_CONFIG_PATH=$PWD/.libero-config
export HF_ENDPOINT=https://hf-mirror.com
```

The current server package uses MuJoCo 2.3.7 and robosuite 1.4.0. GPU EGL rendering should be enabled only after the server's OpenGL/EGL binding is verified. Import-only checks can use `MUJOCO_GL=disable`; headless rollouts should use the server's validated EGL setting and `MUJOCO_EGL_DEVICE_ID`.

## Dataset download

```bash
python upstream/libero/benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial --use-huggingface
```

Repeat for `libero_object`, `libero_goal`, and `libero_100` after the first suite passes integrity checks. Do not use the original expiring URLs. Keep downloads under the configured `datasets` path.

## Smoke checks

```bash
python upstream/libero/benchmark_scripts/check_task_suites.py
python upstream/libero/test_libero.py
```

These are engineering checks only. They do not substitute for the fixed-seed VLA success-rate evaluation in `EXPERIMENT_CHECKLIST.md`.
