# EB-VJEPA — ENVIRONMENT / 基础设施交接文档（v3）

> v3 更新：2026-10-02 10:50（北京时间），by Hermes。v1→v2→v3 变更见文末。
> 项目：Embodied V-JEPA —— 在 DROID robot video 上对 V-JEPA 2.1 ViT-L 做 JEPA continued pretraining；正式训练由 Codex 接手。
> v1→v2 变更：环境完成并全项验证；**AV1 解码重大发现与解法（PyAV）**；checkpoint 稳健续传机制；目录/命令更新。标注【变动】的条目以文件系统实际状态为准。

## 0. TL;DR（先读这个）
- 机器：112.45.47.40，8 x RTX 4090 48GB（共享机，礼让见 §1）
- 项目根：/data/users/jiachangzhang/eb_vjepa/ ；**一切文件写 /data；系统盘 / 已 100% 满**
- 环境：uv venv `envs/eb_vjepa_pretrain`（Python 3.11.16；torch 2.11.0+cu128）【✅ 已全项验证，见 §7】
- 数据：`data/droid_1.0.1`（lerobot v3.0，AV1 编码）—— **✅ 已完整下载（813/813，412.3GB，2026-10-03 08:27；全量 manifest 已重跑，见 §5）**
- Checkpoint：`checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt` —— **✅ 已下载完成+全量加载验证**（10/2 10:42；size 5,151,198,524；sha256 与结构见 §6）
- **训练状态：正式 8 卡训练已于 2026-10-02 11:58:51 启动**（PID 2421560；命令/日志/W&B/校准/冒烟细节见 `experiments/ctp_v1/RUN.md`）——基础设施交接至此收官，训练侧以 RUN.md 为准。

## 1. 机器 / 磁盘 / 使用礼让
- GPU：8 x NVIDIA GeForce RTX 4090 48GB（49140 MiB/卡），驱动 570.86.10，CUDA toolkit 12.8（/usr/local/cuda-12.8）
- CPU：AMD EPYC 7K62，192 线程；RAM 755 GB；Ubuntu 22.04.2，kernel 5.15.0-161
- 磁盘：`/`（sda3 435G）≈100% 满，勿写；`/data`（nvme0n1p1 7T）项目所在，当前 free ~885G；/dev/shm 231G
- 无 sudo；共享机（470+ 用户）：GPU 先用 nvidia-smi 看；psj 组常驻 0-3；长任务 nohup + 日志

## 2. 目录布局（/data/users/jiachangzhang/eb_vjepa/）
```
repos/vjepa2/        @ 204698b45b3712590f06245fbfba32d3be539812 (main)
repos/RoboActualizer/ @ 8cd0f411286f647f60eedeede48182701e65d257（下游评测用）
data/droid_1.0.1/    v3.0 格式，✅ 已完整下载（813 files）
data/manifests/      droid_all.csv + droid_<camera>.csv + droid_summary.json
checkpoints/vjepa2_1/  vjepa2_1_vitl_dist_vitG_384.pt（续传中）+ ctp_download.lock
envs/eb_vjepa_pretrain/  主环境（uv venv）
cache/venv-hf/       下载工具 venv（hf CLI 2.1.1 + modelscope 1.40.1 + av 18.1.0 + imageio-ffmpeg）
cache/{hfhome,xet,modelscope,uv,tmp,pip}/   各类缓存（全在 /data）
tools/               mihomo-my（自建代理实例 :7896）、minode.py
scripts/             make_manifest.py / check_dataloader.py / check_decoder.py / verify_env.sh / verify_nccl.py / min_forward.py / inspect_ckpt.py / ckpt_resume_loop.sh / resume_ctp_checkpoint.sh（后两者为 ckpt 下载器）
logs/                ms_dl.log / env_install*.log / verify_env_manual.log / pip_freeze_20261002.txt / ckpt_resume_loop.log / ckpt_inspect_auto.log（将出现）
wandb/               W&B local dir
```

## 3. 环境（uv；无 conda）
- uv：`/home/jiachangzhang/.local/bin/uv`（0.12.5）；`~/.config/uv/uv.toml`：cache-dir=/data/users/jiachangzhang/uv/cache、index-url=清华 tuna；`UV_PYTHON_INSTALL_DIR` 在 ~/.bashrc。
- 主环境 `envs/eb_vjepa_pretrain`（Python 3.11.16）**【✅ 已装完】**
  激活：`source /data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/activate`
  关键版本（完整快照 logs/pip_freeze_20261002.txt，187 包）：torch 2.11.0+cu128、torchvision、timm 1.0.30、transformers、decord 0.6.0、**av 18.1.0（额外加装，AV1 解码用）**、numpy、wandb、peft 等。
- **torch 装包纪律**：PyPI 默认 torch 是 cu13 栈（需 580+ 驱动）；本机驱动 570 → 一律 `--torch-backend cu128`。例：
  `uv pip install --python .../envs/eb_vjepa_pretrain/bin/python --torch-backend cu128 <pkg>`
- **vjepa2 包 import 注意**：editable 安装只落了元数据（`vjepa2==0.0.2`），**`import vjepa2` 会失败**——训练脚本按 repo 根为 cwd 运行（`from src... / from app...`）或 `PYTHONPATH=repos/vjepa2`。
- ffmpeg：无系统 ffmpeg；静态二进制在
  `cache/venv-hf/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`（ffmpeg 7.0.2，可读 AV1）。
- Python 版本备注：README 用 3.12 示例，但托管 3.12 下载失败，实际 3.11.16（repo 要求 >=3.11）。

## 4. 代码仓库
- `repos/vjepa2` @ `204698b45b3712590f06245fbfba32d3be539812`（main）
  - 训练入口 `app/vjepa_2_1/train.py`；DROID 变体 `app/vjepa_droid/droid.py`；数据管线 `src/datasets/video_dataset.py`（CSV 直读：两列 空格分隔 无表头 `<path> <label>`）
  - 2.1 ViT-L 配置 `configs/train_2_1/vitl16/{pretrain-256px-16f, cooldown-256px-64f}.yaml`（pred_depth=24，与 released ckpt 的差异见 §6）
  - **解码器现状：`video_dataset.py` 与 `droid.py` 都用 decord（本数据读不了，见 §5.3）→ 需要 Codex 适配**
- `repos/RoboActualizer` @ `8cd0f411...`（下游评测用；环境未建，若依赖冲突再建 eb_vjepa_roboactualizer）
- 未改 upstream；改动请用自己的 branch/worktree。

## 5. 数据集 DROID
### 5.1 概况
- 来源 ModelScope `lerobot/droid_1.0.1`（与 HF 同源：info.json md5 一致；总 412,306,732,799 B）
- v3.0：`videos/{camera}/chunk-000/file-XXX.mp4`（每文件含多 episode，单文件时长 ~1h41m）；`data/chunk-000/file-XXX.parquet`（156 个，12.05GB）；`meta/{info.json,stats.json,tasks.parquet,episodes/}`
- 相机：exterior_1_left 312 个 mp4 / 152.5GB；exterior_2_left 310 / 151.5GB；wrist_left 191 / 95.7GB；15 fps；95,658 episodes
- 下载产物命名：完成的 = `file-XXX.mp4`；下载中 = `file-XXX.mp4.incomplete`（manifest 只收完成的）
### 5.2 下载与 manifest
- 下载：✅ 已完成（10/3 08:27；982/982 文件、磁盘 412,306,802,431 B、耗时 31h41m）。日志 `logs/ms_dl.log`；如需重下：断点续传=重跑同命令。
- Manifest：`python3 scripts/make_manifest.py`（幂等，数据全完后重跑）；产物在 `data/manifests/`
### 5.3 ⚠️ 视频编码 = AV1（重大发现）
- 实测（ffmpeg 7.0.2）：`Video: av1 (libdav1d), yuv420p, 320x180, 15 fps, ~633 kb/s`；PyAV 实际解出帧 shape = (270, 320)（h,w；注意与 probe 尺寸不一致，可能带旋转/裁剪元数据，写 transform 时以实际解码为准）
- **decord 无法读 AV1**：decord 0.6.0 与 eva-decord 0.6.1 均报 `cannot find video stream`（bundled ffmpeg 太老）。
- **已验证解法：PyAV（包名 av，18.1.0）能正常解码**——已同时装进 `envs/eb_vjepa_pretrain` 和 `cache/venv-hf`。ffmpeg 7 静态二进制也可用。
- torchvision.io.VideoReader 在当前 torchvision 已移除（改用 torchcodec/其它），不建议走这条路。
- 影响：`VideoDataset`（decord）必须适配（换 PyAV 读取，或换其他可行解码栈）；适配前不要用 repo 自带 dataloader 读本数据（其失败重试逻辑会 randint 死循环）。专用冒烟脚本已备好：`envs/eb_vjepa_pretrain/bin/python scripts/check_decoder.py`

## 6. Checkpoint（V-JEPA 2.1 ViT-L/16 distilled）— ✅ 已完成并验证
- 文件：`checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt`
- 官方 URL：https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
- **size = 5,151,198,524 B（一个字节不差）**；下载完成 2026-10-02 10:42（1.18GB 后单次连续续传完成，未再中断）
- **sha256 = 7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6**（官方未发布对照 hash；此为本地存档值）
- 全量加载验证（torch.load 成功 = 文件完整性 OK；报告 logs/ckpt_inspect_auto.log）：
  - top keys: encoder / predictor / opt / scaler / ema_encoder / epoch / loss / batch_size / world_size / lr
  - encoder: 302 tensors, 304.7M params, 24 blocks（keys 前缀 `module.backbone.`）
  - **ema_encoder: 302 tensors, 304.7M params, 24 blocks —— 官方加载姿势取这个键**（hub factory `checkpoint_key="ema_encoder"`）
  - predictor: 162 tensors, 23.0M；**12 blocks**、8 个 mask tokens、predictor_embed [384,1024]、双输出头 [1664,384]（1664 = 教师维度；官方常量 `vjepa2_1_teacher_embed_dim = 1664`）
  - opt / scaler：空（发布权重不带优化器状态）
- 结构结论（与官方代码逐条对齐过）：
  - 该 ckpt = 从 ViT-G 蒸馏的 ViT-L（hub factory：predictor_depth=12、n_output_distillation=1、return_all_tokens=True）——"1664 头 / 12 层 predictor"是官方规格，不是文件异常；
  - vitl16 config 的 `pred_depth: 24` 与发布权重 12 层不一致（任务书已预判）——以事实为准，**不要改训练逻辑**；加载适配（如用 ema_encoder 初始化 encoder、重建 1024 维 predictor 或输出头适配）由 Codex 实现；
  - repo 中无"非蒸馏"ViT-L 2.1 权重可选；ViT-g/G（1B/2B）与项目规格（300M）不符。

## 7. 验证记录（2026-10-02 05:0x–05:5x）
- `bash scripts/verify_env.sh 2`（→ logs/verify_env_manual.log，rc=0）：
  - python 3.11.16；torch 2.11.0+cu128 | cuda_build 12.8 | available True | ndev 8
  - 8 x RTX 4090 49140 MiB 全部可见；bf16 matmul OK
  - repo imports OK（decord 0.6.0、timm 1.0.30 读入正常；timm.models.layers 有 FutureWarning，无害）
  - NCCL 2-proc allreduce：rank0/rank1 allreduce_ok=True
- `scripts/min_forward.py`：ViT-L forward OK（params 306.0M；in (1,3,16,256,256) → out (1,2048,1024)；bf16 autocast）
- 解码冒烟 `scripts/check_decoder.py`：见本次运行输出（PyAV 读 AV1 OK）
- 包快照 `logs/pip_freeze_20261002.txt`（187 行）
- Checkpoint 全量加载：✅ torch.load 成功；size 精确 5,151,198,524；结构见 §6（logs/ckpt_inspect_auto.log）
- NCCL 8 卡：✅ 独立复核（torchrun --nproc_per_node=8，rank 0-7 allreduce_ok=True，2026-10-02 08:2x）
- 待办验证：dataloader 端到端（需先完成 §5.3 解码适配）

## 8. W&B
- 已登录（~/.netrc）；约定 `WANDB_PROJECT=eb-vjepa`、`WANDB_DIR=/data/users/jiachangzhang/eb_vjepa/wandb`；离线模式可用。

## 9. 网络 / 代理 / 带宽
- 公网出口总吞吐上限实测 ~3.7-4.5 MB/s（国内源/国际源/并发/换节点都撞顶）——不要期待并行提速。
- 代理：系统 mihomo（root，/etc/mihomo）`127.0.0.1:7890`（勿动）；自建实例 `127.0.0.1:7896`（tools/mihomo-my，测速用，可关）；`proxyon/proxyoff` 别名在 ~/.bashrc。
- 大文件走国内源；**xet 全局禁用**（HF_HUB_DISABLE_XET=1；实测 hf_xet 客户端卡死）。
- 环境变量建议：TMPDIR=/data/.../cache/tmp；HF_HOME=/data/.../cache/hfhome；HF_ENDPOINT=https://hf-mirror.com；MODELSCOPE_CACHE=/data/.../cache/modelscope。
- 系统盘 / 只剩几十 MB：一切写 /data。

## 10. 给 Codex 的起步提示与已知适配点
1) **解码器适配（最高优先）**：本数据是 AV1，decord 读不了；PyAV(av) 已在环境里且验证可用。需要改 `src/datasets/video_dataset.py`（VideoDataset.loadvideo_decord）与 `app/vjepa_droid/droid.py` 的读取路径（或提供 PyAV 版本的 reader）。`torchvision.io.VideoReader` 不可用。
2) **启动方式**：官方链以 SLURM/submitit 为中心（config 有 nodes/tasks_per_node 等）；单机 8 卡需适配（torchrun 直启或本地 submitit）。
3) **8 卡通信基线**：`envs/eb_vjepa_pretrain/bin/torchrun --standalone --nproc_per_node=8 scripts/verify_nccl.py`（单机 NCCL 已验证 2-proc 通过；GPU 全空闲）
4) 数据不必等全量：parquet 已基本齐、mp4 陆续落盘（.incomplete 是进行中）；manifest 生成器幂等可重跑。
5) **torch 装包一律 `--torch-backend cu128`**；不要裸装 cu13。
6) pred_depth 12/24 以 ckpt 检查为准（§6）。
7) 不要开始 72h 正式训练（等验证闭环 + 用户确认）。
8) 共享机礼让：先 nvidia-smi，长任务 nohup + 日志。

## 11. 变更记录
- 2026-10-02 01:15 v1：机器侦查/目录/clone/uv+cu128 环境/数据下载启动/脚本部署。
- 2026-10-02 05:55 v2：环境装完并全项验证（torch2.11+cu128/CUDA/8GPU/bf16/NCCL/forward）；**AV1 解码发现与 PyAV 解法（av 已装入两环境）**；checkpoint 稳健续传循环部署；新增 check_decoder.py；各项状态刷新。
- 2026-10-02 10:50 v3：checkpoint 下载完成（5,151,198,524 B 精确、单次续传）+ 全量加载验证（sha256 存档；encoder/ema 24 块×302 张量、predictor 12 块/1664 双头，与官方 hub factory 规格对齐）；8 卡 NCCL 独立复核；数据 125.6GB。
- 2026-10-02 12:05：正式训练启动（PID 2421560，11:58:51；30k 步/72h 先到为准）；本环境文档转入维护状态。
- 2026-10-03 08:30：DROID 全量下载完成（813/813，412.3GB，31h41m）；全量 manifest 重跑；解码抽检 6/6；v1 训练 10-03 07:59 完成（30k 步）。
