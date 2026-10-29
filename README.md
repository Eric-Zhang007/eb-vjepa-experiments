# Embodied V-JEPA / Dream-VLA 实验交接

这是研究代码与恢复入口，不是服务器文件树的镜像。上游仓库通过 **固定 commit + 修改补丁 + 新增文件** 还原；自己的代码、实际配置、实验总表保留在本仓库。

## 新服务器：先跑通评测

Linux GPU 节点上准备 `git`、Python 3.10、`uv`、与 CUDA 12.1/PyTorch 2.5.1 兼容的 NVIDIA 驱动。FlashAttention 编译需要 `nvcc` 和编译工具；MuJoCo 离屏渲染需要 NVIDIA EGL。`env` 命令会安装 Python 依赖，不自动修改系统包。

```bash
git clone https://github.com/Eric-Zhang007/eb-vjepa-experiments.git
cd eb-vjepa-experiments

# 只需指定两个新服务器路径；不要写回旧服务器路径。
python3 handoff.py init --workspace /scratch/$USER/eb-runtime --data-root /scratch/$USER/datasets/rlds
python3 handoff.py sources
python3 handoff.py env
python3 handoff.py assets baseline
python3 handoff.py doctor --gpu

# 先用两个 episode 验证，不改变正式评测协议。
python3 handoff.py eval --model baseline --benchmark clean --gpus 0 --episodes 2
```

若完整 HF checkpoint 尚在上传，`doctor`/模型加载会指出缺失文件；不要把空仓库当作下载成功。公共上游资源的访问受集群网络影响，可以使用集群已有镜像或代理；下载版本记录在来源脚本和 manifest 中。

```bash
# 最终魔改 Dream-JEPA：还需要原版 base 和 grid11k。
python3 handoff.py assets base grid11k dream-jepa
python3 handoff.py eval --model dream-jepa --benchmark clean --gpus 0 --episodes 2

# 正式 clean200 / Plus280，使用已有并行评测接口。
python3 handoff.py assets plus
python3 handoff.py eval --model baseline --benchmark clean --gpus 0,1 --environments 16 --batch-size 1
python3 handoff.py eval --model dream-jepa --benchmark plus --gpus 0,1 --environments 16 --batch-size 1
```

默认推理 batch=1 是可复核起点；可显式调大。去噪4步、执行8步后重规划、各 suite 归一化、语言策略沿用保存代码。没有自动上报评测到 W&B。

## 训练与续训

```bash
python3 handoff.py assets base grid11k baseline rlds
# Dream-JEPA 新训所需 raw feature cache；很大，应先确保 scratch 空间充足。
python3 handoff.py cache --gpus 0,1,2,3,4,5
# 不加 --execute 只输出启动命令，避免 clone/setup 自动占 GPU。
python3 handoff.py train --model dream-jepa --gpus 0,1,2,3,4,5
python3 handoff.py train --model dream-jepa --gpus 0,1,2,3,4,5 --execute
```

这条 `train` 是从 Dream-VLA base **新训**，不是从156500续训。完整续训状态已保留，但旧 run 对源码、路径、数据与迭代器做了严格绑定；请读 [恢复说明](docs/RESTORE.md) 的续训部分，让新服务器 Codex 做显式迁移检查，不能直接关掉校验。

## 目录

| 位置 | 用途 |
|---|---|
| `handoff.py`、`tools/` | 新机器恢复、依赖检查与 CPU 测试 |
| `ctp_full/`、`ctp_v1/` | 两代 continued-pretraining；grid11k 属于 full 配方的新 grid run |
| `elaf/` | E-LAF 历史实现、诊断、统一并行评测接口 |
| `dream_baseline/` | 历史兼容包，部分评测代码会导入；不作为官方 baseline 训练入口 |
| `upstream/` | Dream baseline、最终 Dream-JEPA、RoboActualizer 等差分和新增文件 |
| `scripts/` | 实际训练协调、数据准备、评测、表征与语言诊断 |
| `configs/` | 按实验 run 保存的配置；保留历史值，不伪装为通用配置 |
| `reports/` | 总表、诊断报告、各次评测 summary |
| `environments/` | 原环境包版本记录与可重建 Dream 环境依赖 |
| `provenance/` | commit、文件来源、源码差异、最终训练 handoff |

不包含重复完整上游源码、环境二进制、数据集、模型、大特征缓存或完整逐步日志。迁移、删除和过期 watcher 不进入主仓库；原始备份仍保留在 E 盘。

## 保留模型与记录

| 模型/记录 | Hugging Face 仓库 |
|---|---|
| grid11k encoder | `Isla8899/eb-vjepa-grid11k` |
| LIBERO mixed baseline，200K | `Isla8899/dream-vla-libero-mix4-200k` |
| 最终 Dream-JEPA，156500 | `Isla8899/dream-jepa-dualview-156500` |
| 所有保留的实验原始记录 | 数据集 `Isla8899/eb-vjepa-experiment-records` |

历史 11K 后 encoder、旧 E-LAF 中间权重和 DNFT4RP 产物已按要求退休；对应结果不是仍可加载的模型列表。`reports/DREAM_VLA_EXPERIMENT_HISTORY_20261023.md` 截止10月23日，最终停止状态以 `provenance/training_handoff.json` 的156500为准。clean200/Plus280 是探索性子集，不等同论文完整 benchmark。

## 验证边界

`python3 tools/test_handoff.py` 测路径、固定版本恢复、补丁、幂等性和启动参数。已在 Windows/WSL 执行；真实上游源码恢复也在 WSL 验证。新集群的 GPU、FlashAttention、EGL、闭环 rollout 和续训仍须在目标机器验证，不能将 CPU 测试描述成训练复现成功。

上游代码遵守各自许可证；本仓库不替第三方代码声明新的统一许可证。
