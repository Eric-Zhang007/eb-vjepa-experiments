# 新集群恢复细节

## 路径与来源

`cluster.local.json` 只记录 workspace 和 RLDS 根目录，不提交。`sources` 会在 workspace 构造原有包布局，避免改掉研究代码 import；GitHub 仓库本身保持简洁。

每个上游：拉固定 commit → `git apply --check` → 应用补丁 → 复制新增文件 → 映射旧路径 → 写 `handoff/source_*.json` 回执。重复执行不 reset 工作区。若配方变化，使用新 workspace 或由 Codex 审核已有差异，不自动覆盖修改。复现来源见 `provenance/upstream_repositories.json`。

最后 Dream-JEPA 原型在构造 frozen encoder 时无条件读取训练缓存索引，导致只做评测也依赖几百 GiB 的训练缓存。恢复入口只将该索引改为延迟要求：`encode_videos` 可独立运行，训练调用仍明确检查索引。其余计算路径不改。此差异与路径改写一起记录在 source receipt 中，不能拿改写后的源码通过旧 run 的原始源码指纹校验。

## 环境

已保存环境的 Python、torch、transformers、TensorFlow、robosuite 等版本。部分旧环境没有 pip，`*.pip-freeze.txt` 可能只是错误信息；以 `*.packages.json` 为实际清单，不把失败的 freeze 当 requirements。

`dream-portable.txt` 从实际 Dream 环境抽取核心依赖。torch2.5.1+cu121/torchvision0.20.1+cu121 先安装；FlashAttention2.5.5 单独编译；dlimp 固定到 `040105d256bd28866cc6620621a3d5f7b6b91b46`。不要把 pretraining 的 Python3.11/torch2.11 环境混进 Dream 的 Python3.10 环境。

新架构 GPU/驱动可能需要适配 torch/FlashAttention；先运行 `doctor --gpu`，记录版本差异，然后做两个真实 episode 对照。不要无记录地把 BF16 改为其他精度。EGL 系统库、CUDA toolkit、编译器由集群提供；必要时用集群容器，`env` 不调用 sudo。

## 模型和 benchmark

baseline 模型分片与 action/proprio/noisy-action 模块需配套使用。既有 loader 检测到 merged 模型时不会再叠加 LoRA，不能自行重复应用 adapter。

Dream-JEPA 的 `training_rank_0.pt` 还携带新增 JEPA 参数；单拿 adapter 无法还原。推理还需要原 Dream-VLA base 和 grid11k encoder。`assets` 从指定公开 HF 仓库获取；原版 base/RLDS 使用已有精确 revision 下载脚本。

LIBERO source 来自固定 git commit，Plus 还有独立 assets.zip。Plus 下载器记录实际 revision/SHA；若要严格复现旧环境，应与旧记录中的 asset identity 比较，而非默认新下载资产完全相同。clean/Plus 固定评测 manifest 已保留。manifest 在运行时重新生成路径和自身哈希，避免只替换路径却仍引用旧哈希。

## 从156500精确续训：给新服务器 Codex

1. 下载最终完整 checkpoint；确认六个 `training_rank_*.pt` 及六组 iterator 状态齐全。加载 checkpoint 来读取 contract，不执行优化器更新。
2. 比较保存的 `contract.config`、`sources`、`asset_identities`、world_size、数据 shard inventory 和 normalization stats。当前主线六卡、global batch48，目标200K；不要把训练计数归零。
3. 优先在容器/管理员支持的 bind mount 下映射原来的 workspace、RLDS、run 根目录。不要靠改 TensorFlow iterator 二进制字符串迁移路径。
4. 如使用路径映射后的 portable source，需做显式 relocation receipt：只允许审核过的路径变化和延迟读取索引变化，保持模型/数据数学逻辑不变。保留原始状态副本；不得直接删除 `load_state` 的 contract 检查。
5. 特征缓存已删除，先按 grid11k 原预处理重建。缓存 manifest 中路径、时间戳等可能改变，所以不能假定重建文件的整体 SHA 与历史完全相同。保留重建配置，核对帧内容、特征形状/数值、归一化及 episode 对应后再迁移 contract。
6. 从保存状态恢复 optimizer、scheduler、global step、每 rank RNG 和数据游标；先做极短 continuation，确认步数156500起、采样/学习率/累计batch正确，再正式提交调度任务。

以上是已知迁移工作，不是“所有恢复测试已通过”。本次交付没有目标集群访问，无法替代其实际 GPU/迭代器恢复验证。新的 `train` 命令故意只支持新训，避免误把一次新训叫续训。

## 自动化边界

`sources/env/assets/doctor` 不启动训练。`eval` 与 `cache` 只使用显式传入的 GPU；新训需 `--execute`。执行前应在集群分配的 GPU 节点上运行。旧 watchdog、自动清理和服务器迁移脚本不属于恢复入口。
