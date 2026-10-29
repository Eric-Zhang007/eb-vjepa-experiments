# Dream-VLA baseline 复现以来的实验汇总

数据截点：2026-10-23 20:20:50（服务器北京时间）；范围：2026-10-15 起的模型训练、重测和机制诊断。

本文件按已保存配置、逐步日志、checkpoint 元数据和逐 episode 记录整理；“—”表示没有该项已核验结果，运行中的数值仅代表截点。

## 1. 名称、数据与统计口径

| 项目 | 本报告采用的定义 |
| --- | --- |
| Dream baseline | 官方 Dream-VLX 训练入口及模型，修正训练 bug；四套 LIBERO 混训的单模型，200K optimizer steps，并非官方四套独立训练配方的逐项复刻。 |
| E-LAF v0/v1/v2 | Dream 主干 LoRA + 32 Current + 32 Delta + 8 Action tokens；共享可学习 Resampler，Delta 与 Action 双 Flow。 |
| grid11k action-only | 冻结 grid encoder 11K + RoboActualizer 派生动作头；不是 E-LAF，也不是官方 V-JEPA Policy。 |
| grid11k latent WM | 在上述动作头上借鉴 V-JEPA Policy 的 predictor/current-position K/V 交互；不使用 Dream 主干。 |
| grid11k small scratch | 独立双视角、小型随机初始化 predictor/action expert，仍是自有实现，文本实际为 umT5。 |
| 当前 Dream 双头 | 回到官方 Dream residual MLP 动作头；加入双视角 Current JEPA tokens 和固定绝对 Future latent Flow。 |
| encoder 11K / 103233 | 视觉编码器预训练步数；不能与下游 action-head、WM、E-LAF 的训练步数相加后混称。 |
| clean200 / Plus280 | 探索性子集：clean 40 tasks×5 trials；Plus 7 类×40 episodes，不是论文完整 benchmark。 |
| 损失窗口 | E-LAF：末 100 个逐步记录的逐项中位数；WM/scratch：末 100 个日志点（每 10 步一次）；各项中位数不保证相加等于 total 中位数。 |
| 数据分支 | Dream mixed baseline 使用官方 RLDS；E-LAF 使用其缓存；RoboActualizer 派生线路使用 1,692 条 LeRobot 示范、排除 Goal episode82，不能与 1,693 条 RLDS 混称。 |
| 跨模型 loss | Action 归一化、时间采样、预测长度、precision 和 reduction 不完全相同；Delta FM、future MSE、future L1 不是同一种指标。 |

说明：下表同时保留模型身份和统计口径，避免把不同训练目标或不同评测协议的数值视为严格对照。

## 2. 实际训练与分叉总账

| ID | 实验 / run 目录 | 初始化 / 父实验 | 已执行步数区间 | 状态 | 训练 W&B ID |
| --- | --- | --- | --- | --- | --- |
| D-pre8 | 单套、8卡初试 | base | 未保留完整终点 | 停止，无 checkpoint | p4hhy9c2 |
| D-pre6 | 单套、6卡初试 | base | 保留5,000 | 早于 Flow-head DDP 修复 | lz3n1xc3 |
| D0 | Dream mixed baseline | base | 0→200,000 | 完成 | mrmlszkq |
| E0 | elaf_v0_grid11k_72token | base 新训 | 0→4,363 | 后改为 v1 | 7rqbbuu3 |
| E1 | elaf_v1_grid11k_72token | base 新训 | 0→19,287 | 失稳后暂停 | gtnxs7ir |
| ER | elaf_recovery_from5000_20261019 | E1 step5,000；新 optimizer | 新计数0→1,000 | 短程恢复完成 | ifzfip50 |
| EF | elaf_fresh_flowclip_2k_20261019 | base 新训 | 0→34,021 | 先2K，后续训失稳暂停 | j2mi08sz |
| EZ | elaf_lora_zero_control_20261020 | EF step3,000 | 3,000→6,000 | LoRA LR=0 对照完成 | v0h0u6fl |
| V2 | elaf_v2_spatial_staged_20261020_run | base 新训 | 0→12,742 | 三阶段主线；随后分叉 | uql4sld4 |
| V2-H | elaf_v2_lora10x_20261020 | V2 step12,742 | 12,742→15,000 | LoRA LR×10 分叉 | d6ntut6h |
| V2-X | elaf_v2_low_lr_from15k_20261020 | V2-H step15,000 | 15,000→15,043 | 起点被用户纠正；废弃分叉 | hmpqlfwz |
| V2-L1 | elaf_v2_low_lr_from12742_20261020 | V2 step12,742 | 12,742→20,000 | 回到低 LR 主线 | s8e0zbbx |
| V2-L2 | elaf_v2_low_lr_20k_to30k_20261020 | V2-L1 step20,000 | 20,000→30,000 | 低 LR 续训 | eukpfjfv |
| V2-L3 | elaf_v2_low_lr_to200k_20261021 | V2-L2 step30,000 | 30,000→42,015 | 用户停止，未完成200K | mqkdrbkj |
| W20 | grid11k_latent_wm_20261021 | action-only20K warm start | WM0→20,000 | 完成 | d7c7b857 |
| W57 | grid11k_latent_wm_extend57k_20261021 | W20完整状态 | WM20,000→57,000 | 完成 | d7c7b857 |
| S21 | grid11k_small_scratch_20261022 | predictor/action 随机初始化 | 0→21,360 | 完成 | 991f1d35 |
| D-J | dream_jepa_dualview_20261023 | Dream base 新训 | 0→3,300 | 截点仍在训练 | lot7ed8l |

说明：V2-H 与 V2-L1 是从同一 12,742 步分出的两条路线，W57 的 57K 则是在原 action-only20K 之外累计的 WM 更新。

## 3. E-LAF 实际优化配置

| ID | 卡×micro-batch×累积=global | 目标 LR：LoRA / Resampler / 新模块 | 调度 | warmup | Resampler/Delta | clip=1 分组 | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 6×1×10=60 | 5e-05 / 0.0001 / 0.0002 | cosine | 3.0% | BF16 autocast | global | S01 |
| E1 | 6×8×1=48 | 0.0001 / 0.0001 / 0.0001 | multistep | 0% | BF16 autocast | global | S02 |
| ER | 6×8×1=48 | 2e-05 / 5e-05 / 0.0001 | warmup_constant | 200 | FP32 | group | S03 |
| EF | 6×8×1=48 | 2e-05 / 5e-05 / 0.0001 | warmup_constant | 200 | FP32 | flow | S04 |
| EZ | 6×8×1=48 | 0 / 5e-5 / 1e-4 | warmup_constant | 200 | FP32 | flow | S05 |
| V2 | 6×8×1=48 | S1:1e-5/5e-5/1e-4；S3:2e-06/2e-05/5e-05 | three_stage | 200 | FP32 | flow | S06 |
| V2-H | 6×8×1=48 | S1:1e-5/5e-5/1e-4；S3:2e-05/2e-05/5e-05 | three_stage | 200 | FP32 | flow | S07 |
| V2-X | 6×8×1=48 | S1:1e-5/5e-5/1e-4；S3:2e-06/2e-05/5e-05 | three_stage | 200 | FP32 | flow | S08 |
| V2-L1 | 6×8×1=48 | S1:1e-5/5e-5/1e-4；S3:2e-06/2e-05/5e-05 | three_stage | 200 | FP32 | flow | S09 |
| V2-L2 | 6×8×1=48 | S1:1e-5/5e-5/1e-4；S3:2e-06/2e-05/5e-05 | three_stage | 200 | FP32 | flow | S10 |
| V2-L3 | 6×8×1=48 | S1:1e-5/5e-5/1e-4；S3:2e-06/2e-05/5e-05 | three_stage | 200 | FP32 | flow | S11 |

说明：v0 的 global batch60 与后续48不同，v1 对齐 baseline LR/调度后仍失稳，而恢复与 fresh 试验又显式改变了精度和裁剪。

## 4. 非 E-LAF 训练配置

| ID | 模型与初始化 | global batch | 更新 | LR / 调度 | Action / Future | 相机 / 文本 |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | Dream base + r32/α16 all-linear LoRA，官方 MLP head | 48 | 200,000 | 1e-4；100K×0.1；无 warmup | H8 Action FM；无 Future | 原生双图 + proprio；Dream 文本 |
| W20 | 12层 DiT-S predictor/action；复用动作头468个 tensors | 240 | 新增20,000 | 2e-4；5% warmup；cosine→2e-6 | H16 Action FM + 4端点 normalized frozen-latent MSE，1:1 | 双相机横拼224×448；4帧间隔2；umT5 |
| W57 | W20完整续训；保留模型和 optimizer | 240 | 再37,000，累计57K | 从2e-6，500步升至5e-5，cosine→1e-5 | 同W20 | 同W20 |
| S21 | 约63.606M可训练参数；predictor/action随机 | 128 | 21,360 | 1e-4；5% warmup；cosine→1e-6 | H32 Action FM + 4端点 per-token LayerNorm L1，1:1 | 独立双视角224×224；4帧间隔2；实际umT5-XXL |
| D-J | Dream base + 官方 MLP action/future heads + 输入Resampler | 48 | 计划200,000；运行中 | 1e-4；100K×0.1；无 warmup | H8 Action FM + 固定绝对Future latent FM，1:1 | 原生双图；JEPA双视角各16；Dream文本 |

说明：W20 是 warm-start 增量实验，S21 才是随机初始化下游模型的配方试验，但 S21 没有用到尚未就绪的 T5 权重。

## 5. 完成的闭环评测总表

| ID | 被评 checkpoint step | 去噪步数 | 每次执行动作数 | LIBERO clean200 | LIBERO-Plus280 | 协议族 | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | 200000 | 4 | 8 | 195/200（97.50%） | 171/280（61.07%） | Dream | S12 / S13 |
| G11-4 | 20000 | 4 | 10 | 165/200（82.50%） | 156/280（55.71%） | Robo | S14 / S15 |
| G11-10 | 20000 | 10 | 10 | 157/200（78.50%） | 155/280（55.36%） | Robo | S16 / S17 |
| G103-4 | 20000 | 4 | 10 | 102/200（51.00%） | 84/280（30.00%） | Robo | S18 / S19 |
| V2-H14.5 | 14500 | 4 | 8 | 36/200（18.00%） | 16/280（5.71%） | Dream | S20 / S21 |
| V2-L30 | 30000 | 4 | 8 | 46/200（23.00%） | 22/280（7.86%） | Dream | S22 / S23 |
| W20 | 20000 | 4 | 10 | 172/200（86.00%） | 171/280（61.07%） | Robo | S24 / S25 |
| W57-4 | 57000 | 4 | 10 | 167/200（83.50%） | 179/280（63.93%） | Robo | S26 / S27 |
| W57-10 | 57000 | 10 | 10 | 173/200（86.50%） | 177/280（63.21%） | Robo | S28 / S29 |
| S21 | 21360 | 4 | 16 | 163/200（81.50%） | 136/280（48.57%） | Robo | S30 / S31 |

说明：以上成功数已按逐 episode 记录复算且无重复，Dream 与 Robo 两种协议的差异见下一表，不能把所有行当作完全相同条件的消融。

## 6. 闭环评测协议差异

| 项目 | Dream baseline / E-LAF v2 | grid action-only / latent WM | small scratch |
| --- | --- | --- | --- |
| clean horizon：Spatial/Object/Goal/Long | 220 / 280 / 300 / 520 | 400 / 400 / 400 / 700 | 400 / 400 / 400 / 700 |
| settling steps | 10 | 30 | 30 |
| Action chunk | 8 | 16 | 32 |
| 执行后重规划 | 8 | 10 | 16 |
| 默认去噪步数 | 4 | 4；另有10步消融 | 4 |
| clean instruction | task.language | task.language | task.language |
| Plus instruction | canonical_clean：非语言扰动用对应clean指令，语言类保留改写（Dream与v2同一launcher分支） | 保留manifest BDDL原始/改写指令 | 保留manifest BDDL原始/改写指令 |
| 动作/状态归一化 | Dream官方bounds_q99；对应suite统计 | RoboActualizer min/max；继承grid11k stats | 继承grid11k min/max stats |
| 评测子集 | clean200 / Plus280 | clean200 / Plus280 | clean200 / Plus280 |

说明：特别是 Plus 非语言类的指令替换、settling 和 horizon 不同，会影响跨协议比较的解释。

## 7. clean200 分 suite 结果

| ID | Spatial | Object | Goal | LIBERO-10 |
| --- | --- | --- | --- | --- |
| D0 | 49/50（98.00%） | 50/50（100.00%） | 46/50（92.00%） | 50/50（100.00%） |
| G11-4 | 49/50（98.00%） | 48/50（96.00%） | 44/50（88.00%） | 24/50（48.00%） |
| G11-10 | 49/50（98.00%） | 47/50（94.00%） | 40/50（80.00%） | 21/50（42.00%） |
| G103-4 | 38/50（76.00%） | 30/50（60.00%） | 32/50（64.00%） | 2/50（4.00%） |
| V2-H14.5 | 7/50（14.00%） | 13/50（26.00%） | 15/50（30.00%） | 1/50（2.00%） |
| V2-L30 | 9/50（18.00%） | 18/50（36.00%） | 18/50（36.00%） | 1/50（2.00%） |
| W20 | 47/50（94.00%） | 49/50（98.00%） | 45/50（90.00%） | 31/50（62.00%） |
| W57-4 | 49/50（98.00%） | 49/50（98.00%） | 43/50（86.00%） | 26/50（52.00%） |
| W57-10 | 49/50（98.00%） | 50/50（100.00%） | 44/50（88.00%） | 30/50（60.00%） |
| S21 | 47/50（94.00%） | 49/50（98.00%） | 37/50（74.00%） | 30/50（60.00%） |

说明：每个 suite 固定50个 episodes，14.5K为高LR分支、30K为低LR分支，不能解释为同一模型的连续提升曲线。

## 8. Plus280 分扰动类别结果

| ID | 视角 | 初始状态 | 语言 | 光照 | 背景 | 传感噪声 | 布局 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | 12/40（30.00%） | 20/40（50.00%） | 20/40（50.00%） | 34/40（85.00%） | 32/40（80.00%） | 23/40（57.50%） | 30/40（75.00%） |
| G11-4 | 16/40（40.00%） | 22/40（55.00%） | 25/40（62.50%） | 29/40（72.50%） | 23/40（57.50%） | 13/40（32.50%） | 28/40（70.00%） |
| G11-10 | 15/40（37.50%） | 28/40（70.00%） | 22/40（55.00%） | 27/40（67.50%） | 20/40（50.00%） | 14/40（35.00%） | 29/40（72.50%） |
| G103-4 | 10/40（25.00%） | 13/40（32.50%） | 16/40（40.00%） | 18/40（45.00%） | 9/40（22.50%） | 7/40（17.50%） | 11/40（27.50%） |
| V2-H14.5 | 0/40（0.00%） | 4/40（10.00%） | 5/40（12.50%） | 1/40（2.50%） | 1/40（2.50%） | 0/40（0.00%） | 5/40（12.50%） |
| V2-L30 | 0/40（0.00%） | 3/40（7.50%） | 4/40（10.00%） | 3/40（7.50%） | 4/40（10.00%） | 1/40（2.50%） | 7/40（17.50%） |
| W20 | 18/40（45.00%） | 27/40（67.50%） | 24/40（60.00%） | 31/40（77.50%） | 23/40（57.50%） | 21/40（52.50%） | 27/40（67.50%） |
| W57-4 | 22/40（55.00%） | 35/40（87.50%） | 25/40（62.50%） | 30/40（75.00%） | 22/40（55.00%） | 18/40（45.00%） | 27/40（67.50%） |
| W57-10 | 21/40（52.50%） | 29/40（72.50%） | 24/40（60.00%） | 30/40（75.00%） | 26/40（65.00%） | 18/40（45.00%） | 29/40（72.50%） |
| S21 | 21/40（52.50%） | 26/40（65.00%） | 15/40（37.50%） | 22/40（55.00%） | 17/40（42.50%） | 14/40（35.00%） | 21/40（52.50%） |

说明：每类40个 episodes，单次成功对应2.5个百分点，类别结果均由原始记录重算。

## 9. Plus280 分 suite 结果

| ID | Spatial | Object | Goal | LIBERO-10 |
| --- | --- | --- | --- | --- |
| D0 | 43/70（61.43%） | 46/70（65.71%） | 40/70（57.14%） | 42/70（60.00%） |
| G11-4 | 49/70（70.00%） | 54/70（77.14%） | 33/70（47.14%） | 20/70（28.57%） |
| G11-10 | 48/70（68.57%） | 55/70（78.57%） | 31/70（44.29%） | 21/70（30.00%） |
| G103-4 | 34/70（48.57%） | 35/70（50.00%） | 13/70（18.57%） | 2/70（2.86%） |
| V2-H14.5 | 5/70（7.14%） | 2/70（2.86%） | 8/70（11.43%） | 1/70（1.43%） |
| V2-L30 | 7/70（10.00%） | 6/70（8.57%） | 9/70（12.86%） | 0/70（0.00%） |
| W20 | 52/70（74.29%） | 61/70（87.14%） | 34/70（48.57%） | 24/70（34.29%） |
| W57-4 | 55/70（78.57%） | 59/70（84.29%） | 38/70（54.29%） | 27/70（38.57%） |
| W57-10 | 53/70（75.71%） | 62/70（88.57%） | 34/70（48.57%） | 28/70（40.00%） |
| S21 | 27/70（38.57%） | 59/70（84.29%） | 24/70（34.29%） | 26/70（37.14%） |

说明：Plus 的 suite 统计与上一表使用同一批280条记录，仅改变分组维度。

## 10. 早期无效评测与协议定位

| 实验 / 干预 | 结果 | 样本 | 效力 / 问题 | 来源 |
| --- | --- | --- | --- | --- |
| baseline200K 最初错误协议 | 26/200 = 13.00% | clean200 | 作废：实际1步而非4步；还叠加语言、模拟器、settling/horizon差异 | S32 |
| 仅切4步的早期重跑 | 不纳入正式成绩 | 中间重跑 | 仍有BDDL语言与模拟器不一致；被corrected取代 | S33 |
| 原生官方 Goal 控制 | 10/10 | 10 tasks×1初态 | 小样本接口控制 | S34 |
| 共享评测器+官方模拟器+旧语言 | 8/10 | 同Goal控制 | 语言尚未修正 | 同上 |
| 共享评测器+旧模拟器+旧语言 | 3/10 | 同Goal控制 | 模拟器依赖差异明显 | 同上 |
| 共享评测器+官方模拟器+canonical语言 | 10/10 | 同Goal控制 | 修正语言后的控制 | 同上 |
| 最终生产RPC控制 | 9/10 | 同Goal控制 | 并行数值/轨迹敏感；非完整成绩 | 同上 |
| 单套5K checkpoint / tiny模型 / base Plus smoke | 仅接口验证 | 少量episodes | 不可当作修正baseline或成熟模型的质量成绩 | DREAM_VLA_OFFICIAL_TRAINING.md |

说明：13%已被明确标记为无效协议结果，正式修正后的 Dream baseline 是195/200，而不是模型训练本身仅有13%成功率。

## 11. E-LAF 各 run 终点训练统计

| ID | 终点step | Action loss | Delta loss | Total loss | LoRA preclip范数 | 秒/update | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 4363 | 1.2073025 | 0.5493224 | 1.7770933 | — | 4.8323662 | S35 |
| E1 | 19287 | 1.1889203 | 1.000456 | 2.1894636 | — | 1.1942204 | S36 |
| ER | 1000 | 1.0233617 | 0.9890142 | 2.0103428 | 1618.2487183 | 1.2927496 | S37 |
| EF | 34021 | 3.0743436 | 1.9364126 | 5.0270814 | 2.003e+09 | 1.3676334 | S38 |
| EZ | 6000 | 0.1969435 | 0.0672551 | 0.2670217 | 2.8003618 | 1.1485423 | S39 |
| V2 | 12742 | 0.1594697 | 0.1978725 | 0.3520836 | 2.2911315 | 1.3245044 | S40 |
| V2-H | 15000 | 0.154759 | 0.2640961 | 0.4130066 | 19.7290192 | 1.1727603 | S41 |
| V2-X | 15043 | 0.1522653 | 0.2693055 | 0.4219305 | 19.0931473 | 1.4696018 | S42 |
| V2-L1 | 20000 | 0.1461101 | 0.1918972 | 0.3368084 | 2.165884 | 1.3169856 | S43 |
| V2-L2 | 30000 | 0.1358129 | 0.1894079 | 0.3235243 | 2.059144 | 1.2006643 | S44 |
| V2-L3 | 42015 | 0.1418205 | 0.196811 | 0.3404233 | 9.2302704 | 1.1887704 | S45 |

说明：这里均为该run末100个记录的逐项中位数，V2-X仅运行43步，因此使用其全部43个记录。

## 12. Baseline、WM、scratch 的终段损失

| ID | 窗口 | 聚合口径 | Action loss | Future loss | LR | 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | 195,001–200,000 | 跨6卡、30,000个microbatch均值 | 0.0211598 | 不适用 | 1e-05 | S46 |
| D0 | W&B step199,990 | rank0最后一次日志单点 | 0.0054321 | 不适用 | 1e-05 | S47 |
| W20 | 19010–20000 | 末100个日志点中位数 | 0.0093807 | 0.3478998 | 2.331e-06 | S48 |
| W57 | 56010–57000 | 末100个日志点中位数 | 0.0064988 | 0.3248194 | 1.002e-05 | S49 |
| S21 | 20370–21360 | 末100个日志点中位数 | 0.0263607 | 0.3265852 | 1.145e-06 | S50 |

说明：baseline的0.00543单点不能替代0.02116跨卡窗口均值，WM/scratch的更低Action数值也不能直接解释成更高成功率。

## 13. 两轮失稳的时间窗口

| ID | step区间[左闭右开] | 记录数 | Action | Delta | global preclip | LoRA preclip | LoRA相对更新量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0-500 | 499 | 0.320537 | 0.9952581 | 6.3149314 | — | — |
| E1 | 500-1000 | 500 | 0.2356794 | 0.1815443 | 5.7339714 | — | — |
| E1 | 1000-1500 | 500 | 1.1003245 | 0.1949262 | 40.3031979 | — | — |
| E1 | 1500-2000 | 500 | 1.2041496 | 0.9237967 | 159.6370544 | — | — |
| E1 | 2000-3000 | 1000 | 1.1569793 | 0.9724437 | 891.9866333 | — | — |
| EF | 2000-3000 | 1000 | 0.2296386 | 0.1156139 | 7.0953592 | 3.468897 | 0.0004151 |
| EF | 3000-3500 | 500 | 0.2284441 | 0.1082138 | 9.301957 | 5.6807995 | 0.0004547 |
| EF | 3500-4000 | 500 | 0.2263411 | 0.1291411 | 22.8574237 | 21.2308025 | 0.0003576 |
| EF | 4000-4500 | 500 | 0.2426839 | 0.177189 | 75.0121356 | 73.0410118 | 0.0002769 |
| EF | 4500-6000 | 1500 | 0.2094584 | 0.2815802 | 247.3008189 | 240.126442 | 0.000211 |
| EF | 5800-5850 | 50 | 0.1998015 | 0.3361222 | 400.7939631 | 418.4362183 | 0.0001763 |
| EF | 5850-5900 | 50 | 1.3708125 | 0.6147123 | 388.4079059 | 345.7859497 | 0.0017061 |
| EF | 6000-7000 | 1000 | 1.1880819 | 0.4124158 | 427.5472375 | 387.5810852 | 0.0001475 |
| EF | 33000-35000 | 1022 | 1.5791361 | 3.2496118 | 1.76e+09 | 1.732e+09 | 9.159e-05 |

说明：fresh轮在3K后LoRA梯度先增长，约5,850步Action突变，后期巨梯度不能等同于每一步参数更新同比例变大。

## 14. 5000起点恢复与固定batch优化检查

| 检查 | 样本 / 条件 | Action | Delta模型MSE | 解析MSE | 零速度MSE | 其他 |
| --- | --- | --- | --- | --- | --- | --- |
| 退化step5000恢复1000步 | 新optimizer；200warmup；FP32；分组clip | 1.0233617 | 0.9890142 | — | — | 不是失稳前恢复 |
| 恢复终点解析检查 | 固定4样本；u=s=.5 | — | 0.98569 | 0.0012598 | 0.998078 | Delta RMS=.0177468 |
| 全新固定batch：update0 | 4样本；固定噪声；无aug | 1.816097 | 1.661134 | 0.017317 | — | 新LoRA/Resampler/heads |
| 全新固定batch：update60 | 同4样本与噪声 | 0.012743 | 0.393323 | 0.015278 | — | 仅证明可优化/记忆小batch |

说明：恢复模型没有学好简单去噪映射，而新初始化小batch可以降低两项损失，但这不是动力学泛化证据。

## 15. 随机Query Resampler的冗余诊断

| step | Current unique能量 | Delta unique能量 | attention熵/max | per-head JSD(nats) | CKA均值 | Current有效秩 | Delta RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.01985% | 0.02674% | 0.9999965 | 9.794e-05 | 0.9999979 | 7.8929181 | 0.0302272 |
| 1000 | 0.08949% | 0.18284% | 0.9992829 | 0.0003137 | 0.9994767 | 9.4228296 | 0.0482696 |
| 1500 | 0.28522% | 0.65047% | 0.9982594 | 0.0008633 | 0.9968812 | 9.6491356 | 0.0455473 |
| 2000 | 0.38743% | 0.88993% | 0.9972268 | 0.0011555 | 0.9945973 | 9.684659 | 0.0420366 |

说明：固定1024对样本显示slot开始分化但共同变化仍占主导，这些attention与能量指标比仅凭CKA更直接。

## 16. Step2000 slot-specific消融

| Action时间s | 原始Action MSE | 移除Current/Delta特异变化后 | 相对变化 |
| --- | --- | --- | --- |
| 0.25 | 0.13776 | 0.13813 | +0.27% |
| 0.5 | 0.21841 | 0.21948 | +0.49% |
| 0.75 | 0.31959 | 0.31951 | −0.03% |

说明：32个固定样本、保留slot固定偏置与共同成分的离线消融表明，当时动作预测对slot-specific变化不敏感，而不是整个Future分支无用。

## 17. Step2000 Future信息路径消融

| 干预 | Action velocity MSE | 相对正常 |
| --- | --- | --- |
| 正常Joint Flow | 0.214463 | 参考 |
| 打乱干净Delta，保留各样本噪声 | 0.215986 | +0.71% |
| 逐层禁止Action读取Delta | 0.217412 | +1.38% |
| 逐层禁止Action读取Current JEPA | 0.564316 | +163.13% |
| 逐层禁止所有非Delta token读取Delta | 0.228306 | +6.45% |

说明：固定32样本、s=.5及10个u中点、原生FlashAttention的最终有效诊断支持Current信息使用更明显，但不是闭环成功率消融。

## 18. Step2000 Delta按时间的解析基线

| u | 模型Delta MSE | −d_u/(1−u) MSE | 零速度MSE |
| --- | --- | --- | --- |
| 0.05 | 0.073867 | 0.002075 | 1.003167 |
| 0.15 | 0.068261 | 0.002592 | 1.003167 |
| 0.25 | 0.064092 | 0.003329 | 1.003167 |
| 0.35 | 0.061521 | 0.004432 | 1.003167 |
| 0.45 | 0.062264 | 0.00619 | 1.003167 |
| 0.55 | 0.065977 | 0.009247 | 1.003167 |
| 0.65 | 0.07705 | 0.015286 | 1.003167 |
| 0.75 | 0.104231 | 0.029962 | 1.003167 |
| 0.85 | 0.185789 | 0.083226 | 1.003167 |
| 0.95 | 0.548756 | 0.749038 | 1.003167 |

说明：九个中点上模型不及无参数解析去噪器，接近u=1时解析式自身放大误差，不能仅凭最后一行宣称学到了动力学。

## 19. 健康/失稳checkpoint逐层诊断及LoRA零LR对照

| 项目 | EF step3000 | EF step34021 | EZ：3000→6000 |
| --- | --- | --- | --- |
| 固定batch Action MSE | 0.1751 | 2.1502 | 终段训练中位数 0.1969435 |
| 固定batch Delta MSE | 0.03939 | 1.9142 | 终段训练中位数 0.0672551 |
| Action→LoRA梯度范数 | 5.295 | 239.976 | LoRA LR=0，保留requires_grad |
| Delta→LoRA梯度范数 | 3.136 | 67871895 | 其他组维持原配置 |
| 两任务LoRA gradient cosine | -0.00233 | -0.03435 | 不重置非LoRA状态 |
| block27 Q LoRA/base RMS | 0.89 | 45.383 | 完整对照已完成 |
| block27 K LoRA/base RMS | — | 26.204 | LoRA权重逐项hash不变：True |

说明：健康/失稳列为固定4样本×3时间点的诊断中位数，零LR列为训练终段统计，两者不同口径，但冻结对照确实稳定完成到6K。

## 20. v2空间锚定初始化对照

| 指标 | 原随机Query初始化 | 空间锚定初始化 |
| --- | --- | --- |
| Current unique energy | 0.0001985 | 0.80135 |
| Delta unique energy | 0.0002674 | 0.9153363 |
| Current CKA均值 | 0.9999779 | 0.4646646 |
| Current有效秩 | 7.8929162 | 24.2141075 |
| Delta有效秩 | 30.5920582 | 160.4510193 |
| Delta RMS | 0.0302272 | 0.2315802 |
| per-head Query JSD | 9.794e-05 | 0.0183921 |
| attention熵/max | 0.9999966 | 0.9971032 |

说明：同一1024样本、FP32重测，真实网格2×24×24池化为2×4×4，Z=P+0.1×attention残差保留空间结构，unique energy此表以0–1比例表示。

## 21. v2三阶段实际执行

| 阶段 | step边界 | LoRA | Resampler LR | 新模块LR | 状态迁移证据 |
| --- | --- | --- | --- | --- | --- |
| S1 | 0–2999 | 1e-5；200步warmup | 5e-05 | 0.0001 | base新初始化；A随机/B零 |
| S2 | 3000–9999 | requires_grad=False；权重冻结 | 5e-05 | 0.0001 | step3000与10000 LoRA hash完全相同 |
| S3低LR | 10000起 | 1000步warmup→2e-6 | 2e-05 | 5e-05 | 重新创建LoRA moments；保留非LoRA状态；cosine最低比0.1 |
| S3高LR分叉 | 12742→15000 | 目标2e-5 | 2e-05 | 5e-05 | 独立V2-H；14500评测来自此分支 |

说明：三阶段切换已有transitions记录，低LR主线从12,742重新分叉续到42,015，不经过高LR15K权重。

## 22. v2阶段性表征监控

| ID | step | Delta RMS | Current有效秩 | Delta有效秩 | CKA均值 | Current unique | Delta unique |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | 0 | 0.2315802 | 24.2141075 | 160.4510193 | 0.4646646 | 0.80135 | 0.9153363 |
| V2 | 3000 | 0.2019273 | 23.7155399 | 252.9141541 | 0.6128952 | 0.765296 | 0.8814948 |
| V2 | 10000 | 0.1887528 | 22.6232414 | 284.5577393 | 0.6374028 | 0.7569088 | 0.8751028 |
| V2 | 12000 | 0.1867594 | 22.4074421 | 288.9899292 | 0.6379756 | 0.7575149 | 0.8759526 |
| V2-H | 15000 | 0.1844665 | 22.193985 | 296.4523315 | 0.6378687 | 0.7596614 | 0.8822219 |
| V2-L1 | 20000 | 0.182056 | 22.0445251 | 298.578186 | 0.6410251 | 0.7558483 | 0.8718115 |
| V2-L2 | 30000 | 0.1787939 | 21.9687138 | 304.9963379 | 0.6450334 | 0.7557176 | 0.8678535 |
| V2-L3 | 42000 | 0.1753784 | 22.1255512 | 314.113678 | 0.6488576 | 0.7608134 | 0.8672472 |

说明：unique为0–1比例，空间锚定后的多样性明显不同于随机Query旧版，但这些指标不替代闭环控制结果。

## 22a. v2学习率分叉与下降速度的窗口对照

| ID | step区间[左闭右开] | 记录数 | Action中位数 | Delta中位数 | LoRA preclip | LoRA LR |
| --- | --- | --- | --- | --- | --- | --- |
| V2 | 2000-3000 | 1000 | 0.2101992 | 0.2903108 | 2.1001499 | 1e-05 |
| V2 | 10000-11000 | 1000 | 0.1606816 | 0.2024096 | 1.3559465 | 9.97e-07 |
| V2 | 11000-12000 | 1000 | 0.1562763 | 0.1957571 | 1.3804428 | 2e-06 |
| V2 | 12000-12743 | 743 | 0.1549427 | 0.1966237 | 1.4297648 | 2e-06 |
| V2-H | 14000-15000 | 1000 | 0.1552222 | 0.2283316 | 5.7753668 | 1.998e-05 |
| V2-L1 | 14000-15000 | 1000 | 0.151317 | 0.1914458 | 1.4880998 | 1.998e-06 |
| V2-L1 | 19000-20000 | 1000 | 0.1451365 | 0.18729 | 1.9191383 | 1.991e-06 |
| V2-L2 | 20000-21000 | 999 | 0.1446875 | 0.1869694 | 1.770534 | 1.989e-06 |
| V2-L2 | 29000-30000 | 1000 | 0.1373078 | 0.1861275 | 2.2223268 | 1.958e-06 |

说明：这些固定窗口可用于比较10K–20K和20K–30K的下降趋势，但中途分叉与学习率变化需按ID区分。

## 23. Latent WM与scratch的预测指标

| ID | Action MSE | Future训练loss | Future预测MSE | 零预测MSE | 复制当前MSE | grad norm | Action sigma均值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W20 | 0.0093807 | 0.3478998 | 0.3478998 | 0.9776349 | 0.8895287 | 0.2423735 | 0.7466087 |
| W57 | 0.0064988 | 0.3248194 | 0.3248194 | 0.9796157 | 0.8894356 | 0.240459 | 0.7479735 |
| S21 | 0.0263607 | 0.3265852 | 0.2450241 | 0.9531221 | 0.5630923 | 0.4720065 | 0.7479162 |

说明：W20/W57的Future训练loss是MSE，S21是L1而另记预测MSE，低训练loss并未保证额外训练后成功率单调提升。

## 24. Scratch数值精度检查

| sigma | FP32 Action MSE | BF16 Action MSE | FP32 Future L1 | BF16 Future L1 | 两者finite |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.2295668 | 0.2270329 | 0.334276 | 0.3342469 | True |
| 0.2 | 0.0571383 | 0.0567667 | 0.334276 | 0.3342469 | True |
| 0.5 | 0.0117361 | 0.0117516 | 0.334276 | 0.3342469 | True |
| 0.8 | 0.0328664 | 0.0333057 | 0.334276 | 0.3342469 | True |

说明：固定检查中动作差异RMS=0.001660、max=0.013249，未见足以据此解释成功率差距的精度灾难。

## 24a. Encoder 11K之后的整体latent对照

| encoder | latent RMS | Delta RMS | 相对Delta | 空间有效秩 | patch平均cosine | 时序cosine | 像素/latent变化Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| official | 1.8810161 | 0.7537047 | 0.4018027 | 39.4646111 | 0.4285566 | 0.9239417 | 0.7086195 |
| grid11000 | 2.0438993 | 0.8527931 | 0.4177668 | 40.0729198 | 0.4133833 | 0.9114399 | 0.6761489 |
| grid12000 | 2.0710071 | 0.8775497 | 0.4248264 | 40.1043205 | 0.4079032 | 0.9088845 | 0.6740481 |
| grid14000 | 2.1105366 | 0.9276677 | 0.4387729 | 40.2286911 | 0.4019465 | 0.8985477 | 0.6561496 |
| grid17000 | 2.1697549 | 0.9628208 | 0.4442106 | 40.8203526 | 0.3991289 | 0.8988633 | 0.6671184 |
| grid17358 | 2.1789994 | 0.9523933 | 0.4374806 | 41.0688782 | 0.4016892 | 0.9025386 | 0.6715432 |
| full10000 | 2.0223986 | 0.8325541 | 0.4126998 | 40.1522999 | 0.4155912 | 0.9121442 | 0.6847193 |
| full20000 | 2.2241654 | 0.9050182 | 0.4070476 | 39.5688286 | 0.4244798 | 0.9154185 | 0.6852659 |
| full40000 | 2.2350357 | 0.8908227 | 0.3993834 | 41.2032242 | 0.4292777 | 0.9213097 | 0.6750756 |
| full100000 | 2.2211149 | 0.8475667 | 0.3834616 | 32.6257687 | 0.4829499 | 0.9283223 | 0.6647212 |

说明：同16个episode的32对示范窗口，grid11K→17,358是同一短训线，full10K→100K是相同配方的另一独立run，不能称100K为grid11K直接续训。

## 25. Encoder任务区域跨时刻检索

| encoder | 操作物体Top-K命中 | 目标区域Top-K命中 | 夹爪Top-K命中 | 操作物体AUC | 操作物体raw AUC |
| --- | --- | --- | --- | --- | --- |
| official | 38.89% | 72.67% | 60.45% | 0.9694322 | 0.9691737 |
| grid11000 | 47.22% | 72.67% | 68.69% | 0.9635529 | 0.9607554 |
| full10000 | 38.89% | 69.89% | 64.90% | 0.9598264 | 0.9590809 |
| full40000 | 41.67% | 75.78% | 76.72% | 0.9790005 | 0.9674051 |
| full100000 | 75.00% | 76.22% | 73.94% | 0.9889133 | 0.9824364 |
| full103233 | 75.00% | 78.44% | 73.94% | 0.9890589 | 0.9826314 |

说明：只有三个手工近似ROI案例、以当前区域原型检索t+8未来区域，K等于目标框patch数，不能将局部对应改善直接等同于控制成功率改善。

## 26. Encoder区域内部一致性与可分性

| 区域 | encoder | 区域内cosine | 相对背景margin | 相对干扰物margin |
| --- | --- | --- | --- | --- |
| object | official | 0.6629341 | 0.3840529 | 0.1066189 |
| object | grid11000 | 0.6405692 | 0.3268691 | 0.0626364 |
| object | full100000 | 0.7409481 | 0.3782029 | 0.1094967 |
| object | full103233 | 0.7411752 | 0.376396 | 0.106363 |
| target | official | 0.6170203 | 0.3052334 | 0.1442577 |
| target | grid11000 | 0.6203844 | 0.2446411 | 0.0797089 |
| target | full100000 | 0.7173069 | 0.3022908 | 0.1022028 |
| target | full103233 | 0.7178665 | 0.3027305 | 0.1023836 |
| gripper | official | 0.603278 | 0.3125529 | 0.171264 |
| gripper | grid11000 | 0.6382991 | 0.3112984 | 0.1928181 |
| gripper | full100000 | 0.7302606 | 0.3736349 | 0.2634579 |
| gripper | full103233 | 0.7316339 | 0.3748595 | 0.2584381 |

说明：这组局部latent统计与G103下游重测51%/30%并存，说明“看得到关键物体”和“动作头能有效使用”是不同问题。

## 27. 当前Dream双头实验的最终实际配置

| 项目 | 配置 / 继承来源 |
| --- | --- |
| 起点 | Dream-VLA base；不加载旧E-LAF或mixed baseline训练权重 |
| 主干 | 实际hidden3584、28层；LoRA r32/alpha16 all-linear；BF16 |
| 原生条件 | 保留Dream图像、语言、proprio输入及原生图像增强 |
| JEPA当前输入 | 冻结grid11k；主视角16+wrist16；shared spatial-anchored Resampler；各自4帧 |
| 真实网格 / tubelet | 384分辨率，Conv3d tubelet2×patch16×16；2×24×24→1×4×4 |
| 空间/相机编码 | 新增固定2D sin/cos + learned camera/type；主干原RoPE保留 |
| Future目标 | 主视角固定池化16×1024绝对latent；不经过可学习目标Resampler；t+8；无Delta |
| 输出头 | Action保留官方residual MLP；Future沿用相同MLP模板、输出1024 |
| 交互 | Action/Future tokens在Dream各层交互，非独立两套专家Transformer |
| 20% Future dropout | 按样本屏蔽所有非Future query读取Future keys（含Future时间），保留Future监督 |
| 训练 / 推理 | Action/Future FM=1:1；独立训练时间；4-step联合Euler推理 |
| 数值 / batch | BF16主干；FP32 Resampler/target/loss；6×8=48；gradient checkpointing关闭 |
| 优化器 / 保存 | 官方AdamW LR1e-4、100K×0.1、无warmup；每5K保存完整latest/best |
| 当前step / 终段 | step 3300；action_loss=0.15719, future_loss=0.31019, seconds_per_update=2.04925 |
| 闭环成功率 | 尚未评测 |

说明：这是用户重新确认的官方MLP双头方向，已与旧72-token Delta-Flow E-LAF分开记录，当前仅有早期训练数据。

## 28. 已准备但没有正式质量结果的工作

| 项目 | 已执行 | 尚缺 / 不应宣称 |
| --- | --- | --- |
| V-JEPA Policy官方代码 | 保存revision d0e75c44463f85942f3580813fbcfe904cc0cf15；imports/CUDA/server CLI通过 | 所存状态记录官方policy checkpoint及匹配stats未就绪，未完成本机官方权重rollout |
| T5下载 | 曾使用代理、镜像及续传；S21按授权改用umT5 | 不能把S21标为T5-XXL实验 |
| LIBERO-Pro / RoboTwin / RoboDojo | 接口/依赖与部分headless smoke准备 | 没有本报告模型的完整有效成功率可列 |
| infra缓存改动 | 已开发部分缓存/读取改动并做独立测试 | 按最新要求暂不继续；不计入模型实验结果 |

说明：下载、环境和smoke不等于模型质量实验，公开论文中的97.25%/98.70%等也不作为我们自己的结果入表。

## 29. 原始证据索引

| 编号 | 相对服务器experiments目录的路径 / 本地证据键 |
| --- | --- |
| S01 | elaf_v0_grid11k_72token/resolved_config.json |
| S02 | elaf_v1_grid11k_72token/resolved_config.json |
| S03 | elaf_recovery_from5000_20261019/resolved_config.json |
| S04 | elaf_fresh_flowclip_2k_20261019/resolved_config.json |
| S05 | elaf_lora_zero_control_20261020/resolved_config.json |
| S06 | elaf_v2_spatial_staged_20261020_run/resolved_config.json |
| S07 | elaf_v2_lora10x_20261020/resolved_config.json |
| S08 | elaf_v2_low_lr_from15k_20261020/resolved_config.json |
| S09 | elaf_v2_low_lr_from12742_20261020/resolved_config.json |
| S10 | elaf_v2_low_lr_20k_to30k_20261020/resolved_config.json |
| S11 | elaf_v2_low_lr_to200k_20261021/resolved_config.json |
| S12 | dream_baseline200k_eval_corrected_20261019_143839/clean/summary.json |
| S13 | dream_baseline200k_eval_corrected_20261019_143839_plus_retry2/plus/summary.json |
| S14 | action_only/grid11k_action20k_instruction_retest_20261019_154622/clean/summary.json |
| S15 | action_only/grid11k_action20k_instruction_retest_20261019_154622/plus/summary.json |
| S16 | action_only/grid11k_action20k_denoise10_20261020/clean/summary.json |
| S17 | action_only/grid11k_action20k_denoise10_20261020/plus/summary.json |
| S18 | action_only/full103233_action20k_instruction_retest_20261021/clean/summary.json |
| S19 | action_only/full103233_action20k_instruction_retest_20261021/plus/summary.json |
| S20 | elaf_v2_step14500_eval_gpu67_20261020/clean/summary.json |
| S21 | elaf_v2_step14500_eval_gpu67_20261020/plus/summary.json |
| S22 | elaf_v2_low_lr_step30000_eval_gpu67_20261021/clean/summary.json |
| S23 | elaf_v2_low_lr_step30000_eval_gpu67_20261021/plus/summary.json |
| S24 | grid11k_latent_wm_20261021/evaluation_020000/clean/summary.json |
| S25 | grid11k_latent_wm_20261021/evaluation_020000/plus/summary.json |
| S26 | grid11k_latent_wm_extend57k_20261021/evaluation_057000/clean/summary.json |
| S27 | grid11k_latent_wm_extend57k_20261021/evaluation_057000/plus/summary.json |
| S28 | grid11k_latent_wm_extend57k_20261021/evaluation_057000_denoise10/clean/summary.json |
| S29 | grid11k_latent_wm_extend57k_20261021/evaluation_057000_denoise10/plus/summary.json |
| S30 | grid11k_small_scratch_20261022/evaluation_021360/clean/summary.json |
| S31 | grid11k_small_scratch_20261022/evaluation_021360/plus/summary.json |
| S32 | dream_baseline200k_eval_20261019_132600/clean/summary.json |
| S33 | dream_baseline200k_eval_flow4_20261019_134959/invalid_protocol.json |
| S34 | dream_eval_audit_20261019/findings.json |
| S35 | elaf_v0_grid11k_72token/metrics.jsonl |
| S36 | elaf_v1_grid11k_72token/metrics.jsonl |
| S37 | elaf_recovery_from5000_20261019/metrics.jsonl |
| S38 | elaf_fresh_flowclip_2k_20261019/metrics.jsonl |
| S39 | elaf_lora_zero_control_20261020/metrics.jsonl |
| S40 | elaf_v2_spatial_staged_20261020_run/metrics.jsonl |
| S41 | elaf_v2_lora10x_20261020/metrics.jsonl |
| S42 | elaf_v2_low_lr_from15k_20261020/metrics.jsonl |
| S43 | elaf_v2_low_lr_from12742_20261020/metrics.jsonl |
| S44 | elaf_v2_low_lr_20k_to30k_20261020/metrics.jsonl |
| S45 | elaf_v2_low_lr_to200k_20261021/metrics.jsonl |
| S46 | baseline_checkpoint（experiment_history_20261023_final.json） |
| S47 | baseline_wandb_summary（experiment_history_20261023_final.json） |
| S48 | grid11k_latent_wm_20261021/train/train_metrics.jsonl |
| S49 | grid11k_latent_wm_extend57k_20261021/train/train_metrics.jsonl |
| S50 | grid11k_small_scratch_20261022/train/train_metrics.jsonl |
| S51 | elaf_fresh_flowclip_2k_20261019/slot_ablation/results.json |
| S52 | elaf_fresh_flowclip_2k_20261019/branch_usage_native/results.json |
| S53 | elaf_layer_diagnostic_20261020/results.json |
| S54 | elaf_lora_zero_control_20261020/completion_verification.json |
| S55 | elaf_v2_spatial_staged_20261020_run/ELAF_V2_INIT_DIAGNOSTICS.json |
| S56 | elaf_v2_spatial_staged_20261020_run/transitions.jsonl |
| S57 | grid11k_small_scratch_20261022/precision_audit.json |
| S58 | libero_encoder_final103233_regions_20261021/task_regions/summary.json |
| S59 | libero_encoder_final103233_regions_20261021/region_separation/summary.json |

说明：服务器根目录为 /data/users/jiachangzhang/eb_vjepa/experiments，本地同目录的 experiment_history_20261023_{evidence,extra,final,protocol}.json 保存了本次整理所用快照。
