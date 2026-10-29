# E-LAF 健康/失稳逐层对照

原训练已安全暂停在34021步。preserved_step2000、preserved_step3000、preserved_latest_paused独立目录保存完整checkpoint（training.pt和六个rank状态），避免latest/best后续清理影响。

实验A：4个固定训练样本，每套一个；u=s取.25/.5/.75；同样噪声、无图像增强、eval模式、原生BF16主干和FP32 Resampler。各checkpoint共12次，分别对Action/Delta Loss求未裁剪LoRA梯度，不执行优化更新。以下是12次的中位数，不是训练全局batch统计。

| 指标 | Step3000 | Step34021 |
|---|---:|---:|
| Action loss | .1751 | 2.1502 |
| Delta loss | .03939 | 1.9142 |
| Action→LoRA梯度范数 | 5.295 | 239.976 |
| Delta→LoRA梯度范数 | 3.136 | 67871895 |
| 两任务LoRA梯度cosine | -.00233 | -.03435 |
| block27 q_proj LoRA/base RMS | .890 | 45.383 |

失稳模型block27 k_proj LoRA/base RMS为26.204，v_proj为2.058；block26 mlp.down_proj为1.492。Delta梯度最大的block为26（约6382万），随后25、27、24。block编号从0起。RMS比值是每个线性模块在其自身输入上的缩放后LoRA输出与base输出之比，不是整层Jacobian；大的比值也可能受base输出尺度影响，原始文件保留两者绝对RMS。

这些证据定位了失稳状态：Delta监督的反向敏感性远大于Action，末端注意力投影被LoRA明显改变。但没有证明最初触发点，也不排除后期结果与早期机制不同。任务梯度cosine中位数接近0，未显示强烈的全局反向冲突；逐层/逐样本结果需另看，不能据全局均值排除局部冲突。

本地官方Dream-VLX vla/vla-scripts/finetune_dream.py第915–917行同样设置alpha=min(rank,16)、target_modules=all-linear。当前配置不是在这两项上偏离本地官方脚本；这不保证该设置适合E-LAF新增双流。

实验B已启动：从完整Step3000状态恢复，到绝对Step6000结束（3000次新更新）；只将LoRA优化器lr、initial_lr及scheduler.base_lrs置0，保留requires_grad和AdamW一二阶矩。其他模块LR、数据游标、每卡RNG和调度阶段保持。新W&B run v0h0u6fl，项目eb_vjepa。监控继承原run的Step0尺度校准，并记录来源。新run重置best选择窗口，保留绝对step，未重启warmup。

入口scripts/run_elaf_lora_zero_control.py是诊断专用包装，没有修改原训练核心代码。control_receipt.json保存起始LoRA权重摘要，完成后应核对权重不变。新run输出experiments/elaf_lora_zero_control_20261020。该对照尚无完整结果，不能提前宣布稳定。

实验A原始结果：artifacts/elaf_layer_diagnostic_20261020/results.json，含所有28层的分支activation RMS/max、每个LoRA模块base/adapter RMS、每层分任务梯度与cosine。脚本scripts/diagnose_elaf_layers.py。
