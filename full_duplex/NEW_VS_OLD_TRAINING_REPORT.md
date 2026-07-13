# Previous-GT 新方案与 Autoregressive 旧方案总结

## 1. 新方案视频

新方案使用 step-100 最佳 checkpoint 做 fresh forward，得到总体预测 latent
`[1,19,16,60,104]`，再通过真实冻结 Wan2.1 VAE decoder 导出：

- 视频：`full_duplex/outputs/smallest_000000/teacher_forced_b30_s8_ckpt10/video_export/prediction_step_000100_teacher_forced.mp4`
- 预览：`full_duplex/outputs/smallest_000000/teacher_forced_b30_s8_ckpt10/video_export/prediction_step_000100_teacher_forced_preview.png`
- provenance：`full_duplex/outputs/smallest_000000/teacher_forced_b30_s8_ckpt10/video_export/prediction_step_000100_teacher_forced.json`

| 项目 | 结果 |
|---|---:|
| predicted latent | `[1,19,16,60,104]` |
| decoded RGB | `[1,3,73,480,832]` |
| 编码 | H.264, CRF 18 |
| FPS / 时长 | 24 / 3.0417 秒 |
| latent / RGB finite | 是 |
| VAE decode 峰值显存 | 15,407,987,200 bytes |
| VAE checkpoint SHA256 | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |

provenance 明确记录：

```text
ground_truth_world_inputs_used_for_prediction = true
ground_truth_camera_inputs_used_for_prediction = false
ground_truth_current_output_visible_to_model = false
ground_truth_decoded = false
```

因此这段视频是“previous world GT 条件下的 next-state reconstruction”，
不是不依赖 GT 的 autonomous rollout。人物/场景粗轮廓、色彩和相机运动仍
可见，但有明显密集彩色噪点，主体边缘不清晰；相较旧方案没有可确认的清晰度
跃升。

## 2. 训练机制差异

| 项目 | 旧方案 | 新方案 |
|---|---|---|
| world input, turn 0 | zero/null | zero/null |
| world input, turn `t>0` | 上一轮预测 `pred[t]` | 真实缓存 `GT state[t]` |
| state target | `GT state[t+1]` | `GT state[t+1]` |
| state loss | `MSE(pred[t+1], GT[t+1])` | 同左 |
| 历史预测值 | 可见且保留 graph | 可见但 detached |
| 跨 turn 反向传播 | 完整 19-turn BPTT | 无 |
| 每轮内部 10-step Flow graph | 保留 | 保留 |
| optimizer update | rollout 后整体 backward/update | 每轮 `loss/19` backward，19 轮后 update |
| autonomous generation | 后续不需要 world GT | 每轮需要 previous world GT |
| 当前 target GT 是否泄漏 | 否 | 否 |
| camera input | 上一轮预测 camera | 同左；未 teacher-force camera |

新方案逐轮 backward 与 19 个独立 transition loss 的 mean parameter gradient
数学等价，但刻意删除了未来 turn loss 通过 predicted world state 回传到早期
turn 的梯度。

## 3. 实际实验条件差异

这不是只改变一个变量的严格对照，不能把所有差异都归因于 GT 输入。

| 条件 | 旧方案最终 run | 新方案最终 run |
|---|---:|---:|
| optimizer steps | 200 | 100 |
| executed Transformer blocks | 1 | 30 |
| spatial token stride | 8 | 8 |
| denoising steps / turn | 10 | 10 |
| turns | 19 | 19 |
| trainable parameters | 8,190,055 | 8,190,055 |
| Wan backbone | 冻结 | 冻结 |
| activation checkpoint | 全部执行层 | 前 10/30 层 |
| 最大记录显存 | 11.93 GiB | 72.90 GiB |
| 平均 optimizer step | 13.43 秒 | 65.58 秒 |

新方案从旧方案 step-200 task delta warm-start，再把执行深度从 1 改到 30。
深层特征分布变化令新方案 step-1 state MSE 暂时升到 3.6073。因此新方案的
下降曲线同时受 warm-start、深度和 teacher forcing 影响，不是单变量因果结论。

新方案能执行完整 30 层的关键，是不再保存跨 19 turns 的大图。完全关闭
activation checkpoint 会在约 79.15 GiB OOM；最终 checkpoint 前 10/30 层，
峰值约 72.90 GiB。

## 4. Loss 对比

### 相同 optimizer step 100

| 指标 | 旧方案 step 100 | 新方案 step 100 | 新方案变化 |
|---|---:|---:|---:|
| total loss | 2.640217 | 2.540905 | -3.76% |
| flow loss | 1.321738 | 1.274728 | -3.56% |
| state MSE | 1.318127 | 1.266101 | -3.95% |
| camera loss | 0.000352 | 0.000075 | -78.54% |

该表只说明各自第 100 步实际值；由于初始化、block 深度和输入难度不同，不能
当作受控消融。

### 各方案最终最佳 checkpoint 的 fresh evaluation

| 指标 | 旧方案 step 200 | 新方案 step 100 | 差异 |
|---|---:|---:|---:|
| overall state MSE | 1.266472 | 1.266015 | 新方案低 0.036% |
| latent cosine | 0.413235 | 0.416657 | 新方案高 0.83% |
| predicted latent mean | 0.102214 | 0.098531 | target 0.098777 |
| predicted latent std | 1.170548 | 1.174650 | target 0.828415 |
| camera translation L2 | 0.012891 | 0.008279 | 新方案低 35.78% |
| camera rotation | 0.489831° | 0.689707° | 新方案高 40.81% |

训练 total loss 最终值：旧方案 `2.534786`，新方案 `2.540905`；旧方案低
0.24%。两者 state MSE 几乎相同，且都高于 zero-latent baseline
`0.696027`，所以都未完成高质量 world latent 过拟合。

## 5. 视频对比

旧方案同规格输出：

`full_duplex/outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/video_export/prediction_step_000200.mp4`

新旧逐帧比较：

| 指标 | 数值 |
|---|---:|
| PSNR | 36.9896 dB |
| SSIM | 0.982617 |
| 两组 predicted latent 间 MSE | 0.004750 |

高 SSIM 与 contact sheet 肉眼观察一致：两段输出非常相似。新方案 camera
translation 更好，但没有把 world 视频从噪点状态提升到清晰复现。

## 6. 为什么没有明显变清晰

1. stride 8 在 30×52 patch grid 上每个 world modality 只取 28 个 token，
   再插值回 60×104 latent，空间容量很低。
2. 1.49B Wan backbone 全部冻结，只训练 8.19M task embeddings/heads/prior。
3. 删除 exposure error 和跨 turn 图不能恢复 stride 8 已丢弃的高频信息。
4. 两方案 predicted latent std 都约 1.17，target std 仅 0.828，噪声/幅度
   偏大仍未解决。
5. 新方案 step 100 state MSE 已在约 1.266 附近明显变平。

## 7. 结论与下一步

- 新方案正确实现 `previous GT -> next GT`，并使完整 30 层可训练。
- camera reconstruction 已很好拟合；world latent 没有实质性质量跃升。
- 新方案最终 MSE/cosine 只略好于旧方案，视频仍高度相似且噪点明显。
- teacher-forced 视频不能代表 autonomous world-model rollout。

下一步最有价值的严格消融是：保持 GT→GT 协议探测 stride 4；对
`blocks=4/8/16/30` 使用相同初始化和相同步数；在显存允许下增加可训练容量；
同时分别报告 teacher-forced reconstruction 与 autonomous rollout。
