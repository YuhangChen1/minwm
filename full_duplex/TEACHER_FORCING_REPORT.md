# Previous-GT → next-GT 微调实验报告

## 结论

独立 teacher-forcing 训练路径已实现并在真实最小数据集上完成 100 个
optimizer steps。该路径未使用 LoRA，也未保留跨 19 轮的 autograd graph。

严格世界状态协议为：

```text
turn 0:  zero/null world -> GT state[1]
turn t:  GT state[t]     -> GT state[t+1], t > 0
L_state = MSE(predicted final latent, GT state[t+1])
```

训练完成后，独立加载 step-100 最佳 checkpoint 得到：

| 指标 | step 10 | step 100 | 变化 |
|---|---:|---:|---:|
| overall state MSE | 1.512288 | 1.266015 | -16.29% |
| overall latent cosine | 0.354715 | 0.416657 | +17.46% |
| camera translation L2 | 0.264005 | 0.008279 | -96.86% |
| camera rotation | 2.984869° | 0.689707° | -76.89% |

训练本身稳定、100 次相邻 optimizer step 的 total loss 全部下降，但 world
latent 尚未达到“完整过拟合”：step-100 MSE 仍高于 zero-latent baseline
`0.696027`。因此本报告不把它描述为已经成功复现全部 latent 细节。

## 实现文件

- `teacher_forcing_training.py`
  - `previous_ground_truth_world_input`: turn 0 返回完全同 shape 的零状态；
    turn `t>0` 返回缓存中的精确 `world_states[t]` view。
  - `TeacherForcedTransitionTrainer.forward_loss`: 独立预测/评估路径。
  - `TeacherForcedTransitionTrainer.train_step`: 每轮执行
    `(turn_loss / num_turns).backward()`，累积参数梯度后只做一次 optimizer
    update；历史预测先 `detach()`，随后才进入下一轮序列。
  - `load_task_warm_start`: 只接受经过 key 集合、shape、finite、缓存哈希、
    基础 checkpoint identity 和固定噪声哈希验证的非 LoRA task delta。
- `train_teacher_forcing.py`
  - 独立 CLI、warm-start/resume、训练 regime 检查、层数/stride/去噪步数与
    partial activation checkpointing 控制。
- `model.py`
  - `gradient_checkpointing_blocks` 支持只重计算前 N 个已执行 block；该
    选项只改变 backward 的内存/计算权衡，不改变 forward 函数。
- `predict_checkpoint.py`
  - 根据 checkpoint 中的 `training_regime` 自动选择 teacher-forced 或原
    autoregressive predictor；输出输入和目标 state index manifest。
- `tests/test_teacher_forcing.py`
  - 断言 turn 0 为零、turn 1..18 与 GT state 1..18 bitwise 相同；
  - 断言历史 tensor 没有 `requires_grad/grad_fn`；
  - 断言逐轮 backward 与整体 mean loss 的参数梯度数学等价。
- `summarize_metrics.py`
  - 保留 `input_state_index`、`target_state_index` 和 teacher-forcing regime
    字段，生成 aggregate/per-turn CSV、曲线和 JSON 摘要。

## 前向语义和反向边界

每个 optimizer step 仍遍历全部 19 个真实 transition。每轮内部保留 10
步可微 Flow/Euler 去噪图；该轮 backward 结束后立即清除图和模型 encoding
cache。参数梯度跨轮累积，直到 19 轮完成后统一 clip 和 optimizer step。

因此：

- 当前 world input：精确 previous GT；
- 当前 state target：精确 next GT；
- 当前输出槽：masked，不读取 GT target；
- 历史输出值：可见但 detached；
- future turn：不可见；
- cross-turn BPTT：关闭；
- within-turn 10-step differentiation：保留；
- camera：仍使用上一轮预测，未启用 camera teacher forcing；
- VAE decoder/RGB：未进入训练或 latent 指标计算；训练完成后另行通过冻结
  decoder 做可视化导出，不反向传播。

实际 per-turn CSV 的 state index 从：

```text
step 1, turn 0: input=-1 (NULL), target=1
step 1, turn 1: input=1,         target=2
...
step 100, turn 18: input=18,     target=19
```

## Transformer 层数与显存探测

所有探测均使用真实缓存、19 turns、10 denoising steps、stride 8、bf16，
基础 Wan 权重严格加载且冻结。初始 task delta 来自此前的 1-block 最佳点，
因此更深层初始 loss 包含 task head 的特征分布适配成本。

| executed blocks | step-1 state MSE | total loss | 峰值显存 GiB | 秒/步 |
|---:|---:|---:|---:|---:|
| 4 | 1.392140 | 2.835983 | 6.988 | 24.14 |
| 8 | 1.499942 | 3.053548 | 7.413 | 35.81 |
| 16 | 1.924277 | 4.474695 | 8.267 | 57.38 |
| 30 | 3.607297 | 7.209465 | 9.761 | 89.23 |

上述结果证明“不保留多轮图”后完整 30/30 层能够运行。30 层在 10 步内把
state MSE 从 `3.607297` 降到 `1.554140`，所以继续选择完整深度训练。

进一步的 backward 内存消融：

- 30/30 blocks 全部不 checkpoint：真实 OOM，进程使用约 79.15 GiB；
- checkpoint 全部 30 blocks：约 9.82 GiB；
- checkpoint 前 10/30 blocks：step 14 峰值 72.895 GiB、72.69 秒，稳定；
- 最终选择 10/30，以约 6 GiB 分配余量换取更高吞吐。

## 100 步训练结果

配置：

```text
training_regime=teacher_forced_previous_gt_transition
teacher_forcing_ratio=1.0
teacher_forced_world_inputs=true
teacher_force_camera=false
sequential_turn_backward=true
detach_between_turns=true
num_backbone_blocks=30
spatial_token_stride=8
num_micro_turns=19
num_denoising_steps=10
gradient_checkpointing_blocks=10
lora_enabled=false
```

模型总参数 `1,498,011,815`；冻结基础参数 `1,489,821,760`；训练 task
modules `8,190,055`。训练 loss 里程碑：

| step | total | flow | state MSE | camera |
|---:|---:|---:|---:|---:|
| 1 | 7.209465 | 3.470663 | 3.607297 | 0.131505 |
| 10 | 3.128948 | 1.560910 | 1.554140 | 0.013898 |
| 20 | 2.788138 | 1.398338 | 1.385512 | 0.004288 |
| 40 | 2.614527 | 1.312188 | 1.299950 | 0.002390 |
| 60 | 2.562689 | 1.286790 | 1.275678 | 0.000222 |
| 80 | 2.547594 | 1.278557 | 1.268835 | 0.000202 |
| 100 | 2.540905 | 1.274728 | 1.266101 | 0.000075 |

总 loss 从 step 1 到 100 下降 `64.76%`，最低点为 step 100；所有 99 个
相邻转移均下降。平均单步耗时 65.58 秒（包含多次恢复后的首次编译步），
记录到的最大峰值显存 72.90 GiB。

step-100 独立预测（checkpoint 更新后 fresh forward）：

- latent shape: `[19,16,60,104]`；
- overall state MSE: `1.266015`；
- overall latent cosine: `0.416657`；
- predicted mean/std: `0.098531 / 1.174650`；
- target mean/std: `0.098777 / 0.828415`；
- best turn: 0，MSE `1.207818`；
- worst turn: 18，MSE `1.292603`；
- camera translation L2: `0.008279`；
- camera rotation: `0.689707°`；
- camera intrinsics RMSE: `0.004248`。

## Checkpoint 与产物

- 最佳/最终 checkpoint:
  `full_duplex/outputs/smallest_000000/teacher_forced_b30_s8_ckpt10/checkpoints/best.pt`
- 命名 checkpoint:
  `step_000100_total_2.540905_state_1.266101_camera_0.000075.pt`
- reload test: flow/camera 最大绝对误差均为 `0.0`；
- 独立预测: `prediction_step100.pt`；
- 最终评估: `evaluation_step100.json` 和 `.csv`；
- 100 步日志: `metrics.jsonl`；
- aggregate/per-turn 表: `loss_history.csv`、`per_turn_loss_history.csv`；
- 曲线: `loss_curve.png`；
- 配置和加载证据: `run_manifest.json`（其中包含完整 warm-start report）。

训练结束后，`prediction_step100.pt` 已通过真实冻结 Wan2.1 VAE decoder
导出为 73 帧、480×832、24 FPS 的 H.264 视频：

- `video_export/prediction_step_000100_teacher_forced.mp4`；
- `video_export/prediction_step_000100_teacher_forced_preview.png`；
- `video_export/prediction_step_000100_teacher_forced.json`。

该 manifest 明确记录 previous world GT 被用于预测、当前目标 GT 对模型不可见、
GT 没有被 decoder 代替预测结果。视频仍有明显彩色噪点，完整旧/新对比见
`NEW_VS_OLD_TRAINING_REPORT.md`。

## 可复现命令

```bash
export PYTHONPATH=/output/minwm:/output/minwm/Wan21:/output/minwm/shared
PY=/hyperai/home/conda_envs/minwm/bin/python

$PY -u -m full_duplex.train_teacher_forcing \
  --warm-start full_duplex/outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/checkpoints/best.pt \
  --run-name teacher_forced_b30_s8_ckpt10 --max-steps 100 \
  --num-turns 19 --num-denoising-steps 10 --blocks 30 \
  --spatial-token-stride 8 --attention-pad-to-turns 19 \
  --checkpoint-blocks 10

$PY -u -m full_duplex.predict_checkpoint \
  --checkpoint full_duplex/outputs/smallest_000000/teacher_forced_b30_s8_ckpt10/checkpoints/best.pt \
  --output full_duplex/outputs/smallest_000000/teacher_forced_b30_s8_ckpt10/prediction_step100.pt

$PY -u -m full_duplex.evaluate_predictions \
  --predictions full_duplex/outputs/smallest_000000/teacher_forced_b30_s8_ckpt10/prediction_step100.pt \
  --checkpoint full_duplex/outputs/smallest_000000/teacher_forced_b30_s8_ckpt10/checkpoints/best.pt \
  --output full_duplex/outputs/smallest_000000/teacher_forced_b30_s8_ckpt10/evaluation_step100.json
```

## 尚未解决的问题

world latent 曲线在约 1.27 附近明显变平，并且预测标准差偏高。最可能的
已确认容量限制是 stride 8 每个 world modality 只保留 28 个空间 token，
最后再插值回 `[60,104]`；其验证 low-pass grid 只有 `[8,14]`。此外本次
只训练 8.19M task modules，1.49B Wan 基础权重全部冻结。

因此，若下一阶段目标是明显降低噪点而不只是继续压 camera loss，优先级应
是 teacher-forced 路径上的 stride 4 / 更多可训练容量消融，而不是简单把
同一配置从 100 盲目延长到 500。该选择尚未执行，需另行做显存和训练验证。
