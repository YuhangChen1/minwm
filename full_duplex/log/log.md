# minWM Full-Duplex 微调任务总报告

> 报告日期：2026-07-13  
> 用户指定路径：`/hyperai/home/minwm/full_duplex/log/log.md`  
> 当前容器等价路径：`/output/minwm/full_duplex/log/log.md`  
> 代码仓库：`/hyperai/home/minwm`  
> Python 环境：`/hyperai/home/conda_envs/minwm`

---

## 0. 总结结论

本任务围绕 Wan2.1 / Action2V 的 `ar_diffusion_tf` checkpoint，实现并真实
运行了 latent-space Full-Duplex micro-turn 微调框架。工作不是只写设计文档，
而是完成了代码审计、真实数据预编码、严格 checkpoint 加载、序列与 mask、
Flow Matching、多轮训练、checkpoint 恢复、100/200 步最小样本过拟合、独立
latent 评估以及冻结 VAE decoder 视频导出。

我们最终重点比较了两种训练方式：

1. **旧方案：Autoregressive rollout + 跨轮 BPTT**  
   turn `t>0` 的 world input 是上一轮模型预测；19 轮图连在一起，未来 turn
   loss 可以回传到早期预测。最终长期训练配置为了单卡效率只执行 1 个 Wan
   Transformer block。
2. **新方案：Previous-GT → Next-GT teacher forcing + 逐轮 backward**  
   turn `t>0` 的 world input 是真实缓存 `GT state[t]`；每轮完成 backward 后
   释放该轮图，不保留 19 轮 autograd graph。它使单卡 H100 能执行完整 30/30
   Transformer blocks。

两个方案都成功下降并拟合 camera，但最终 world latent 质量几乎相同：

| Fresh checkpoint 指标 | 旧方案 step 200 | 新方案 step 100 |
|---|---:|---:|
| latent MSE | 1.266472 | 1.266015 |
| latent cosine | 0.413235 | 0.416657 |
| camera translation L2 | 0.012891 | 0.008279 |
| camera rotation error | 0.489831° | 0.689707° |

新方案 MSE 只低 `0.036%`，cosine 只高 `0.83%`。两段 VAE 解码视频之间
SSIM 为 `0.982617`，视觉上高度相似，仍有明显彩色噪点。因此：

- 新方案在**训练图内存与完整 30 层可运行性**方面成功；
- 新方案在**camera translation**方面更好；
- 新方案尚未带来可确认的 world latent 或 RGB 清晰度跃升；
- 两种方案的 MSE 都高于 zero-latent baseline `0.696027`；
- `predicted latent ≈ ground truth` 的严格过拟合目标仍未建立。

---

## 1. 原始任务与实际完成范围

原始任务要求在不重新预训练 Wan2.1 的前提下：

- 严格加载 `Wan21/Action2V/ar_diffusion_tf/model.pt`；
- 使用 Full-Duplex、time-aligned micro-turn token 序列；
- 显式建模 world、camera、action、noise、文本条件；
- 保留 Wan checkpoint 的 patch embedding、RoPE、PRoPE、Transformer blocks、
  cross-attention、FFN、timestep embedding 和兼容输出 head；
- 训练 world Flow Matching、final latent MSE 和 camera loss；
- 使用固定初始 noise 做 10-step differentiable denoising；
- 在真实最小数据集上做过拟合；
- 保存 best/latest checkpoint，并做 fresh-model reload test；
- 初始阶段不把 VAE decoder 放进训练闭环。

实际完成内容：

- 完成仓库、checkpoint、数据、VAE/T5、camera/action、scheduler 和原训练/推理
  路径审计；
- 完成真实视频 VAE 编码与真实 prompt UMT5 编码缓存；
- 完成 41 个独立特殊 token、序列 span、prediction mask、causal mask 和测试；
- 完成 explicit camera/action token、world/noise patch token、文本 cross-attention；
- 完成 strict base load + 显式 task delta checkpoint；
- 完成固定 noise 10-step Flow/Euler 去噪、world/camera loss；
- 完成旧 autoregressive rollout 和新 previous-GT teacher-forced 两条训练路径；
- 完成 LoRA 旁路实验，但因质量显著退化而停止使用；
- 完成两条主路径的训练、独立 latent 评估与 VAE 视频导出；
- 完成日志、CSV、loss 曲线、per-turn 指标和对比报告。

VAE decoder 始终未进入训练图。视频是在训练完成后，显式使用冻结 VAE 做的
独立可视化。

---

## 2. 环境与基础 checkpoint 审计

### 2.1 环境

| 项目 | 实际值 |
|---|---|
| Python | `/hyperai/home/conda_envs/minwm/bin/python` |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA H100 80GB HBM3 |
| bf16 | 支持并用于训练 autocast |

### 2.2 基础 checkpoint

绝对路径：

```text
/hyperai/home/minwm/ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```

审计事实：

| 项目 | 结果 |
|---|---:|
| 文件大小 | 5,959,605,031 bytes |
| 顶层 key | `generator` |
| tensor 数 | 885 |
| 参数元素数 | 1,489,821,760 |
| dtype | 全部 float32 |
| EMA | 无 |
| optimizer/scheduler | 无 |
| Transformer depth | 30 |
| hidden dim | 1536 |
| FFN dim | 8960 |
| attention heads | 12 |
| latent channels | 16 |
| patch size | `(1,2,2)` |
| text width / max length | 4096 / 512 |

基础模型使用 strict load，missing/unexpected 均为空，加载比例 `1.0`。真实缓存
输入前向输出 `[1,1,16,60,104]` flow/x0，全部 finite。基础 checkpoint
包含每一层的 `self_attn.prope_o.{weight,bias}`。

审计入口与证据：

- `full_duplex/audit_environment.py`
- `full_duplex/audit_checkpoint.py`
- `full_duplex/AUDIT.md`
- `full_duplex/outputs/smallest_000000/checkpoint_audit.log`

---

## 3. 真实数据、动作与缓存

### 3.1 实际数据与任务书差异

任务书中出现过“11 actions/turns”和“19 actions”两种描述。真实文件验证结果：

| 项目 | 实际值 |
|---|---:|
| RGB frames | 77 |
| FPS | 24 |
| 分辨率 | 832×480 |
| VAE latent states | 20 |
| latent shape | `[20,16,60,104]` |
| transitions/actions | 19 |
| `right` | 8 |
| `a` | 11 |

第 0 帧是初始 context；frames 1..76 被分为 19 段，每段 4 帧。VAE temporal
chunk 是 `1,4,4,...`，因此 77 RGB frames 对齐为 20 个 latent states。

动作与 state 对齐：

```text
turn 0:  action right, frames 1..4,   target latent 1
...
turn 7:  action right, frames 29..32, target latent 8
turn 8:  action a,     frames 33..36, target latent 9
...
turn 18: action a,     frames 73..76, target latent 19
```

每个 action 的真实跨度为 `4/24 = 166.67 ms`，不是图中名义的 200 ms。

### 3.2 Camera

样本没有物理测量 camera 文件。camera target 是仓库 camera trajectory 代码
根据真实动作序列确定性生成的 OpenCV `w2c`：

```text
[translation(3), rotation6d(6), intrinsics(4)] = 13 values
```

`right` 驱动旋转，`a` 驱动平移。因此 camera loss 是对仓库生成轨迹的拟合，
不是传感器 ground truth。

### 3.3 缓存

缓存目录：

```text
/hyperai/home/minwm/full_duplex/cache/smallest_000000
```

缓存内容包括：

- world latents `[20,16,60,104]` float16；
- prompt embedding `[512,4096]` bfloat16；
- prompt attention mask，132 个非 padding token；
- camera `[20,13]`；
- action IDs/names；
- 19 个 action/state/frame 对齐；
- 视频、VAE、T5、base checkpoint identity；
- preprocessing config hash；
- latent mean/std/min/max 和 finite 检查。

缓存重复加载通过 bitwise equality。VAE/T5 identity 或预处理配置变化时，缓存
不会仅凭文件存在而复用。

关键文件：

- `full_duplex/preencode.py`
- `full_duplex/audit_data.py`
- `full_duplex/cache/smallest_000000/metadata.json`
- `full_duplex/outputs/smallest_000000/data_audit.json`

---

## 4. Full-Duplex 模型与训练公共组件

### 4.1 特殊 token 与序列

实现了 41 个独立、可训练 special embeddings，shape `[41,1536]`。包含：

- input/output stream start/end；
- modality end；
- noise end；
- null world state；
- masked world/camera output；
- `TIME_INDEX_0..31`。

每 turn 按任务书顺序构造：

```text
INPUT_STREAM_START
world_input + MODALITY_END
camera_input + MODALITY_END
action_input + MODALITY_END
noise_input + NOISE_END
INPUT_STREAM_END
OUTPUT_STREAM_START
MASKED_WORLD_OUTPUT + MODALITY_END
MASKED_CAMERA_OUTPUT + MODALITY_END
OUTPUT_STREAM_END
TIME_INDEX_t
```

当前 GT output 内容不进入模型 token；world/camera 输出区域使用 masked
representation，边界 token 可见。

### 4.2 Attention / prediction mask

mask 规则：

- 所有历史 turn 可见；
- 当前 turn 全部 input 和边界 token 可见；
- 当前 turn GT output content 不可见；
- future turn 全部不可见；
- special boundary/time/null token 不是 prediction target；
- padding 不可作为有效 key。

小尺寸二维 mask、span 和断言由 `full_duplex/tests/test_tokens.py` 验证。

### 4.3 各模态编码

- world/noise：复用 checkpoint `patch_embedding`；
- stride 1 时 `[1,1,16,60,104] -> [1,1560,1536]`；
- stride 8 时每个 world/noise modality 只保留 28 tokens；
- camera：`[B,13] -> [B,1,1536]` explicit token；
- action：ID embedding `[B] -> [B,1,1536]`；
- 文本：冻结 UMT5 `[B,512,4096]`，投影后作为每个 block 的 cross-attention
  K/V，sequence hidden 是 Q；
- RoPE：规则 `(time,height,width)`；
- PRoPE：每个 video token 的 `w2c + K` projective geometry；
- turn embedding / TIME token：micro-turn 位置；
- 三者相互独立，没有混用。

### 4.4 Flow 与 loss

沿用 checkpoint scheduler 符号：

```text
x_sigma = (1-sigma) * x0 + sigma * epsilon
flow target = epsilon - x0
```

每 turn 使用固定 seed 生成的同一 initial noise，10 个 denoising steps 内不
重新采样。固定 noise SHA256：

```text
03c127eb5a7913510bb0ff836ddef3bc988e678ca8b52d92c5ec1b9f570ec2ec
```

loss：

```text
L_flow  = mean over 10 steps MSE(pred_flow, epsilon - x0)
L_state = MSE(final_denoised_latent, next_GT_latent)
L_camera = translation + rotation + intrinsics-weighted loss
L_total = lambda_flow*L_flow + lambda_state*L_state + lambda_camera*L_camera
```

### 4.5 Checkpoint

checkpoint 保存：

- task model delta 或 full model；
- optimizer / LR scheduler；
- global step / epoch / best loss / loss history；
- Python、NumPy、PyTorch、CUDA RNG；
- model/training config；
- token IDs / action vocabulary / camera representation；
- preprocessing hash / base checkpoint identity；
- fixed-noise seed/hash；
- warm-start report。

所有 `strict=False` 均只用于已枚举 task delta；全部 expected missing keys 会
打印并逐集合检查，unexpected 必须为空。保存后通过 fresh-model output parity
测试。

公共代码：

- `full_duplex/tokens.py`
- `full_duplex/model.py`
- `full_duplex/flow.py`
- `full_duplex/camera.py`
- `full_duplex/training.py`
- `full_duplex/predict_checkpoint.py`
- `full_duplex/evaluate_predictions.py`
- `full_duplex/summarize_metrics.py`
- `full_duplex/decode_predictions.py`

---

## 5. 做法一：旧 Autoregressive rollout + 跨轮 BPTT

### 5.1 核心定义

```text
turn 0 world input = ZERO + NULL_WORLD_STATE
turn t>0 world input = pred_state[t]
target at turn t = GT state[t+1]
pred_state[t] -> next turn input, no detach
```

camera 也使用上一轮预测 camera。完整 19 turns 形成连续计算图：

```text
pred_0 -> input_1 -> pred_1 -> ... -> pred_18
```

19 turn loss 取 mean 后整体 backward。turn 0 的预测不仅收到本 turn state
loss，还能收到未来 turn loss 的梯度。实测 future-only gradient norm
`0.084259`，证明跨 turn BPTT 真实存在。

### 5.2 对应文件

| 文件 | 作用 |
|---|---|
| `full_duplex/training.py` | `FullDuplexTrainer.forward_loss/train_step`，autoregressive graph、BPTT、loss、checkpoint |
| `full_duplex/train_overfit.py` | 训练 worker CLI，控制 steps/blocks/stride/denoise |
| `full_duplex/control_training.py` | 同步前台 controller、日志/status/异常处理 |
| `full_duplex/configs/overfit.yaml` | 公共路径、loss 权重和默认超参数 |
| `full_duplex/audit_gradients.py` | flow/state/camera 与 early-turn gradient 审计 |

### 5.3 资源探索

- native stride 1、30 blocks、19 turns、10 denoise 的大图真实 OOM，H100 使用
  约 79.09 GiB；
- stride 8、30 blocks、19 turns 的单步 backward 可以通过：44.18 GiB、
  160.94 秒；
- 为运行数百步，最终长期配置选择 stride 8、1 block、19 turns、10 denoise；
- 最大显存 11.93 GiB，平均 13.43 秒/optimizer step。

### 5.4 最终训练配置

```text
run_name = rollout_19turn_stride8_1block_worldprior_final200
num_micro_turns = 19
num_denoising_steps = 10
num_backbone_blocks = 1
spatial_token_stride = 8
max_history_turns = -1
teacher_forcing_ratio = 0.0
detach_between_turns = false
train_backbone = false
world_residual_head = true
world_time_space_prior = true
trainable parameters = 8,190,055
optimizer steps = 200
```

训练过程中 world head 和 time×space prior 使用独立 LR group；训练由多个
resume 阶段完成，optimizer moments 没有被丢弃。准确 staged LR 与历史保存在
最终 checkpoint 和 `run_manifest.json`。

### 5.5 Loss 与评估

| 指标 | Step 1 | Step 100 | Step 200 |
|---|---:|---:|---:|
| total loss | 5.706124 | 2.640217 | 2.534786 |
| flow loss | 2.865153 | 1.321738 | 1.268220 |
| state MSE | 2.755545 | 1.318127 | 1.266439 |
| camera loss | 0.085427 | 0.000352 | 0.000127 |

- total loss 下降 `55.58%`；
- 92.46% 相邻 step 的 total loss 下降；
- best step = 200；
- reload flow/camera max error = `0.0 / 0.0`。

fresh checkpoint autonomous rollout：

| 指标 | 结果 |
|---|---:|
| overall latent MSE | 1.266472 |
| latent cosine | 0.413235 |
| predicted mean/std | 0.102214 / 1.170548 |
| target mean/std | 0.098777 / 0.828415 |
| camera translation L2 | 0.012891 |
| camera rotation | 0.489831° |
| camera intrinsics RMSE | 0.007636 |

### 5.6 最佳 checkpoint 和视频

```text
/hyperai/home/minwm/full_duplex/outputs/smallest_000000/
rollout_19turn_stride8_1block_worldprior_final200/checkpoints/best.pt
```

视频：

```text
/hyperai/home/minwm/full_duplex/outputs/smallest_000000/
rollout_19turn_stride8_1block_worldprior_final200/video_export/
prediction_step_000200.mp4
```

规格：19 predicted states → 73 frames，832×480，24 FPS，3.0417 秒。视频有
明显彩色高频/块状噪点，没有达到清晰重现。

### 5.7 旧方案优缺点

优点：

- 符合真正部署时的 rollout：除 turn 0 外不依赖 world GT；
- 训练 exposure error；
- 未来 loss 能修正早期预测；
- 可以作为 autonomous world model 使用。

缺点：

- 保存 19 轮 × 10 denoise 的长图；
- 完整层数/空间 token 时显存与时间非常高；
- 长期训练只能降低 blocks/stride；
- 早期误差会传播并污染后续输入。

---

## 6. 做法二：新 Previous-GT → Next-GT + 逐轮 backward

### 6.1 核心定义

严格映射：

```text
turn 0:  ZERO/NULL       -> predict GT state[1]
turn 1:  GT state[1]     -> predict GT state[2]
turn 2:  GT state[2]     -> predict GT state[3]
...
turn 18: GT state[18]    -> predict GT state[19]
```

每轮 state loss 仍是：

```text
MSE(predicted final state[t+1], GT state[t+1])
```

当前目标 GT 仍被 masked，模型只能读取 previous GT world input，不能读取
当前 target。

### 6.2 反向传播方式

每个 optimizer step 仍处理全部 19 transitions，但不构建跨轮图：

```python
for turn in 19 turns:
    result = denoise_10_steps(previous_GT, current_target)
    (result.total_loss / 19).backward()
    save_history_values(result.detach())
optimizer.step()
```

参数梯度在 19 轮之间累积，一次 optimizer step 仍对应 19 个独立 transition
loss 的平均值。单元测试证明这种 sequential backward 与整体 mean loss 的
parameter gradient 数学等价。

保留和删除的图：

| 图路径 | 状态 |
|---|---|
| 当前 turn 内 10-step differentiable denoising | 保留 |
| 当前 turn world/state/flow/camera loss | 保留 |
| 历史 prediction value 作为可见 token | 保留数值 |
| 历史 prediction 的 grad_fn | 删除/detach |
| `pred_state_t -> input_state_t+1` | 不存在；改为 GT input |
| future loss -> early prediction | 不存在 |

camera 没有 teacher forcing：turn `t>0` 仍使用上一轮预测 camera。因此新方案
只改变了 world input 语义。

### 6.3 对应文件

| 文件 | 作用 |
|---|---|
| `full_duplex/teacher_forcing_training.py` | previous GT input、逐轮 backward、detached history、strict warm-start |
| `full_duplex/train_teacher_forcing.py` | 独立 CLI、warm-start/resume、blocks/stride/checkpoint blocks |
| `full_duplex/tests/test_teacher_forcing.py` | 零首状态、GT 精确 view、无跨轮 graph、gradient 等价测试 |
| `full_duplex/predict_checkpoint.py` | 按 `training_regime` 自动选择旧/新 predictor |
| `full_duplex/model.py` | partial activation checkpoint：只重计算前 N blocks |
| `full_duplex/summarize_metrics.py` | 记录 input/target state index 和 teacher-forcing flags |

### 6.4 严格运行标志

```text
training_regime = teacher_forced_previous_gt_transition
teacher_forcing_ratio = 1.0
teacher_forced_world_inputs = true
teacher_force_camera = false
sequential_turn_backward = true
detach_between_turns = true
cross_turn_bptt = false
```

per-turn CSV 明确记录：

```text
turn 0: input_state_index=-1, target_state_index=1
turn 1: input_state_index=1,  target_state_index=2
...
turn 18: input_state_index=18, target_state_index=19
```

### 6.5 Transformer 层数与显存消融

初始 task delta 从旧方案 step-200 warm-start，stride 8、19 turns、10 denoise：

| executed blocks | Step-1 state MSE | Total loss | 峰值 GiB | 秒/步 |
|---:|---:|---:|---:|---:|
| 4 | 1.392140 | 2.835983 | 6.988 | 24.14 |
| 8 | 1.499942 | 3.053548 | 7.413 | 35.81 |
| 16 | 1.924277 | 4.474695 | 8.267 | 57.38 |
| 30 | 3.607297 | 7.209465 | 9.761 | 89.23 |

更深层初始 loss 上升，是旧 task head 从 1-block hidden distribution 切换到
30-block distribution 的适配成本。30 层在前 10 步快速从 state MSE 3.6073
降到 1.5541，证明可以适配，因此选择完整 30 层继续。

activation checkpoint 消融：

- checkpoint 全 30 blocks：约 9.82 GiB，约 82–90 秒/步；
- 完全不 checkpoint：使用约 79.15 GiB 后 OOM；
- checkpoint 前 10/30 blocks：约 72.90 GiB，约 60–73 秒/步；
- 最终选择 checkpoint 前 10 blocks。

### 6.6 最终训练配置

```text
run_name = teacher_forced_b30_s8_ckpt10
num_micro_turns = 19
num_denoising_steps = 10
num_backbone_blocks = 30
spatial_token_stride = 8
max_history_turns = -1
gradient_checkpointing_blocks = 10
teacher_forcing_ratio = 1.0
detach_between_turns = true
train_backbone = false
world_residual_head = true
world_time_space_prior = true
lora_enabled = false
trainable parameters = 8,190,055
optimizer steps = 100
```

### 6.7 Loss 与评估

| Step | Total | Flow | State MSE | Camera |
|---:|---:|---:|---:|---:|
| 1 | 7.209465 | 3.470663 | 3.607297 | 0.131505 |
| 10 | 3.128948 | 1.560910 | 1.554140 | 0.013898 |
| 20 | 2.788138 | 1.398338 | 1.385512 | 0.004288 |
| 40 | 2.614527 | 1.312188 | 1.299950 | 0.002390 |
| 60 | 2.562689 | 1.286790 | 1.275678 | 0.000222 |
| 80 | 2.547594 | 1.278557 | 1.268835 | 0.000202 |
| 100 | 2.540905 | 1.274728 | 1.266101 | 0.000075 |

- total loss 下降 `64.76%`；
- 99/99 相邻 step 全部下降；
- best step = 100；
- 平均 65.58 秒/step；
- 最大记录显存 72.90 GiB；
- reload flow/camera max error = `0.0 / 0.0`。

fresh teacher-forced reconstruction：

| 指标 | 结果 |
|---|---:|
| overall latent MSE | 1.266015 |
| latent cosine | 0.416657 |
| predicted mean/std | 0.098531 / 1.174650 |
| target mean/std | 0.098777 / 0.828415 |
| best turn | turn 0, MSE 1.207818 |
| worst turn | turn 18, MSE 1.292603 |
| camera translation L2 | 0.008279 |
| camera rotation | 0.689707° |
| camera intrinsics RMSE | 0.004248 |

### 6.8 最佳 checkpoint 和视频

```text
/hyperai/home/minwm/full_duplex/outputs/smallest_000000/
teacher_forced_b30_s8_ckpt10/checkpoints/best.pt
```

独立预测 latent：

```text
/hyperai/home/minwm/full_duplex/outputs/smallest_000000/
teacher_forced_b30_s8_ckpt10/prediction_step100.pt
```

VAE 视频：

```text
/hyperai/home/minwm/full_duplex/outputs/smallest_000000/
teacher_forced_b30_s8_ckpt10/video_export/
prediction_step_000100_teacher_forced.mp4
```

规格：73 frames、832×480、24 FPS、3.0417 秒、H.264 CRF 18。latent/RGB
finite，VAE decode 峰值 15,407,987,200 bytes。

provenance：

```text
ground_truth_world_inputs_used_for_prediction = true
ground_truth_camera_inputs_used_for_prediction = false
ground_truth_current_output_visible_to_model = false
ground_truth_decoded = false
```

这段视频是 teacher-forced next-state reconstruction，不是 autonomous rollout。

### 6.9 新方案优缺点

优点：

- 删除 19-turn 跨轮 graph；
- 完整 30/30 Wan blocks 可运行；
- 每个 transition 的 previous world state 没有 exposure error；
- 单 transition 调试和过拟合更直接；
- camera translation 指标更好。

缺点：

- 推理依赖 previous world GT，不能单独做 autonomous rollout；
- 不训练 error accumulation 和 recovery；
- future loss 不再修正 early prediction；
- 完整 30 层使计算/显存仍很高；
- 最终 world latent/RGB 质量没有实质性提升。

---

## 7. 两种做法的直接对比

### 7.1 训练语义

| 项目 | 旧方案 | 新方案 |
|---|---|---|
| turn 0 world input | zero/null | zero/null |
| turn `t>0` world input | `pred_state[t]` | `GT state[t]` |
| 当前 next-state GT | masked target | masked target |
| state MSE target | `GT state[t+1]` | `GT state[t+1]` |
| historical prediction graph | 保留 | detach |
| cross-turn BPTT | 是 | 否 |
| turn 内 10-step graph | 是 | 是 |
| camera recurrence | predicted | predicted |
| autonomous rollout | 是 | 否 |

### 7.2 实际配置不是严格单变量消融

| 条件 | 旧最终 run | 新最终 run |
|---|---:|---:|
| steps | 200 | 100 |
| blocks | 1 | 30 |
| stride | 8 | 8 |
| trainable params | 8.19M | 8.19M |
| backbone | frozen | frozen |
| world initialization | 分阶段旧训练 | warm-start 旧 step 200 |
| max memory | 11.93 GiB | 72.90 GiB |
| mean step time | 13.43 s | 65.58 s |

新方案从旧 step-200 task delta warm-start，并同时改变执行深度，所以不能把
所有变化归因于 previous GT。要做科学严格消融，需要相同初始化、相同 blocks、
相同 steps，只改变 world input/gradient boundary。

### 7.3 相同 Step 100 的训练 loss

| 指标 | 旧 step 100 | 新 step 100 | 新方案变化 |
|---|---:|---:|---:|
| total | 2.640217 | 2.540905 | -3.76% |
| flow | 1.321738 | 1.274728 | -3.56% |
| state MSE | 1.318127 | 1.266101 | -3.95% |
| camera | 0.000352 | 0.000075 | -78.54% |

该表是实际值，不是严格因果消融。

### 7.4 各自最终最佳 checkpoint

| 指标 | 旧 step 200 | 新 step 100 | 解读 |
|---|---:|---:|---|
| training total | 2.534786 | 2.540905 | 旧低 0.24% |
| fresh state MSE | 1.266472 | 1.266015 | 新低 0.036%，基本持平 |
| latent cosine | 0.413235 | 0.416657 | 新高 0.83% |
| camera translation | 0.012891 | 0.008279 | 新低 35.78% |
| camera rotation | 0.489831° | 0.689707° | 旧低 28.98%（新高 40.81%） |

两个方案 predicted latent 之间 MSE `0.004750`，解码视频逐帧：

| 视频相似度 | 结果 |
|---|---:|
| PSNR | 36.9896 dB |
| SSIM | 0.982617 |

高 SSIM 说明新旧输出非常相似。新方案没有把旧方案的噪点视频变成清晰视频。

### 7.5 为什么最终结果接近

1. 两个方案都使用 stride 8。30×52 patch grid 仅取 28 个 world tokens，
   最后插值回 60×104；高频空间信息在模型输入/输出瓶颈处已被丢弃。
2. 两个方案都冻结 1.49B Wan backbone，只训练 8.19M task modules。
3. 新方案虽然执行 30 blocks，但 task delta warm-start 自旧 1-block hidden
   distribution，需要先重新适配。
4. predicted latent std 都约 1.17，而 target std 仅 0.828，噪声幅度问题仍在。
5. 去掉跨 turn exposure error 不能恢复 stride 8 丢掉的空间细节。
6. 100 步末新方案 state MSE 已在 1.266 附近明显平台。

---

## 8. LoRA 旁路实验（已放弃）

在两条主方案之间，我们还实现过只给物理最后若干 Transformer blocks 加
LoRA 的独立路径：

- 完整执行 30 blocks；
- blocks 26..29 加 rank-8 LoRA；
- 40 个 Linear modules；
- LoRA trainable elements = 1,458,176；
- 原 Wan 和 8.19M task delta 冻结；
- 19 turns、10 denoise，约 43.42 GiB、100–107 秒/step。

step 1→10 total loss `14.598776→12.106856`，但 fresh MSE/cosine 为
`6.110075/0.093171`，远差于旧主线 `1.266472/0.413235`；camera 也退化。
因此 LoRA plumbing 验证成功，但质量失败，按用户决定停止并回到非 LoRA
方案。

对应文件：

- `full_duplex/lora.py`
- `full_duplex/train_lora.py`
- `full_duplex/tests/test_lora.py`
- `full_duplex/LORA_REPORT.md`

---

## 9. VAE 视频与结果解读

### 9.1 旧视频

```text
/hyperai/home/minwm/full_duplex/outputs/smallest_000000/
rollout_19turn_stride8_1block_worldprior_final200/video_export/
prediction_step_000200.mp4
```

- autonomous rollout，不使用 world GT；
- 73 frames，832×480，24 FPS；
- 粗色块/高频噪点明显；
- 证明 end-to-end rollout→VAE→MP4 路径工作，但不代表高质量预测。

### 9.2 新视频

```text
/hyperai/home/minwm/full_duplex/outputs/smallest_000000/
teacher_forced_b30_s8_ckpt10/video_export/
prediction_step_000100_teacher_forced.mp4
```

- previous world GT teacher forcing；
- 当前 target GT 对模型不可见；
- decoder 只解码 predicted latent，没有用 GT 替换；
- 73 frames，832×480，24 FPS；
- camera/粗轮廓可见，但仍有密集彩色噪点；
- 与旧视频 SSIM 0.982617，视觉质量基本相同。

### 9.3 Denoising steps 消融

旧 checkpoint 用相同 fixed noise 测过 5/10/20/30 inference Euler steps：

| Steps | MSE | Cosine | Camera translation | Rotation |
|---:|---:|---:|---:|---:|
| 5 | 1.266417 | 0.415925 | 0.049090 | 3.1543° |
| 10 | 1.266472 | 0.413235 | 0.012891 | 0.4898° |
| 20 | 1.266896 | 0.410843 | 0.069755 | 2.1640° |
| 30 | 1.267201 | 0.409737 | 0.094575 | 2.7760° |

增加 inference denoising steps 没有降低 world MSE，反而损害 camera。因此当前
噪点不是简单由“采样步数不够”造成，10 步仍是匹配训练的选择。

---

## 10. 训练与复现入口

### 10.1 环境

```bash
conda activate /hyperai/home/conda_envs/minwm
cd /hyperai/home/minwm
export PYTHONPATH=/hyperai/home/minwm:/hyperai/home/minwm/Wan21:/hyperai/home/minwm/shared
```

### 10.2 审计与缓存

```bash
python full_duplex/audit_environment.py
python full_duplex/audit_checkpoint.py --config full_duplex/configs/overfit.yaml
python full_duplex/audit_data.py --config full_duplex/configs/overfit.yaml
python full_duplex/preencode.py --config full_duplex/configs/overfit.yaml
python -m unittest discover -s full_duplex/tests -v
```

### 10.3 旧方案入口

```bash
python -u full_duplex/control_training.py \
  --mode rollout \
  --run-name rollout_19turn_stride8_1block_worldprior_final200 \
  --max-steps 200 \
  --num-denoising-steps 10 \
  --blocks 1 \
  --spatial-token-stride 8 \
  --attention-pad-to-turns 19 \
  --freeze-backbone \
  --world-residual-head \
  --world-time-space-prior
```

旧最终结果由多阶段 resume/LR 调整得到；精确恢复应直接加载最终 checkpoint
中的 training config 与 optimizer state，而不是假设上面单一命令能 bitwise
重现全部历史。

### 10.4 新方案入口

```bash
python -u -m full_duplex.train_teacher_forcing \
  --warm-start full_duplex/outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/checkpoints/best.pt \
  --run-name teacher_forced_b30_s8_ckpt10 \
  --max-steps 100 \
  --num-turns 19 \
  --num-denoising-steps 10 \
  --blocks 30 \
  --spatial-token-stride 8 \
  --attention-pad-to-turns 19 \
  --checkpoint-blocks 10
```

恢复时优先使用 `best.pt`，因为 milestone `latest.pt` 在被中断时可能比逐步
更新的 best 更旧：

```bash
python -u -m full_duplex.train_teacher_forcing \
  --resume RUN/checkpoints/best.pt \
  --run-name teacher_forced_b30_s8_ckpt10 \
  --max-steps 100
```

### 10.5 预测、评估与视频

```bash
python -m full_duplex.predict_checkpoint \
  --checkpoint RUN/checkpoints/best.pt \
  --output RUN/prediction.pt

python -m full_duplex.evaluate_predictions \
  --predictions RUN/prediction.pt \
  --checkpoint RUN/checkpoints/best.pt \
  --output RUN/evaluation.json

python -m full_duplex.decode_predictions \
  --predictions RUN/prediction.pt \
  --vae-checkpoint ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
  --project-root /hyperai/home/minwm \
  --output RUN/prediction.mp4 \
  --fps 24 --crf 18
```

---

## 11. 关键文件索引

### 审计与数据

- `full_duplex/audit_environment.py`
- `full_duplex/audit_checkpoint.py`
- `full_duplex/audit_data.py`
- `full_duplex/preencode.py`
- `full_duplex/AUDIT.md`

### 模型与协议

- `full_duplex/tokens.py`
- `full_duplex/model.py`
- `full_duplex/camera.py`
- `full_duplex/flow.py`
- `full_duplex/inspect_sequence.py`
- `full_duplex/visualize_mask.py`

### 旧训练路径

- `full_duplex/training.py`
- `full_duplex/train_overfit.py`
- `full_duplex/control_training.py`

### 新训练路径

- `full_duplex/teacher_forcing_training.py`
- `full_duplex/train_teacher_forcing.py`
- `full_duplex/tests/test_teacher_forcing.py`

### 预测、评估和视频

- `full_duplex/predict_checkpoint.py`
- `full_duplex/evaluate_predictions.py`
- `full_duplex/summarize_metrics.py`
- `full_duplex/decode_predictions.py`

### 报告

- `full_duplex/FINAL_REPORT.md`
- `full_duplex/TEACHER_FORCING_REPORT.md`
- `full_duplex/NEW_VS_OLD_TRAINING_REPORT.md`
- `full_duplex/LORA_REPORT.md`
- `full_duplex/ACCEPTANCE_CHECKLIST.md`
- `full_duplex/log/log.md`（本报告）

---

## 12. 测试与数值验证

最终运行过的测试覆盖：

- cache real shape/finiteness；
- camera representation roundtrip/loss；
- Flow target 符号和 Euler 解；
- special vocabulary/token layout；
- causal/prediction mask；
- attention padding buckets；
- LoRA zero residual/physical last blocks；
- teacher-forced zero first input；
- turn 1..18 previous GT bitwise equality；
- historical prediction 无 grad_fn；
- sequential backward 与 mean-loss gradient 等价。

最终单元测试：`15/15 PASS`。

额外运行时检查：

- strict base load missing/unexpected = 0/0；
- task-delta expected missing 精确集合匹配；
- model output、loss、gradient finite；
- gradient norm / parameter norm；
- initial fixed noise mutation 断言；
- checkpoint reload flow/camera output max error = 0/0；
- 100-step checkpoint 含 model/optimizer/scheduler/RNG/config/vocabulary/hash；
- best/latest 指向 step 100，model tensors 精确相同；
- VAE 输出 73 帧，ffprobe 验证 H.264、832×480、24 FPS、3.041667 秒。

---

## 13. 当前完成度与未达成项

### 已完成

- checkpoint 严格加载与真实前向；
- 真实 VAE/T5 预编码和可复用 cache；
- 19 actions ↔ 20 latents 对齐；
- Full-Duplex token 序列、mask、显式 camera/action/noise/world；
- 文本 cross-attention、RoPE、PRoPE；
- 10-step fixed-noise differentiable Flow Matching；
- world flow/state 和 camera loss；
- 旧 19-turn autoregressive BPTT；
- 新 previous-GT→next-GT sequential backward；
- checkpoint best/latest/resume/reload；
- 200-step 旧训练与 100-step 新训练；
- per-turn metrics、曲线、CSV、fresh evaluation；
- VAE 视频导出与旧/新视觉对比。

### 尚未达成

- 两种方案的 world latent MSE 都未低于 zero baseline `0.696027`；
- predicted latent std 仍明显大于 target；
- RGB 视频仍以噪点为主，没有清晰人物/场景复现；
- 新方案依赖 previous GT，不能取代旧 autonomous rollout；
- 还没有相同初始化/相同 blocks/相同 steps 的严格单变量新旧消融；
- 还没有完成 stride 4 的 teacher-forced 训练；
- 没有证明对最小样本之外的数据具有泛化能力。

---

## 14. 下一步建议

按当前证据，继续在完全相同的 stride-8 配置上从 100 盲目训练到 500，预期
收益较低。推荐顺序：

1. **严格对照实验**：固定初始化、blocks、stride、steps，只切换
   autoregressive vs previous-GT；
2. **stride 8 → 4**：优先增加空间 token 密度，观察 latent MSE 和 RGB
   噪点是否实质改善；
3. **显存探测**：stride 4 下逐步探测 blocks 4→8→16→30，并使用 partial
   activation checkpoint；
4. **增加可训练容量**：若只训练 8.19M task delta 已平台，考虑解冻靠近输出
   的 block/head 或使用与 hidden distribution 匹配的适配方式；
5. **双指标报告**：同时保留 teacher-forced reconstruction 与 autonomous
   rollout，避免把使用 GT 的结果当成自主预测；
6. **噪声幅度校正**：重点分析 predicted latent std 1.17 vs target 0.828，
   检查输出尺度、world prior 与 Flow/state loss 权重。

当前最有可能改善清晰度的变量是空间 token stride 和可训练容量，而不是单纯
增加 inference denoising steps。

---

## 15. 最终结论

本任务已经完成了一个可运行、可检查、可恢复的 Full-Duplex latent 微调系统，
并真实验证了两种 world-state 训练范式：

- 旧 autoregressive BPTT 具备 autonomous rollout 语义，但长图限制资源；
- 新 previous-GT sequential training 显著降低跨轮图压力并执行完整 30 层，
  但依赖 GT，且目前没有带来明显视频质量提升。

因此，新思路的主要成功是**训练系统与显存结构成功**，而不是**最终视觉质量
已经成功**。当前 checkpoint 可以继续作为 stride 4、更多可训练容量和严格
单变量消融的基础，但不应被描述成已经完成清晰视频复现。
