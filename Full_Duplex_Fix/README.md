# Full_Duplex_Fix

本目录实现了基于 minWM/Wan2.1 AR checkpoint 的单状态交错式 Teacher-Forcing
实验。实现不修改 `Wan21/` 的共享源码，而是用参数为零的 runtime adapter 复用原
Wan generator、30 层 Transformer、RoPE、PRoPE、文本交叉注意力和 Flow Head。

## 实验协议

一条 77 帧、`[3,77,480,832]` 视频经冻结 Wan VAE 预编码为
`[20,16,60,104]`。每个 latent state 经原 `(1,2,2)` Conv3D patch embedding
得到 1560 tokens。主训练序列固定为：

```text
N0,W0,N1,W1,...,N19,W19
```

因此共有 20 个 noisy target、20 个 GT clean state、40 spans 和 62,400
tokens。`W19` 保留以维持对称形状和支持跨窗口 carry，但所有 noisy query 都
不能读取它。Mask 由显式 `role + physical_time` 构建：

```text
W_t -> W_0...W_t
N_t -> W_0...W_{t-1} + N_t
```

`N_t` 不能读取 `W_t` 或任何其他 noisy span。RoPE physical time 和 camera
index 都是 `0,0,1,1,...,19,19`。Camera 不作为 token，而是按 span 扩展
后进入原仓库 `prope_qkv`。文本由冻结 UMT5 预编码并通过原 cross-attention
注入。20 个 noisy span 的 Flow Matching target 都是 `epsilon - x`，总 loss
是 20 个状态等权平均，并分别记录 `L_init` 与 19 个 transition 的均值。

## 环境

所有命令从项目根目录执行：

```bash
cd /workspace/yuhang/minwm
export PYTHONPATH="$PWD:$PWD/Wan21:$PWD/shared:${PYTHONPATH:-}"
```

默认配置是 [configs/overfit.yaml](configs/overfit.yaml)。它绑定真实单样本、
VAE/T5、`ar_diffusion_tf/model.pt` 的路径、大小和 SHA256。当前经过验证的是单
H200 路径：`world_size=1`、无 FSDP、无 sequence parallel。旧 39-span 协议的
前后向峰值约 40.42 GiB；新的 62,400-token 协议需要重新记录真实峰值。

默认启用 W&B online 模式。API key 不写入 YAML、manifest 或 checkpoint；首次
运行前使用下列任一方式认证：

```bash
wandb login
# 或仅在当前 shell 中设置：export WANDB_API_KEY=...
```

无网络时传 `--wandb-mode offline`，完全禁用时传 `--no-wandb`。默认 project
为 `minwm-full-duplex-fix`，可用 `--wandb-project`、`--wandb-run-name`、
`--wandb-group` 和 `--wandb-tags` 覆盖运行信息。

## 执行顺序

1. 预编码冻结 VAE、UMT5 和 camera side stream：

```bash
python -m Full_Duplex_Fix.preencode \
  --config Full_Duplex_Fix/configs/overfit.yaml --device cuda:0
```

使用 `--force` 可强制重建。缓存保存到
`Full_Duplex_Fix/cache/smallest_000000/`，metadata 包含输入/checkpoint hash、
每个缓存 tensor 的 SHA256、latent 统计、19 个 action 对齐和 camera 约定；每次
加载都会重新校验 tensor 内容。

2. 检查 layout、mask 和静态/缓存测试：

```bash
python -m Full_Duplex_Fix.inspect_sequence
python -m Full_Duplex_Fix.visualize_mask
python -m unittest discover -s Full_Duplex_Fix/tests -v
```

人类可读 mask 位于 `Full_Duplex_Fix/outputs/mask/`。

3. 审计基础 AR checkpoint：

```bash
python -m Full_Duplex_Fix.audit_checkpoint \
  --config Full_Duplex_Fix/configs/overfit.yaml --device cpu
```

该命令要求 `missing=[]`、`unexpected=[]`、885 个 state tensors、30 个
`prope_o`，并逐值检查 patch embedding、早期 attention、首尾 PRoPE 和 Flow
Head 的代表性 tensor。

4. 运行原 4-state attention graph 的排列等价性：

```bash
python -m Full_Duplex_Fix.permutation_equivalence \
  --config Full_Duplex_Fix/configs/overfit.yaml --device cuda:0 --dtype bf16
python -m Full_Duplex_Fix.permutation_equivalence \
  --config Full_Duplex_Fix/configs/overfit.yaml --device cuda:0 --dtype fp32 \
  --output Full_Duplex_Fix/outputs/smallest_000000/permutation_equivalence_fp32.json
```

该测试临时使用完整 40 spans 和 `num_frame_per_block=4`，只改变 span 排列，
不会更新权重。实际 FP32 路径明显慢于 BF16；已有结果可在不重跑 Wan 的情况下
按当前显式容差重新判定：

```bash
python -m Full_Duplex_Fix.permutation_equivalence --assess-existing \
  --output Full_Duplex_Fix/outputs/smallest_000000/permutation_equivalence_fp32.json
```

5. 运行完整 40-span 真实模型验证：

```bash
python -m Full_Duplex_Fix.smoke_model \
  --config Full_Duplex_Fix/configs/overfit.yaml --device cuda:0 --backward \
  --output Full_Duplex_Fix/outputs/smallest_000000/smoke_model_backward.json
python -m Full_Duplex_Fix.no_leakage \
  --config Full_Duplex_Fix/configs/overfit.yaml --device cuda:0
python -m Full_Duplex_Fix.gradient_audit \
  --config Full_Duplex_Fix/configs/overfit.yaml --device cuda:0
python -m Full_Duplex_Fix.smoke_train_step \
  --config Full_Duplex_Fix/configs/overfit.yaml --device cuda:0
```

前两项分别验证完整 backward 和输入扰动不泄漏；`gradient_audit` 对
`L_init`、`L_transition` 分开反向传播；最后一项执行一次真实 AdamW step，
但不序列化大型 checkpoint。

6. 单样本过拟合：

```bash
python -m Full_Duplex_Fix.train_overfit \
  --config Full_Duplex_Fix/configs/overfit.yaml \
  --device cuda:0 --skip-preencode --max-steps 2000 \
  --wandb-run-name smallest-000000-overfit-2000
```

默认 `eval_every=2000`、`save_every=2000`、`log_every=1`：只在 step 0 和最终
step 2000 运行固定评估/保存 checkpoint，不再每 10 步改写 `.pt`；训练标量仍
逐步写入本地 JSONL 和 W&B。

恢复训练：

```bash
python -m Full_Duplex_Fix.train_overfit \
  --config Full_Duplex_Fix/configs/overfit.yaml \
  --device cuda:0 --skip-preencode --max-steps 4000 \
  --resume Full_Duplex_Fix/outputs/smallest_000000/latest.pt
```

checkpoint 保存完整 generator、AdamW、constant LR scheduler、global/best step、
Python/NumPy/PyTorch/CUDA RNG、resolved config、cache/checkpoint identity 和严格
resume contract，并保存 W&B run ID；恢复时自动继续同一个 run。完整 optimizer
checkpoint 很大，运行前应预留足够空间。

W&B 中记录总 loss、`L_init`、`L_transition`、20 个逐 state loss、梯度范数、
学习率、timestep/输出统计、耗时/显存，以及固定评估的 latent MSE、cosine、
20 个逐 state 指标和 zero baseline。`wandb_log_checkpoints: false` 默认只登记
checkpoint 路径、step、大小和 best 指标，不上传大型 `.pt` artifact；需要上传
时再显式修改配置。离线 run 可在训练后执行 `wandb sync <offline-run-directory>`。

生成 loss 曲线：

```bash
python -m Full_Duplex_Fix.plot_metrics
```

7. Fresh checkpoint 评估与保存/恢复一致性：

```bash
python -m Full_Duplex_Fix.checkpoint_parity \
  --config Full_Duplex_Fix/configs/overfit.yaml \
  --checkpoint Full_Duplex_Fix/outputs/smallest_000000/best.pt --device cuda:0
python -m Full_Duplex_Fix.evaluate \
  --config Full_Duplex_Fix/configs/overfit.yaml \
  --checkpoint Full_Duplex_Fix/outputs/smallest_000000/best.pt --device cuda:0
```

第一条应使用包含固定评估指标的 `initial.pt` 或 `best.pt`。第二条默认计算固定
teacher-forced 单步指标；增加 `--autonomous` 才运行完整 fresh T2V rollout。

8. 完全自主 T2V 推理：

```bash
python -m Full_Duplex_Fix.inference \
  --config Full_Duplex_Fix/configs/overfit.yaml \
  --checkpoint Full_Duplex_Fix/outputs/smallest_000000/best.pt \
  --device cuda:0 \
  --output Full_Duplex_Fix/outputs/smallest_000000/generated_latents.pt
```

默认执行配置中的 50-step UniPC、CFG=3.0，依次从噪声生成 `x0...x19`。每个
状态去噪期间只覆盖当前位置；生成完成后以 timestep 0、RoPE time `t`、camera
`C_t` 重跑并写入普通/PRoPE clean cache。推理不读取 GT latent。诊断时可显式
传 `--sampling-steps 1`，但该结果不能代替正式 50-step 评估。

9. 冻结 VAE 导出 77 帧视频：

```bash
python -m Full_Duplex_Fix.decode \
  --config Full_Duplex_Fix/configs/overfit.yaml \
  --latents Full_Duplex_Fix/outputs/smallest_000000/generated_latents.pt \
  --output Full_Duplex_Fix/outputs/smallest_000000/generated.mp4 \
  --device cuda:0
```

该命令检查 latent `[1,20,16,60,104]`，输出 77 帧、480x832、24 FPS MP4、
8 帧 contact sheet 和 `*_decode.json`；该命名不会覆盖 latent sampler 的
provenance JSON。

## 文件职责

- `layout.py`：显式 40-span metadata、token coordinates 和 gather index。
- `mask.py`：role/time FlexAttention mask、padding 规则和可读矩阵。
- `rope.py`：由显式 `(t,h,w)` 坐标应用原 Wan 3D RoPE 频率。
- `model.py`：无新增参数的交错 Wan adapter、PRoPE 和 noisy-only Flow Head。
- `preencode.py` / `data.py`：真实单样本 VAE/T5/camera 缓存与审计。
- `flow.py`：原 scheduler 噪声、target、权重、20-state loss 和 x0 恢复。
- `training.py`：完整 generator 微调、固定评估、日志、checkpoint/resume。
- `wandb_tracking.py`：正式训练/单步优化器 smoke 共用的 W&B 生命周期和指标展开。
- `inference.py`：CFG、UniPC、普通/PRoPE/cross-attention 独立 cache 的自主 rollout。
- `decode.py`：冻结 VAE 视频、contact sheet 和 provenance 导出。
- `permutation_equivalence.py`、`no_leakage.py`、`gradient_audit.py`：关键语义验证。

## 结果边界

代码和当前真实验证结果记录在 [FINAL_REPORT.md](FINAL_REPORT.md)。一次基础权重、
1-step sampler 的输出只验证自主推理和 VAE 管线，它明显含噪，不代表方案已经
过拟合。只有 2000-step 训练、fresh reload、正式 50-step rollout 和可视化结果
共同改善后，才能讨论该单样本实验的科学成功。
