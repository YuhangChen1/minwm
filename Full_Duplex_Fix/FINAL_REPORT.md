# Full_Duplex_Fix 验证报告

> 历史记录：本报告中的指标来自旧的 39-span 主协议。当前代码已切换为包含
> `W19` 的 40-span、62,400-token 协议；必须重新运行真实模型验证后才能更新
> 本报告中的资源和数值结论。

日期：2026-07-16  
仓库：`/workspace/yuhang/minwm`  
验证设备：NVIDIA H200，单 GPU 运行每项真实模型测试  
软件：PyTorch `2.9.1+cu128`，CUDA runtime `12.8`

## 结论范围

本次完成的是改造代码、真实单样本预编码、核心语义测试、完整模型前后向、一次
优化器 step、自主推理缓存闭环和 VAE 导出验证。没有运行 2000-step 单样本过拟合，
也没有产生训练后的 `best.pt`，因此不能声称模型已经学会该视频或单状态分解已经
取得科学效果。

实现全部位于 `Full_Duplex_Fix/`。没有为本方案修改 `Wan21/` 共享源码；adapter
没有新增可训练参数。

## 已验证事实

### 数据与预编码

- 输入视频实际解码为 77 帧、480x832、24 FPS。
- 冻结 Wan VAE 输出 `[20,16,60,104]`，保存为 FP16。
- latent 统计：min `-3.25`、max `3.453125`、mean `0.0943720`、std
  `0.8160623`，全部 finite。
- 正/负文本各经冻结 UMT5 编码一次，输出 `[512,4096]` BF16；有效 token 数
  分别为 132 和 126，padding embedding 为零。
- 使用仓库 `poses_from_pose_str("right-8, a-11")` 和
  `build_viewmats_and_Ks` 构建 20 个 camera states；`C0` 为单位外参，K 在整段
  视频中固定。
- action manifest 严格覆盖 19 个转移和源帧 1..76：8 个 `right` 后接 11 个
  `a`。
- cache 写入后逐 tensor bitwise reload 相等，并在每次加载时校验 7 个 tensor
  的独立 SHA256。预处理 hash 为
  `3f875b4d0b936b4cc43f527d1eca70dd9ffc3ca1b460f26b46e6959433ed7cb4`；
  world latent SHA256 为
  `6a03ccd0e71b404c2808d9d83b715d78ce3a10930d50c897893df69ef2f6dfb5`。

### Checkpoint 与参数

- 基础权重是 `ar_diffusion_tf/model.pt`，大小 `5,959,605,031` bytes，SHA256：
  `af73a86322f982cbab0446c6934f6f8dcc9f555f4d0652863baf04f4485a96dd`。
- strict load：885 tensors，`missing=[]`，`unexpected=[]`。
- 参数总数和实际 trainable 数均为 `1,489,821,760`；冻结参数数为 0，adapter
  新参数数为 0。
- 30 层 `prope_o` 全部来自 checkpoint。patch embedding、block 0 Q、block 0/29
  PRoPE 和 Flow Head 的代表性 tensor 已逐值检查与 checkpoint 相等。
- VAE 与 UMT5 不在训练 graph 中，只存在于预编码/解码阶段。

### Layout、mask、RoPE 与 PRoPE

- 主 layout 为 39 spans、20 noisy、19 clean、每 span 1560 tokens、总长
  60,840；顺序严格为 `N0,W0,...,N18,W18,N19`，不存在 `W19`。
- RoPE physical time 和 PRoPE camera index 都是
  `0,0,1,1,...,18,18,19`；没有使用 flat span index 20..38 作为时间。
- 显式坐标 RoPE 与原 Wan contiguous `rope_apply` 在单测中逐值完全相等。
- 18 项静态/单元测试全部通过，包含真实 cache、77/20/19 计数、camera、Flow
  target 符号、mask padding、layout、checkpoint 前缀、推理 cache 状态机和 W&B
  指标展开/CLI override。
- 真实 39-span 扰动测试通过：修改 `W5` 后 `N5` max abs difference 为 0，
  `N6` 为 `1.66796875`；修改 `N5` 后仅 `N5` 自身改变，其他 19 个 noisy
  输出差值全为 0。

### 4-state 排列等价性

BF16/autocast、完整 40 spans、30 blocks 的结果：

- 仓库原路径 vs adapter 原布局：max/mean/relative error 均为 0。
- adapter 原布局 vs 交错布局：max abs `0.1484375`，mean abs
  `0.00645965`，relative mean `0.00636065`。
- 显式 BF16 容差为 max abs `0.2`、relative mean `0.01`，结果 `passed=true`。

交错排列改变 FlexAttention 的 block/reduction 顺序，因此 BF16 不要求 bitwise
相等；约 0.636% 的相对均值误差在代码显式记录的容差内。仓库原路径与 adapter
原布局完全相等，排除了 adapter 重写本身的偏差。

FP32、完整 40 spans、30 blocks 也已真实完成：

- 仓库原路径 vs adapter 原布局：max abs `0.00119948`，mean abs
  `2.58297e-5`，relative mean `2.54720e-5`。
- adapter 原布局 vs 交错布局：max abs `0.00106025`，mean abs
  `2.59893e-5`，relative mean `2.56294e-5`。
- 交错 mean error / 原路径内核 baseline mean error 为 `1.00618`，即重排误差
  与两种 attention 调用路径本身的数值基线相当。
- 显式 FP32 容差为 max abs `0.002`、relative mean `5e-5`，结果
  `passed=true`。
- 三个 forward 分别耗时 `52.49s`、`642.94s` 和 `776.58s`；这是实际
  FP32 FlexAttention 路径，不是缩短序列或减少 block 的代理测试。

### 完整前后向与梯度

真实 39 spans、60,840 tokens、全部空间 token、30 blocks 的 BF16 autocast
forward/backward 已完成：

- Flow shape `[1,20,16,60,104]`，全部 finite。
- loss `0.07854443`，`L_init=0.11446774`，
  `L_transition=0.07665373`。
- forward `4.95s`，backward `9.64s`。
- 峰值显存 `40.422 GiB`，reserved `54.775 GiB`。
- patch embedding、block 0/15/29 Q 和 Flow Head 都有有限非零梯度。

另一次分项审计分别对 `L_init` 和 `L_transition` 反向传播。两者对上述五组
代表性参数的梯度都 finite 且非零。例如 Flow Head norm 分别为 `0.0839053` 和
`0.0445068`，证明初始化目标没有被训练图忽略。

一次真实 full-generator AdamW step 也已完成：

- global step `1`，constant LR scheduler epoch `1`。
- optimizer 为全部 885 组 parameter tensors 建立 state。
- grad norm 约 `0.9492`，step time `11.17s`，LR `2e-6`。
- 峰值显存 `40.447 GiB`；step 后 optimizer/model 常驻约 `22.31 GiB`。

这只证明优化器路径可运行，不能用于判断 loss 趋势。

W&B `0.25.1` 已接入正式过拟合和单步 optimizer smoke 路径。真实 offline run 已
验证初始化、逐 state 指标写入、run ID 落盘和正常结束；online run 尚未执行。
checkpoint 默认只在 W&B summary 登记路径、step、大小和 best 指标，不上传大型
`.pt` artifact；checkpoint 内保存 run ID，以便 resume 延续同一 run。

### 自主推理与 VAE 导出

使用未微调的基础 AR 权重进行了完整 20-state、1-step UniPC 诊断 rollout：

- 从 20 个独立噪声开始，没有向 sampler 注入 GT latent。
- 正/负 CFG 分支各自拥有独立 normal RoPE、PRoPE 和 cross-attention cache。
- 每个 denoising call 后逐层检查 cache endpoint；同一物理时间没有追加位置。
- 每个状态完成后以 timestep 0 重跑 clean state。
- 20 状态结束时 normal/PRoPE endpoint 都精确为 `31,200` tokens。
- 输出 `[1,20,16,60,104]` 全部 finite，总 sampler 时间约 `8.77s`。

冻结 VAE 成功解码为 `[1,3,77,480,832]`，全部 finite，并写出 24 FPS MP4、
contact sheet 和 JSON；解码耗时约 `4.38s`，编码文件回读确认仍为 77 帧、
480x832、24 FPS。

该基础权重、1-step 诊断结果质量很差：latent MSE `0.71954`，高于 zero-latent
baseline `0.67486`；cosine `0.19353`；prediction std `0.41592`，明显低于 target
std `0.81606`。contact sheet 除首帧色块外基本为灰色噪点，没有稳定结构。这是
预期的管线冒烟结果，不能描述成生成成功或接近 GT。

## 尚未完成

以下科学成功标准仍未达到，必须在正式实验中继续执行：

1. 运行配置中的 2000-step 单样本过拟合并保存 `initial/best/latest.pt`；训练指标
   同步记录到 W&B，并保留本地 JSONL。
2. 记录 `L_init`、`L_transition` 和 per-state loss 的真实下降趋势与 loss curve。
3. 对 `best.pt` 执行 `checkpoint_parity.py`，验证 optimizer/RNG/scheduler 恢复和
   fixed evaluation parity。
4. 从 fresh `best.pt` 运行正式 50-step、CFG=3.0 的完整自主 20-state rollout。
5. 比较 initial/current 的 latent MSE/cosine/std，确认是否低于 zero baseline。
6. 解码训练后 77 帧视频，并检查噪点、结构、camera motion 和 teacher-forcing /
   autonomous exposure gap。

## 主要产物

- `cache/smallest_000000/metadata.json`：真实数据/cache 审计。
- `outputs/mask/span_mask.txt` 和 `.png`：人类可读 39-span mask。
- `outputs/smallest_000000/permutation_equivalence.json`：BF16 排列等价性。
- `outputs/smallest_000000/smoke_model_backward.json`：完整前后向资源和梯度。
- `outputs/smallest_000000/no_leakage.json`：真实输入扰动结果。
- `outputs/smallest_000000/gradient_audit.json`：分项 loss 梯度。
- `outputs/smallest_000000/optimizer_step_smoke.json`：真实 AdamW step。
- `outputs/smallest_000000/base_autonomous_1step.pt/.json`：20-state 诊断 rollout。
- `outputs/smallest_000000/base_autonomous_1step.mp4`、contact sheet 和
  `*_decode.json`：
  冻结 VAE 导出。

这些运行产物和大型 cache/checkpoint 被 `.gitignore` 排除；代码、配置、复现命令
见 `README.md`。
