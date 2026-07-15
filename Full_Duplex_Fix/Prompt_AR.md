# 在 AR checkpoint 上实施单状态 Micro-Turn 改造的完整提示词

你是一名负责 Wan2.1、视频生成、Flow Matching、因果 Transformer、PRoPE 与分布式训练的高级算法工程师。请直接在当前仓库中完成“单状态 Micro-Turn AR”改造、训练入口补全和必要验证，不要只给设计建议或伪代码。

## 一、工作环境与唯一设计依据

- 项目根目录：`/mnt/onelab0/sub5-v2u2/cyh_area/data/0data/minWM`
- Python 环境：`conda activate minwm`
- GPU 验证统一使用：`CUDA_VISIBLE_DEVICES=5`
- 方案文件：`./Full_Duplex_Fix/proposal.md`
- 本路线的初始化 checkpoint：`./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt`
- 原始 AR 配置：`./Wan21/configs/ar_camera_tf.yaml`
- 原始 AR 启动脚本：`./Wan21/scripts/training/run_stage1_ar_camera.sh`
- 原始训练入口：`./Wan21/wan_train.py`
- 原始 AR Trainer：`./Wan21/wan_trainer/camera_ar_diffusion.py`
- 原始 AR loss/model：`./Wan21/model/camera_diffusion.py`
- 原始 causal backbone：`./Wan21/wan/modules/causal_model.py`
- Wan wrapper：`./Wan21/wan_utils/wan_wrapper.py`
- Dataset 与相机转换：`./Wan21/wan_utils/dataset.py`

开始修改前，必须完整阅读 `proposal.md`，再阅读上述相关代码，沿真实调用链理解张量排列、checkpoint 加载、scheduler、PRoPE、Teacher-Forcing mask、Sequence Parallel 和推理缓存。不要根据文件名猜测实现。

`proposal.md` 是功能设计依据；scheduler 的 Flow 方向、timestep 映射、latent 归一化、checkpoint key 和 PRoPE 数学实现必须以仓库现有代码为准。如果文字公式与现有 scheduler 的实际约定冲突，应保留现有 scheduler 约定，并在最终报告中明确指出。

## 二、最终目标

在已训练好的 AR checkpoint 上继续训练，把生成粒度从：

```text
4 个 VAE latent state / AR block
```

改造成：

```text
1 个 VAE latent state / micro-turn
```

一条 77 帧视频经冻结的 Wan2.1 VAE 编码后得到：

```text
X: [B,20,16,60,104]
x0,x1,...,x19
```

将其构造成 19 个真实状态转移：

```text
clean x0  + camera C0  -> noisy x1  + camera C1
clean x1  + camera C1  -> noisy x2  + camera C2
...
clean x18 + camera C18 -> noisy x19 + camera C19
```

其中 `Ct=(viewmat_t,K_t)` 是已知几何条件，不是待预测 latent。动作字符不进入 Transformer；如果推理端接收 `w/a/s/d/i/j/k/l`，动作只用于确定性计算下一相机状态 `C_(t+1)`。

## 三、必须保留的预训练能力

必须以以下相对路径加载模型，而不是从头初始化：

```text
./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```

这是已经完成 AR Teacher Forcing 和相机 PRoPE 训练的 checkpoint。目标模型必须继续使用 causal Wan backbone，并保留、加载和继续训练已有的：

- Conv3D Patch Embedding；
- 30 层 Wan Transformer；
- 普通时空 RoPE；
- PRoPE 及每层已经训练过的 `prope_o`；
- timestep embedding 和 AdaLN modulation；
- 文本投影与 Cross-Attention；
- FFN；
- Flow Prediction Head 和 Unpatchify。

加载 checkpoint 后必须输出并检查：

- missing keys；
- unexpected keys；
- shape mismatch；
- `prope_o` 是否成功加载且不是重新零初始化；
- Flow Head、Patch Embedding、首层和末层 Transformer 权重是否成功加载。

除非确有新增参数，否则不得静默使用 `strict=False` 掩盖不兼容。若必须非严格加载，应逐项解释每个 missing/unexpected key，不能只打印数量。

原 checkpoint 只能作为只读初始化来源。新日志和新 checkpoint 必须写入独立目录，例如：

```text
./logs/micro_turn_from_ar/
./ckpts/Wan21/Action2V/micro_turn_from_ar/
```

不得覆盖 `ar_diffusion_tf/model.pt`。

## 四、数据组织必须实现的语义

LMDB 单样本应继续读取：

```text
clean_latent: [20,16,60,104]
poses:        [20,7]
intrinsics:   [4]
prompt:       string
```

batch 后：

```text
X:        [B,20,16,60,104]
viewmats: [B,20,4,4]
Ks:       [B,20,3,3]
```

构造：

```text
clean_states  = X[:,0:19]   # x0...x18
target_states = X[:,1:20]   # x1...x19
current_cam   = C[:,0:19]
target_cam    = C[:,1:20]
```

禁止以下错误：

- 把 `x0` 替换为全零 world input；
- 训练成 `clean xt -> noisy xt`；
- 生成 `x_(t+1)` 时仍给 noisy target 配 `camera_t`；
- 让 camera 作为随机 embedding 或普通 camera token 代替 PRoPE；
- 给 camera 加扩散噪声；
- 增加 camera prediction head；
- 增加离散 action token、turn embedding、type embedding、MASK token或特殊协议 token。

默认继续使用冻结 VAE 预编码的 LMDB latent；只有原始视频调试路径才调用 VAE Encoder。VAE Encoder/Decoder 和 UMT5 必须冻结。

## 五、Flow Matching 必须复用原仓库实现

对 19 个 `target_states` 分别采样噪声和噪声等级。必须调用现有 scheduler 接口构造 noisy target、training target 和 timestep weight，不要另写一套符号可能相反的 Flow Matching：

```text
noise         与 target_states 同形状
noisy_targets [B,19,16,60,104]
flow_targets  [B,19,16,60,104]
weights       与 19 个 target 对齐
```

第一版将 `num_frame_per_block` 设为 `1`。每个 noisy target 可以具有独立 timestep，但必须确认现有 `_get_timestep()`、scheduler timestep 与 sigma 的映射完全正确。

clean span 沿用原 Teacher-Forcing clean-context timestep 约定：无 noise augmentation 时使用 clean timestep 0；如果保留可选的 clean augmentation，必须与原实现一致且默认关闭。不要新增 timestep token。

最终 loss 只计算 19 个 noisy target，不计算 clean span，也不计算 camera/action 分类损失。

## 六、Token packing、时间位置和角色信息

每个 state：

```text
[B,16,1,60,104]
 -> Conv3D patch embedding (1,2,2)
 -> [B,1560,1536]
```

19 个 transition 共包含 19 个 clean span 和 19 个 noisy span：

```text
38 × 1560 = 59280 tokens
Transformer hidden: [B,59280,1536]
```

逻辑序列为：

```text
clean x0, noisy x1,
clean x1, noisy x2,
...
clean x18, noisy x19
```

物理存储可以采用真正 interleaved packing，也可以采用更利于复用原实现的 contiguous packing，但必须满足完全相同的逻辑可见性、相机对齐、时间位置和 loss span。不得依赖“张量形状相同”猜测 span；必须显式维护稳定的 span metadata/index，例如：

- state id；
- role：clean/noisy；
- transition id；
- token start/end；
- camera id；
- temporal id；
- loss index。

普通 Wan 3D RoPE 的 temporal id 必须表示物理视频状态：

```text
clean x0 -> 0
noisy x1 -> 1
clean x1 -> 1
noisy x2 -> 2
...
noisy x19 -> 19
```

不能因为 packed sequence 有 38 个 span，就错误地把 temporal id 编成 `0...37`。同一 physical state 的 clean/noisy 表示必须共享同一个视频时间位置，但依靠 mask、timestep 和 loss index 区分角色。

## 七、Teacher-Forcing attention mask

必须为 38 个 span 构造可验证的显式 blockwise mask。

对于 `clean x_t` query：

```text
允许：clean x0 ... clean xt
禁止：所有 noisy span、未来 clean span
```

对于 `noisy x_(t+1)` query：

```text
允许：clean x0 ... clean xt
允许：自己的 noisy x_(t+1) span 内部双向注意力
禁止：clean x_(t+1) 及未来 clean span
禁止：其他所有 noisy span
```

尤其验证：

```text
noisy x1  可以读取 clean x0
noisy x2  可以读取 clean x0,x1
noisy x19 可以读取 clean x0...x18
```

不能直接使用普通下三角 token mask，因为它可能使 clean state 读取 earlier-packed noisy state，产生历史污染。也不能沿用原 4-latent TF mask 而不检查索引偏移。

为 mask 编写小尺寸单元测试或可读矩阵审计工具，至少检查第 0、1、中间和第 18 个 transition，并断言不存在 GT target 泄漏。

如果启用 Sequence Parallel，必须保证切分/重排前后的 mask、clean/noisy span、camera matrices、timestep modulation 和最终 noisy gather 完全对齐；单卡逻辑正确后再验证 SP。

## 八、PRoPE 相机对齐

每个 state 的 1560 个 token 必须使用对应相机：

```text
clean xt       -> viewmat_t,     K_t
noisy x_(t+1)  -> viewmat_(t+1), K_(t+1)
```

语义展开后的形状为：

```text
token_viewmats: [B,59280,4,4]
token_Ks:       [B,59280,3,3]
```

可以用广播、索引或按 span 展开，不要求永久物理复制，但传入每层 PRoPE 时必须与 Q/K/V token 顺序一一对应。

保留原 PRoPE 的确定性几何计算和普通 RoPE + PRoPE 并行残差结构，不得把具体轨迹写入 checkpoint，也不得把 `viewmat/K` 拼成主序列 token。

编写检查：改变某一 target camera 时，只有对应 span 的 camera 条件索引变化；clean `x_t` 和 noisy `x_t` 若同时存在，二者必须使用同一个 `C_t`。

## 九、Transformer 输出和 Flow Head

30 层 Transformer 输出仍为：

```text
[B,59280,1536]
```

只根据显式索引提取 19 个 noisy span：

```text
[B,29640,1536]
 -> [B,19,1560,1536]
 -> 原 Flow Head / Unpatchify
 -> [B,19,16,60,104]
```

不要默认“丢掉前一半就是 noisy”，除非物理 packing 明确是 `[all_clean; all_noisy]` 且有断言保证。无论 packing 方式如何，都必须使用明确索引并做形状断言。

训练监督为 scheduler 返回的 19 个 flow target。可视化或 VAE 解码时，禁止把 raw `flow_pred` 直接当成 clean latent；必须经 scheduler 的 `x0` 转换或完整 Flow solver 得到最终 clean latent后再解码。

## 十、训练策略

实现可配置的两阶段微调，但先保证端到端正确：

1. 单样本 overfit：冻结 VAE、UMT5 和大部分 Transformer，训练最后若干 block、PRoPE `prope_o`、LayerNorm/AdaLN、文本投影中必要部分及 Flow Head。
2. 基本链路成功后：低学习率解冻全部 30 层 Transformer、PRoPE、timestep modulation 和 Flow Head。

请确保冻结策略通过参数名白名单/断言实现，并打印 trainable/frozen 参数总量及关键模块状态。不要误冻 `prope_o`，也不要意外训练 VAE/UMT5。

单样本优先使用 proposal 指定数据：

```text
./dataset/SmallestData/000000_right8a11/gen.mp4
```

如果训练入口实际使用对应 LMDB，请找到或构建所需的现有 LMDB 路径，但不要擅自重写原始数据。优先复用 `CameraLatentLMDBDataset`、原相机归一化和原 latent normalization。

## 十一、推理要求

提供与训练语义一致的单状态递归推理路径：

```text
已知 x0,C0
action0 -> 确定性计算 C1 -> 从噪声生成 x1
action1 -> 确定性计算 C2 -> 从噪声生成 x2
...
```

每一步只预测一个 `[B,1,16,60,104]` latent state。动作字符不作为 Transformer token。

第一版可以使用无缓存、固定历史窗口或完整历史，但必须明确选择，并保证：

- 训练/推理 mask 语义一致；
- 全局 temporal id 不因每次新 forward 重置错误；
- 当前 clean history 使用各自 `C_t`；
- noisy target 使用目标 `C_(t+1)`；
- 使用原 Flow scheduler 多步去噪；
- 最终将 `x0...x19` 拼接后交给冻结 Wan VAE Decoder。

不要为追求“实时”擅自把 50 步 solver 改成 1 步；少步化属于后续蒸馏任务。

## 十二、代码组织与安全约束

- 新增独立、命名清楚的 micro-turn trainer/model/config/launch/inference 文件，避免破坏原 `ar_diffusion_tf` 和 bidirectional 路径。
- 优先复用现有 scheduler、wrapper、PRoPE、Dataset、checkpoint loader 和训练基础设施；不要复制大段模型代码形成难以维护的分叉。
- 对必须修改的共享代码增加显式模式开关，默认行为保持原样。
- 不删除或覆盖用户已有修改，不执行破坏性 git 命令。
- 不修改 `Full_Duplex_Fix/proposal.md`。
- 不引入 action embedding、camera encoder/head 或旧 Full-Duplex token 协议。
- 不开启长时间、无边界训练。先完成静态检查、单 batch forward/backward 和短 overfit smoke test。

## 十三、必须完成的验证

至少完成并报告：

1. checkpoint 加载审计，特别是 30 层 `prope_o`；
2. Dataset 单样本形状与 `x0:18 -> x1:19` 对齐；
3. 38 个 span、59280 token、19 个输出 target 的形状断言；
4. temporal id 审计，确认重复 physical state 使用相同 id；
5. camera id 审计，确认 clean/target 分别使用 `C_t/C_(t+1)`；
6. mask 单元测试，确认无 target/future/noisy-history 泄漏；
7. 单 batch forward：输出 `[B,19,16,60,104]` 且无 NaN/Inf；
8. 单 batch backward：关键可训练参数有有限、非零梯度，VAE/UMT5 无梯度；
9. scheduler 方向测试：使用已知 `x0/noise/sigma` 验证 noisy sample、flow target、`x0_pred` 转换一致；
10. 短单样本 overfit smoke test，记录 loss 是否下降；
11. 使用 Flow solver 得到 clean latent 后做少量 VAE decode 检查，不能解码 raw velocity；
12. 原 AR 配置的最小回归检查，确认新增模式未破坏旧路径。

所有 GPU 命令使用：

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n minwm <command>
```

如果依赖环境要求 `conda activate minwm`，可在登录 shell 中执行，但仍只暴露 GPU 5。

## 十四、最终交付格式

完成后不要只说“已修改”。最终报告必须包含：

- 新增和修改的文件清单；
- 从 LMDB 到 loss 的实际调用链；
- 每个关键张量的真实运行形状；
- AR checkpoint 加载结果及 missing/unexpected keys；
- mask 可见性规则和测试结果；
- PRoPE camera 对齐方式；
- 哪些参数训练、哪些冻结；
- 执行过的命令及结果；
- 尚未完成的长训练或风险；
- 新配置、启动脚本、推理脚本和新 checkpoint 输出相对路径。

请现在开始：先审计仓库和 proposal，给出简短实施计划，然后直接完成代码、测试和短 smoke run。不要停留在概念讨论阶段。
