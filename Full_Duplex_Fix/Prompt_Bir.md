# 在 Bidirectional checkpoint 上实施单状态 Micro-Turn 改造的完整提示词

你是一名负责 Wan2.1、视频生成、Flow Matching、因果 Transformer、PRoPE 与分布式训练的高级算法工程师。请直接在当前仓库中完成“由 Bidirectional checkpoint 初始化的单状态 Micro-Turn AR”改造、训练入口补全和必要验证，不要只给设计建议或伪代码。

## 一、工作环境与唯一设计依据

- 项目根目录：`/mnt/onelab0/sub5-v2u2/cyh_area/data/0data/minWM`
- Python 环境：`conda activate minwm`
- GPU 验证统一使用：`CUDA_VISIBLE_DEVICES=5`
- 方案文件：`./Full_Duplex_Fix/proposal.md`
- 本路线的初始化 checkpoint：`./ckpts/Wan21/Action2V/bidirectional/model.pt`
- 原 Bidirectional 配置：`./Wan21/configs/bidirectional_camera.yaml`
- 原 Bidirectional 启动脚本：`./Wan21/scripts/training/run_stage0_bidirectional_camera.sh`
- 仓库已有的 Bidirectional -> AR 参考配置：`./Wan21/configs/ar_camera_tf.yaml`
- 仓库已有的 AR 启动脚本：`./Wan21/scripts/training/run_stage1_ar_camera.sh`
- 训练总入口：`./Wan21/wan_train.py`
- Bidirectional Trainer：`./Wan21/wan_trainer/camera_bidirectional_diffusion.py`
- AR Trainer 参考：`./Wan21/wan_trainer/camera_ar_diffusion.py`
- Bidirectional loss/model：`./Wan21/model/camera_bidirectional_diffusion.py`
- AR loss/model 参考：`./Wan21/model/camera_diffusion.py`
- Causal backbone：`./Wan21/wan/modules/causal_model.py`
- Wan wrapper：`./Wan21/wan_utils/wan_wrapper.py`
- Dataset 与相机转换：`./Wan21/wan_utils/dataset.py`

开始修改前，必须完整阅读 `proposal.md`，并沿仓库真实调用链完整检查 bidirectional 训练、当前 Stage 1 AR 转换、checkpoint loader、scheduler、PRoPE、Teacher-Forcing mask、Sequence Parallel 和推理代码。不要根据文件名猜测。

`proposal.md` 是目标数据流的设计依据；Flow 符号、timestep/sigma 映射、latent normalization、checkpoint key 和 PRoPE 数学实现必须以仓库实际代码为准。如两者存在冲突，保持 scheduler 与预训练权重的原约定，并在最终报告中明确说明。

## 二、这条路线与 AR 初始化路线的关键区别

必须从以下相对路径加载：

```text
./ckpts/Wan21/Action2V/bidirectional/model.pt
```

这个 checkpoint 已经学习过双向视频 denoising 和 PRoPE 相机控制，但还没有经过 `ar_diffusion_tf` 的因果 Teacher-Forcing 训练。因此目标不是继续使用双向 attention，而是：

```text
Bidirectional checkpoint 参数
        ↓ 加载到参数兼容的 CausalWanModel
单状态 causal Teacher-Forcing micro-turn 模型
```

请以仓库现有 `ar_camera_tf.yaml` 的 Stage 0 -> Stage 1 转换方式为权威参考：目标模型必须使用 `is_causal=True`、`causal=true`、`teacher_forcing=true`、`use_camera=true`。因果语义来自 causal backbone 和 attention mask，不是来自 checkpoint 文件名。

禁止出现以下错误：

- 为了容易加载而继续实例化双向 `WanModel`；
- 只把配置中的 `causal=true` 写上，却仍走双向 forward；
- 从 bidirectional checkpoint 加载失败后静默随机初始化 CausalWanModel；
- 丢失或重新零初始化已经训练过的 `prope_o`；
- 把 Bidirectional 和 AR checkpoint 混合加载；
- 将源 checkpoint 覆盖保存。

## 三、最终目标

把 Wan2.1 的生成粒度改造成：

```text
1 个 VAE latent state / micro-turn
```

77 帧视频经冻结 Wan2.1 VAE 得到：

```text
X: [B,20,16,60,104]
x0,x1,...,x19
```

构造 19 个状态转移：

```text
clean x0  + camera C0  -> noisy x1  + camera C1
clean x1  + camera C1  -> noisy x2  + camera C2
...
clean x18 + camera C18 -> noisy x19 + camera C19
```

`Ct=(viewmat_t,K_t)` 是已知相机几何条件。动作不作为离散 token；推理时动作字符只用于确定性计算 `C_(t+1)`。

## 四、checkpoint 加载和模型初始化

目标模型必须保留 Wan2.1 T2V 1.3B 的：

- Conv3D Patch Embedding；
- 30 层 Wan Transformer 参数；
- 普通时空 RoPE；
- PRoPE 与每层 `prope_o`；
- timestep embedding 与 AdaLN；
- 文本投影、Cross-Attention 与 FFN；
- Flow Head 与 Unpatchify。

加载 checkpoint 时必须逐项检查和报告：

- target model 的类名和 `is_causal` 实际值；
- missing keys；
- unexpected keys；
- shape mismatch；
- 30 层 `prope_o.weight/bias` 是否全部加载；
- Patch Embedding、Flow Head、首尾 Transformer block 是否加载；
- checkpoint 加载前后关键参数 norm/checksum，证明没有被后续初始化覆盖。

参考当前仓库 Stage 1 的兼容加载逻辑，尽量做到严格加载。如果 bidirectional 与 causal runtime 仅 attention mask 不同，参数应当大部分或全部兼容。若需要 `strict=False`，必须列出并解释每一个不兼容 key，不能静默忽略。

新日志和 checkpoint 使用独立目录，例如：

```text
./logs/micro_turn_from_bidirectional/
./ckpts/Wan21/Action2V/micro_turn_from_bidirectional/
```

不得覆盖：

```text
./ckpts/Wan21/Action2V/bidirectional/model.pt
./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```

## 五、数据组织

复用现有相机 LMDB 读取与坐标转换：

```text
clean_latent: [B,20,16,60,104]
viewmats:     [B,20,4,4]
Ks:           [B,20,3,3]
prompt:       B 个字符串
```

构造：

```text
clean_states  = X[:,0:19]   # x0...x18
target_states = X[:,1:20]   # x1...x19
current_cam   = C[:,0:19]
target_cam    = C[:,1:20]
```

必须保证第一轮是：

```text
x0,C0 -> noisy x1,C1
```

而不是全零 world input，也不是 `x0 -> noisy x0`。

禁止新建：

- action token/embedding；
- camera token/embedding；
- camera noise；
- camera prediction head；
- turn/type/special token；
- masked world token；
- camera classification/regression loss。

VAE Encoder/Decoder 和 UMT5 必须冻结。默认继续使用 LMDB 中已经编码好的 Wan latent，并复用原有 latent normalization、pose normalization 和 `build_viewmats_and_Ks()`。

## 六、Flow Matching

对 `x1...x19` 构造 19 个 noisy target。必须复用现有 scheduler：

```text
noise          [B,19,16,60,104]
noisy_targets  [B,19,16,60,104]
flow_targets   [B,19,16,60,104]
timestep       [B,19]
weights        与 target 对齐
```

不要重新实现一套可能符号相反的 Flow 公式。通过测试确认现有 `add_noise()`、`training_target()`、`training_weight()` 和 flow-to-x0 转换完全一致。

目标 micro-turn 粒度为 1，因此第一版使用：

```text
num_frame_per_block: 1
```

每个 target 可独立采样 timestep。clean span 沿用原 Teacher-Forcing clean timestep 约定；默认 clean augmentation 关闭。loss 只覆盖 19 个 target span。

## 七、Token packing 与 temporal id

每个 state 经过原 Conv3D Patch Embedding：

```text
[B,16,1,60,104]
 -> [B,1560,1536]
```

19 个 transition 共 38 个 span：

```text
38 × 1560 = 59280 tokens
```

逻辑结构：

```text
clean x0, noisy x1,
clean x1, noisy x2,
...
clean x18, noisy x19
```

可以选择 interleaved 或 contiguous 物理布局，但必须通过显式 span metadata 维护：

- clean/noisy role；
- transition id；
- physical state id；
- camera id；
- temporal id；
- token start/end；
- loss target index。

不得靠相同 shape 或“前后各一半”隐式猜测角色，除非相应布局有强断言。

普通 3D RoPE 使用物理时间：

```text
clean x0 -> temporal id 0
noisy x1 -> temporal id 1
clean x1 -> temporal id 1
...
noisy x19 -> temporal id 19
```

禁止错误编码成 38 个连续视频时刻。Bidirectional checkpoint 已学习原 Wan 视频时间位置，因此 temporal id 的兼容性尤其重要。

## 八、必须从零定义正确的 causal Teacher-Forcing mask

Bidirectional checkpoint 没有赋予目标 runtime 因果性；必须由新的 CausalWanModel 和 mask 严格建立：

对于 `clean x_t` query：

```text
允许：clean x0...clean xt
禁止：所有 noisy span和未来 clean span
```

对于 `noisy x_(t+1)` query：

```text
允许：clean x0...clean xt
允许：自己的 noisy span内部双向 attention
禁止：clean x_(t+1) 及未来 clean
禁止：其他 noisy span
```

必须验证：

```text
noisy x1  能读取 clean x0
noisy x2  能读取 clean x0,x1
noisy x19 能读取 clean x0...x18
```

不能直接使用普通三角 mask；它会受物理 packing 顺序影响，并可能让 clean history 读取 noisy target。也不能直接把原 4-latent AR mask 改一个常量后就认为完成，必须测试每种 query/key span。

请实现小尺寸 mask 单元测试或可视化审计，覆盖第 0、1、中间、第 18 个 transition，并断言没有 clean-target、future 或其他 noisy span 泄漏。

Sequence Parallel 下要特别检查 clean/noisy 拆分、padding、all-to-all/gather、camera/timestep 对齐。先保证 SP=1，再验证原训练所需 SP 配置。

## 九、PRoPE 对齐

每个 world token span 必须配准确相机：

```text
clean xt      -> C_t     -> viewmat_t,     K_t
noisy x(t+1)  -> C_(t+1) -> viewmat_(t+1), K_(t+1)
```

逻辑展开：

```text
token_viewmats: [B,59280,4,4]
token_Ks:       [B,59280,3,3]
```

可使用广播或按 span 展开，但传给每层 PRoPE 时必须与 Q/K/V 的 token 顺序完全一致。保留 bidirectional checkpoint 已训练好的普通 RoPE + PRoPE 并行结构及 `prope_o`。

`viewmat/K` 不是主序列 token，不是模型预测目标，也不应被加噪。修改目标相机 `C_(t+1)` 时，应只改变相应 noisy target span 的几何条件。

## 十、输出与 loss

Transformer 输出：

```text
[B,59280,1536]
```

按显式索引提取 noisy spans：

```text
[B,29640,1536]
 -> [B,19,1560,1536]
 -> 原 Flow Head + Unpatchify
 -> [B,19,16,60,104]
```

监督目标是 scheduler 返回的 19 个 flow target。只有最终 head 预测 flow；中间 Transformer 层输出 hidden state，不应逐层增加 world/camera prediction head。

任何可视化和 VAE decode 必须使用 Flow solver 或正确转换得到的 clean latent，不得直接解码 raw velocity。

## 十一、适合 Bidirectional 初始化的训练策略

由于本 checkpoint 尚未经过 AR Teacher-Forcing，采用保守的分阶段迁移：

### 阶段 A：结构与加载验证

- 冻结 VAE、UMT5；
- 冻结大部分 Transformer；
- 确认 causal mask、packed layout、camera alignment 和 Flow loss 正确；
- 训练最后若干 Transformer block、`prope_o`、LayerNorm/AdaLN 与 Flow Head；
- 使用很低学习率，避免立即破坏双向 checkpoint 的生成与相机能力。

### 阶段 B：单样本过拟合

使用 proposal 指定样本：

```text
./dataset/SmallestData/000000_right8a11/gen.mp4
```

如果训练读取 LMDB，则定位或使用该样本对应 LMDB。目标是证明 19 个 transition 的 loss 能下降，并经 Flow solver 后恢复 `x1...x19`。

### 阶段 C：全部 Transformer 微调

基础链路通过后，低学习率解冻全部 30 层 Transformer、PRoPE、timestep modulation、Cross-Attention 和 Flow Head。仍冻结 VAE 与 UMT5。

冻结/解冻必须通过明确参数白名单与断言实现，并打印 trainable/frozen 参数数目。不得误冻 `prope_o`，不得意外训练 VAE/UMT5。

第一版不要加入 scheduled sampling、camera loss、action loss 或额外蒸馏目标。先证明 Teacher-Forcing 单状态转移可以正确 overfit。

## 十二、推理路径

实现与训练一致的单状态递归推理：

```text
输入 prompt、x0、C0
action0 -> 确定性更新 C1 -> Flow solver 生成 x1
action1 -> 确定性更新 C2 -> Flow solver 生成 x2
...
```

每步只预测一个 `[B,1,16,60,104]` latent。动作不进入 Transformer。

必须保证：

- 推理实际调用 causal model，不是 bidirectional model；
- 当前/历史 clean state 使用自己的相机；
- noisy target 使用下一相机；
- temporal id 保持全局 `0...19`；
- 训练与推理可见性一致；
- 生成 state 经 solver 得到 clean latent 后才能进入下一步；
- 拼接 `x0...x19` 后再用冻结 VAE Decoder 输出 77 帧视频。

第一版可以不使用 KV cache，但必须明确历史窗口策略。不要为“实时”擅自把多步 Flow solver改成单步；少步生成需要后续独立蒸馏。

## 十三、代码组织与安全约束

- 为 Bidirectional 初始化路线新增独立、命名清晰的 trainer/model/config/launch/inference 文件。
- 不破坏原 Bidirectional、AR、causal ODE/CD/DMD 和旧 Full-Duplex 路径。
- 优先复用现有 scheduler、Dataset、PRoPE、wrapper、checkpoint loader 和训练设施。
- 如需改共享 backbone，必须增加默认关闭的显式 micro-turn 模式开关，并保持旧行为回归通过。
- 不复制整个 Wan backbone形成第二套难以维护的实现。
- 不修改 `Full_Duplex_Fix/proposal.md`。
- 不删除或覆盖用户已有修改，不执行破坏性 git 操作。
- 不启动长时间、无边界训练；只运行必要静态测试、单 batch forward/backward 和短 overfit smoke test。

## 十四、必须完成的验证

至少完成并报告：

1. Bidirectional checkpoint -> CausalWanModel 的加载审计；
2. 证明目标 forward 确实使用 causal Teacher-Forcing mask；
3. 30 层 `prope_o` 全部成功继承且未被覆盖初始化；
4. Dataset 的 `x0:18 -> x1:19`、`C0:18 -> C1:19` 对齐；
5. 38 span、59280 token、19 target 的运行时形状；
6. temporal id 与 camera id 审计；
7. mask 单元测试，无 GT/future/noisy-history 泄漏；
8. 单 batch forward 输出 `[B,19,16,60,104]`，无 NaN/Inf；
9. 单 batch backward，关键参数有有限非零梯度，VAE/UMT5 无梯度；
10. scheduler 的 noisy/target/x0 conversion 方向测试；
11. 短单样本 overfit smoke test，记录 loss 下降情况；
12. solver 后 clean latent 的少量 VAE decode 检查；
13. 原 Bidirectional 和原 AR 模式的最小回归检查；
14. 对比初始化时同一输入在 bidirectional runtime 和新 causal runtime 下的差异，并解释差异来自 mask，而不是权重加载失败。

所有 GPU 命令使用：

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n minwm <command>
```

## 十五、最终交付格式

最终报告必须包含：

- 新增和修改文件；
- 从 Bidirectional checkpoint 到 CausalWanModel 的真实加载过程；
- missing/unexpected keys 及逐项解释；
- 从 LMDB 到 Flow loss 的调用链；
- 关键张量真实形状；
- causal mask 和测试结果；
- PRoPE 与 camera span 对齐；
- 训练/冻结参数清单；
- 执行命令及测试结果；
- 新配置、启动脚本、推理脚本、日志和 checkpoint 输出相对路径；
- 未执行的长训练、已知风险以及与 AR checkpoint 初始化路线相比的预期差异。

请现在开始：先完整审计 proposal、Bidirectional 训练代码和现有 Stage 1 AR 转换，再给出简短实施计划，然后直接完成代码、测试和短 smoke run。不要停留在概念讨论阶段。
