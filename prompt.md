下面是整理后的严格版 Prompt，可直接交给 Codex、Claude Code 或其他编码 Agent。它只要求实现、调试和验证微调代码，不处理 VAE 视频恢复。

---

# minWM Full-Duplex Micro-Turn 微调任务

## 0. 角色与总目标

你是一名负责深度学习系统、视频生成、Diffusion Transformer、Flow Matching 和世界模型训练的资深科研工程师。

你需要进入当前运行环境(conda activate /hyperai/home/conda_envs/minwm)，理解并修改 `/hyperai/home/minwm` 代码仓库，为以下基础模型重新实现一套 Full-Duplex、time-aligned micro-turn 微调方案：

```text
Wan2.1
└── Wan21
    └── Action2V
        └── ar_diffusion_tf
```
**所有微调的代码放在@/hyperai/home/minwm/full_duplex 目录下**
【最终目标】

在不重新预训练 Wan2.1 的前提下，成功加载已有 `ar_diffusion_tf` checkpoint（@/hyperai/home/minwm/ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt），并使用新的流式 Transformer 输入组织与训练方法进行微调，使模型能够在 latent space 中自回归预测：

1. 下一时间片的 world state；
2. 下一时间片的 camera state。

首先使用最小数据集(/hyperai/home/minwm/dataset/SmallestData/000000_right8a11原版视频，切分视频，19个动作/hyperai/home/minwm/dataset/SmallestData/script_test_split/actions)过拟合，验证模型预测结果能够逐渐接近 ground truth。

【本阶段范围】

本阶段只完成：

- 数据预处理与缓存；
- checkpoint 加载；
- Full-Duplex token 序列构造；
- attention mask 和 prediction mask；
- 文本 cross-attention；
- world state 与 camera state 预测；
- Flow Matching 去噪；
- rollout 式微时间片训练；
- loss 计算；
- checkpoint 保存；
- 最小数据集过拟合验证。

【禁止】

- ❌ 暂时不要实现最终 RGB 视频恢复流程。
- ❌ 暂时不要把 VAE Decoder 纳入训练闭环。
- ❌ 不要重写整个 minWM 项目。
- ❌ 不要在未检查现有代码的情况下凭经验猜测 tensor shape、checkpoint key 或数据格式。
- ❌ 不要仅输出设计文档或伪代码；必须实际编写并运行代码。
- ❌ 不要为了让程序“看起来能运行”而使用随机 tensor 代替真实数据。
- ❌ 不要静默跳过 checkpoint 中无法加载的参数。

---

# 1. 仓库与数据范围

## 1.1 目标仓库

优先检查：

```text
/minwm
```

如果当前目录不同，先定位仓库实际路径。

【必须阅读】

至少检查以下内容及其实际调用关系：

```text
README.md
training_wan.md

Wan21/configs/ar_camera_tf.yaml
Wan21/model/
Wan21/pipeline/
Wan21/wan/modules/model.py
Wan21/wan/modules/causal_model.py
Wan21/wan/modules/prope.py
Wan21/wan/modules/vae.py
Wan21/wan_utils/wan_wrapper.py
Wan21/wan_utils/dataset.py
Wan21/wan_utils/camera_trajectory.py
Wan21/wan_trainer/ar_diffusion.py
Wan21/wan_trainer/camera_ar_diffusion.py
Wan21/wan_inference.py
```

如果真实代码路径发生变化，应使用代码搜索找到对应实现。

## 1.2 基础 checkpoint

目标 checkpoint 为：

```text
Wan21/Action2V/ar_diffusion_tf
```

【必须】

在修改模型前，先确认：

- checkpoint 的绝对路径；
- checkpoint 文件结构；
- 顶层 key；
- 参数命名规则；
- 模型参数量；
- dtype；
- checkpoint 对应的 Wan2.1 模型规格；
- checkpoint 是否包含 generator、EMA、optimizer 或 scheduler；
- 原始 checkpoint 与当前代码的参数匹配情况。

## 1.3 最小数据集

最小数据集位于：

```text
/minwm/dataset/SmallestData
```

目标样本目录：

```text
/minwm/dataset/SmallestData/000000_right8a11
```
表示right有8个动作，a有11个动作


动作切分目录：

```text
/mnt/onelab0/sub5-v2u2/cyh_area/data/0data/minWM/dataset/SmallestData/split_4f_actions/actions
```

数据特征：

- 原视频约 77 个 RGB frames；
- Wan2.1 VAE 编码后约为 20 个 latent states；
- 第一个 latent state 对应初始帧；
- 后续每个 latent state 大致对应 4 个 RGB frames 的变化；
- 当前样本已经被切分成 19 个动作；
- 19 个动作需要依次进入 19 个 micro time turns。

⚠️【必须验证】

不要直接相信以上数字。必须读取真实数据并打印：

- RGB 视频帧数；
- 动作数量；
- 每个动作对应的帧范围；
- VAE latent 数量；
- latent 与动作之间的对齐关系；
- 是否存在 off-by-one；
- 第一个初始 state 是否属于动作 0；
- 19 个 action 如何映射到 20 个 latent states。

如数据与描述不一致，必须以实际数据为准，并在日志中明确报告差异。

---

# 2. 开始编码前的强制检查

在编写新模型前，先生成一份简短的代码审计结果，但不要停留在审计阶段。

【证据要求】

每项结论必须附带：

```text
文件路径
类名
函数名
相关代码行
输入 shape
输出 shape
dtype
```

需要确认：

1. Wan2.1 VAE 如何编码视频；
2. world state latent 的准确 shape；
3. VAE 的 channel mean/std 如何应用；
4. Transformer 的 patch embedding；
5. UMT5-XXL 的输出 shape；
6. 原模型如何注入文本；
7. RoPE 和 PRoPE 的具体调用位置；
8. camera 的数据格式；
9. camera 是 c2w 还是 w2c；
10. action 的原始格式；
11. 原训练代码的 Flow Matching target；
12. 原始 checkpoint 的完整加载逻辑；
13. 原模型的 causal mask 和 teacher forcing mask；
14. 原推理代码如何进行 blockwise generation；
15. 哪些参数应该冻结，哪些参数应该训练。

【禁止】

如果某个问题不能通过代码确认，必须标记为：

```text
【尚未确认】
```

不能把推断写成事实。

---

# 3. Full-Duplex Micro-Turn 序列协议

## 3.1 总体时间结构

所有 micro time turns 构成一条连续的流式 token 序列：

```text
micro_turn_0
→ micro_turn_1
→ micro_turn_2
→ …
→ micro_turn_N
```

每个 micro turn 同时包含：

```text
Input Stream
+
Output Stream
+
TIME_INDEX_TOKEN
```

不要把它理解成传统的：

```text
完整输入 → 完整输出 → 下一个输入
```

而应理解成按时间对齐的流式片段：

```text
time 0: input_stream_0 + output_stream_0
time 1: input_stream_1 + output_stream_1
time 2: input_stream_2 + output_stream_2
...
```

## 3.2 单个 micro turn 的严格顺序

每个时间片必须按照以下逻辑顺序构造：

```text
INPUT_STREAM_START_TOKEN

world_state_tokens
MODALITY_END_TOKEN

camera_tokens
MODALITY_END_TOKEN

action_tokens
MODALITY_END_TOKEN

noise_state_tokens
NOISE_END_TOKEN

INPUT_STREAM_END_TOKEN

OUTPUT_STREAM_START_TOKEN

ground_truth_world_state_tokens
MODALITY_END_TOKEN

ground_truth_camera_tokens
MODALITY_END_TOKEN

OUTPUT_STREAM_END_TOKEN

TIME_INDEX_TOKEN[t]
```

其中：

- `INPUT_STREAM_START_TOKEN`：输入流开始；
- `INPUT_STREAM_END_TOKEN`：输入流结束；
- `OUTPUT_STREAM_START_TOKEN`：输出流开始；
- `OUTPUT_STREAM_END_TOKEN`：输出流结束；
- `MODALITY_END_TOKEN`：当前模态结束；
- `NOISE_END_TOKEN`：噪声 state 结束；
- `TIME_INDEX_TOKEN[t]`：表示 micro turn 的时间索引，并表示当前时间片结束。

⚠️【重点】

如果实现中没有必要同时使用 `*_START_TOKEN` 和 `*_END_TOKEN`，不得擅自删除。应先保持上述协议，完成最小可运行版本后再报告冗余性。

## 3.3 特殊 token 的实现

【必须】

为每一种特殊 token 建立明确、独立、可训练的 embedding。

不得把所有结束 token 共享为同一个 embedding，除非经过消融实验确认。

需要至少实现：

```text
INPUT_STREAM_START
INPUT_STREAM_END
OUTPUT_STREAM_START
OUTPUT_STREAM_END
MODALITY_END
NOISE_END
TIME_INDEX_0 ... TIME_INDEX_N
NULL_WORLD_STATE
```

如果采用单独的 `NULL_WORLD_STATE` embedding，就不再把它解释成全零 latent；如果严格采用全零 state，则仍需使用特殊 token 标记其语义。

【必须记录】

- 特殊 token 总数；
- 每个 token 的整数 ID；
- embedding shape；
- embedding 是否参与训练；
- checkpoint 中是否存在对应参数；
- 新增参数的初始化方法。

---

# 4. 各模态的编码规范

## 4.1 World state

world state 使用 Wan2.1 VAE 编码后的标准化 latent：

```text
[B, F, 16, H_latent, W_latent]
```

每个 micro turn 只使用与该时间片对应的 state 或 state block。

World state 必须经过现有或兼容的 patch embedding，转换为：

```text
[B, N_world_tokens, D_model]
```

【形状】

必须在首次运行时打印：

```text
pixel video
VAE latent
micro-turn latent slice
patchified world tokens
flattened world token sequence
```

## 4.2 Micro turn 0

`micro_turn_0` 没有上一时刻的 world state。

使用：

```text
全零 world state
```

作为输入 world state，其 shape 必须与正常 state 完全一致。

同时使用 `NULL_WORLD_STATE` 或明确的模态标记，使模型能够区分：

```text
真实的全黑/全零状态
```

和：

```text
时间起点不存在历史状态
```

## 4.3 后续 world state

训练过程中：

```text
micro_turn_t 的输入 world state
=
micro_turn_(t-1) 的预测 world state
```

【核心要求】

不得在 `t > 0` 时直接使用 ground-truth world state 替代上一时间片预测，以免退化成 teacher forcing。

必须实现真实 rollout：

```text
pred_state_0
→ input_state_1
→ pred_state_1
→ input_state_2
→ …
```

除非专门实现可配置的 teacher-forcing warmup，否则默认训练路径必须使用预测状态。

## 4.4 Camera

必须先确认数据中的 camera 表示。

候选形式可能包括：

```text
w2c 4×4
c2w 4×4
translation + quaternion
SE(3) relative pose
intrinsics K
PRoPE viewmats + Ks
```

【重点】

新的 Full-Duplex 序列要求 camera 具有显式 camera token，因此不能只把 camera 通过 PRoPE 隐式注入。

应实现一个 camera encoder，把每个时间片的 camera 参数映射为：

```text
[B, N_camera_tokens, D_model]
```

但可以同时保留 PRoPE 作为额外几何条件。

也就是说，需要明确区分：

```text
显式 camera token：供序列建模和 camera prediction 使用
PRoPE：供视频 token 的几何注意力使用
```

不得混为同一机制。

## 4.5 Action

所有动作已切分为 11 个 action steps。

需要建立 action tokenizer 或 embedding table：

```text
action_id
→ action_embedding
→ [B, N_action_tokens, D_model]
```

【必须确认】

- action vocabulary；
- action 总类别数；
- action ID 与原始动作名称的映射；
- 是否包含 no-op；
- 是否包含组合动作；
- action 与 camera trajectory 的关系；
- 每个 micro turn 使用一个还是多个 action token。

⚠️ 不得继续沿用“action 只通过 camera trajectory 进入 PRoPE”的旧实现。

新结构必须具有显式 `action_tokens`。

## 4.6 Noise state

每个 micro turn 输入一个固定的 noise state：

```text
ε_t
```

它必须与目标 world state 的 latent shape 一致：

```text
[B, C=16, F_turn, H_latent, W_latent]
```

然后通过与 world state 兼容的 patch embedding 转换为 noise tokens。

【重点】

同一个 micro turn 的 10 步去噪过程中，初始噪声必须保持不变。

不得在每个去噪 step 重新采样噪声。

必须支持固定 seed，以保证：

```text
相同样本 + 相同 seed
→ 完全相同的初始 noise
```

## 4.7 文本

使用现有 Wan2.1 冻结的 UMT5-XXL Encoder：

```text
prompt
→ tokenizer
→ UMT5-XXL
→ prompt_embeds
```

文本不直接拼入 Full-Duplex token 序列。

文本通过 cross-attention 注入每一个 micro turn。

【必须】

- 默认冻结 T5；
- padding token embedding 必须清零或使用 attention mask；
- 每个 micro turn 都能访问相同的文本条件；
- 明确 cross-attention 的 Query、Key 和 Value；
- 记录文本 embedding shape；
- 不得重复运行相同 prompt 的 T5 编码；
- 文本 embedding 必须支持磁盘缓存。

---

# 5. Transformer 与位置编码

## 5.1 基础权重加载

必须优先复用：

```text
Wan21/Action2V/ar_diffusion_tf checkpoint
```

新架构应尽可能复用：

- patch embedding；
- Transformer blocks；
- self-attention；
- cross-attention；
- FFN；
- timestep embedding；
- RoPE；
- PRoPE；
-输出 head 中兼容的部分。

新增部分可以包括：

- 特殊 token embedding；
- camera encoder；
- camera prediction head；
- action embedding；
- stream/type embedding；
- time-index embedding；
- Full-Duplex mask；
- 必要的输入投影和输出投影。

【禁止】

不得使用：

```python
load_state_dict(..., strict=False)
```

然后忽略 missing/unexpected keys。

如果为了新增模块必须使用 `strict=False`，必须：

1. 打印全部 missing keys；
2. 打印全部 unexpected keys；
3. 明确哪些 key 是预期新增；
4. 对其他不匹配项直接报错；
5. 输出实际加载参数比例；
6. 验证基础模型的重要权重已经加载。

## 5.2 RoPE

保留 Wan2.1 原有 3D RoPE，用于编码视频 token 的：

```text
time
height
width
```

## 5.3 PRoPE

保留 PRoPE，用于把：

```text
viewmats
+
Ks
```

注入视频 token 的 self-attention 几何关系。

【区分】

```text
RoPE：规则视频网格位置
PRoPE：相机投影几何
TIME_INDEX_TOKEN：micro-turn 序列位置
```

三者不得互相替代。

## 5.4 Micro-turn 位置

每个 micro turn 末尾加入：

```text
TIME_INDEX_TOKEN[t]
```

此外，所有属于同一 micro turn 的 token 应能获得相同的 turn index embedding 或可识别的 turn position。

需要明确说明：

- `TIME_INDEX_TOKEN[t]` 是否为独立词表；
- 是否额外加入 turn-position embedding；
- 最大支持多少 micro turns；
- 超过最大长度时如何处理。

---

# 6. Attention Mask 与预测 Mask

## 6.1 特殊 token 不参与掩码预测

以下特殊 token 必须始终可见，不作为 latent prediction target：

```text
INPUT_STREAM_START_TOKEN
INPUT_STREAM_END_TOKEN
OUTPUT_STREAM_START_TOKEN
OUTPUT_STREAM_END_TOKEN
MODALITY_END_TOKEN
NOISE_END_TOKEN
TIME_INDEX_TOKEN[t]
NULL_WORLD_STATE
```

## 6.2 输出流内容掩码

训练时，每个 micro turn 的以下内容需要作为预测区域：

```text
ground_truth_world_state_tokens
ground_truth_camera_tokens
```

它们在模型输入中必须被 masked representation 或对应占位 token 替代。

特殊边界 token保持可见，使模型能够确定：

- world state 输出区域；
- camera 输出区域；
- 输出流边界；
- 当前 micro turn 边界。

## 6.3 Causal 可见性

默认可见性规则：

```text
当前 turn 的输出预测
可以读取：
- 当前 turn 的全部输入流；
- 当前 turn 的可见特殊 token；
- 当前 turn 的文本 cross-attention；
- 所有历史 turn 的输入和预测结果。

不能读取：
- 当前 turn 的 ground-truth 输出内容；
- 未来 turn 的任何内容；
- 未来 camera；
- 未来 action；
- 未来 noise；
- 未来 world state。
```

【必须】

单独编写 mask 单元测试，以小尺寸假 token 验证每一类 token 的可见关系。

测试必须输出可读的二维 mask 图或矩阵，并通过断言检查：

```text
历史可见
当前输入可见
当前 GT 输出不可见
未来全部不可见
特殊边界 token 可见
```
整体训练框架：如下图：
![框架](/hyperai/home/minwm/image.png)
---

# 7. 训练目标

## 7.1 World state Flow Matching

每个 micro turn 中：

```text
输入：
历史预测 world state
+ 当前 camera
+ 当前 action
+ 当前固定 noise
+ 文本条件

目标：
当前 ground-truth world state
```

使用 Flow Matching 训练 DiT 预测：

```text
velocity / flow
```

必须沿用与原 Wan2.1 checkpoint 兼容的：

- scheduler；
- timestep 定义；
- noise interpolation；
- training target；
- latent normalization。

不得自行更改 flow 的符号或定义。

## 7.2 十步去噪

每个 micro turn 进行最多 10 步去噪：

```text
固定初始 noise
→ denoise step 1
→ ...
→ denoise step 10
→ predicted latent state
```

【必须明确】

实现前检查原 scheduler 的 `step()` 语义，避免混淆：

```text
flow prediction
x0 prediction
next latent
```

十步去噪必须是可配置项：

```yaml
num_denoising_steps: 10
```

## 7.3 World state loss

至少实现以下 loss：

```text
L_flow  = MSE(predicted_flow, target_flow)
L_state = MSE(predicted_final_latent, ground_truth_latent)
```

默认总 loss：

```text
L_world =
    λ_flow  × L_flow
  + λ_state × L_state
```

如果十步采样过程保持可微，则允许 `L_state` 通过全部去噪步骤反向传播。

如果由于 scheduler 实现导致不可微，必须报告原因，不能静默 `detach()`。

## 7.4 Camera loss

模型需要从输出流对应位置预测 camera。

至少实现：

```text
L_camera_translation
L_camera_rotation
```

如果直接预测矩阵，还需要考虑旋转矩阵约束。优先使用稳定表示，例如：

```text
translation + 6D rotation
```

或仓库中已经验证的相机表示。

【禁止】

不得直接对完整 `4×4` 矩阵所有元素做无解释的 MSE，除非这是项目现有且明确验证过的表示。

若 camera 包含 intrinsics，应分别记录：

```text
L_camera_intrinsics
```

总 camera loss：

```text
L_camera =
    λ_translation × L_translation
  + λ_rotation    × L_rotation
  + λ_intrinsics  × L_intrinsics
```

## 7.5 总损失

```text
L_total =
    λ_flow   × L_flow
  + λ_state  × L_state
  + λ_camera × L_camera
```

所有 loss 权重必须进入配置文件，不得硬编码。

日志中分别记录：

```text
total_loss
flow_loss
state_loss
camera_loss
translation_loss
rotation_loss
intrinsics_loss
```

---

# 8. 跨 Micro-Turn 反向传播

【核心要求】

训练序列必须是：

```text
pred_state_t
→ input_state_(t+1)
```

并且默认不在 micro-turn 边界执行：

```python
detach()
```

最终 loss 对前面时间片的预测保留梯度，实现跨 micro-turn 的 BPTT。

需要显式检查：

```text
turn 0 的预测
是否能收到 turn 1、turn 2 ... loss 的梯度
```

至少添加一次调试断言或 gradient hook，证明早期 turn 的 tensor/参数存在非零梯度。

⚠️ 当前机器为 H100，micro turns 数量较少，初始版本不要因为猜测显存不足而提前截断梯度。

如果实际发生 OOM，按以下顺序优化：

1. bf16；
2. gradient checkpointing；
3. sequence parallel；
4. 减少 batch size；
5. 降低 micro-turn 数；
6. 最后才考虑 truncated BPTT。

不得未经测试直接使用 truncated BPTT。

---

# 9. 数据预编码与 Metadata 缓存

## 9.1 预编码内容

为避免每次训练重复运行 VAE 和 T5，建立可复用的预编码缓存。

至少缓存：

```text
sample_id
source_video_path
prompt
prompt_embedding
world_state_latents
latent_mean/std 版本
camera sequence
camera representation
action IDs
action names
action-to-state alignment
micro-turn boundaries
original frame count
latent frame count
resolution
fps
VAE checkpoint identity
T5 checkpoint identity
preprocessing config hash
```

## 9.2 缓存格式

选择适合 tensor 随机读取的格式，例如：

```text
safetensors
pt
npz
LMDB
```

同时保存一个人类可读的：

```text
metadata.json
```

或：

```text
metadata.jsonl
```

【必须】

缓存应具备版本校验：

```text
如果 VAE、T5、分辨率、动作切分或预处理配置发生变化
→ 自动判定缓存失效并重新编码
```

不得因为文件存在就盲目复用旧缓存。

## 9.3 预编码验证

预编码后必须检查：

- 是否存在 NaN/Inf；
- latent mean/std；
- latent min/max；
- 每个 micro turn 的 tensor shape；
- camera 是否连续；
- action 数是否一致；
- prompt embedding 是否一致；
- 重复加载缓存结果是否 bitwise 相同或在允许误差内相同。

---

# 10. 训练与 Checkpoint 管理

## 10.1 同步训练控制脚本

编写一个同步训练脚本，负责：

1. 启动训练；
2. 监控训练进程；
3. 记录 loss；
4. 定期保存 checkpoint；
5. 保存最低 loss checkpoint；
6. 支持断点续训；
7. 训练异常时保留日志；
8. 训练完成后输出最佳结果摘要。

不要把训练放入无法监控的后台进程后立即结束任务。

## 10.2 Checkpoint 命名

checkpoint 名称必须包含至少：

```text
step
total_loss
state_loss
camera_loss
```

例如：

```text
step_000100_total_0.012345_state_0.009876_camera_0.002469.pt
```

同时维护：

```text
best.pt
latest.pt
```

其中：

- `best.pt`：验证或训练总 loss 最低；
- `latest.pt`：最近一次有效保存。

如果最小数据集只含一个样本，应明确注明 best loss 属于训练集过拟合指标。

## 10.3 Checkpoint 内容

每个 checkpoint 至少包含：

```text
model
optimizer
lr_scheduler
global_step
epoch
best_loss
loss_history
random seed
PyTorch RNG state
CUDA RNG state
model config
training config
token vocabulary
token IDs
camera representation
action vocabulary
preprocessing metadata hash
base checkpoint identity
```

【验证】

保存后立刻执行一次 reload test：

1. 新建模型；
2. 加载 checkpoint；
3. 读取同一个 batch；
4. 重新前向；
5. 比较保存前后的输出；
6. 确认误差符合预期。

---

# 11. 分阶段调试和训练计划

必须严格按以下顺序执行，不要直接开始长时间训练。

## Task 1：环境与依赖检查

- [ ] 定位仓库、数据和 checkpoint；
- [ ] 检查 GPU；
- [ ] 确认 H100 可用；
- [ ] 记录 CUDA、PyTorch 和关键依赖版本；
- [ ] 确认 bf16 支持；
- [ ] 运行原项目最小 import test。

## Task 2：原 checkpoint 加载

- [ ] 实例化原 `ar_diffusion_tf` 模型；
- [ ] 加载 checkpoint；
- [ ] 检查 missing/unexpected keys；
- [ ] 输出参数加载比例；
- [ ] 执行原模型最小前向；
- [ ] 确认输出无 NaN/Inf。

【验收标准】

基础 checkpoint 必须成功加载并完成至少一次真实前向。

## Task 3：最小数据预编码

- [ ] 读取 `000000_right8a11`；
- [ ] 编码视频 latent；
- [ ] 编码文本；
- [ ] 读取 camera；
- [ ] 读取 11 个动作；
- [ ] 建立 action/state 对齐；
- [ ] 保存 tensor 缓存和 metadata；
- [ ] 重新加载并验证。

## Task 4：特殊 token 与序列构造

- [ ] 实现特殊 token vocabulary；
- [ ] 实现 world/camera/action/noise encoder；
- [ ] 实现单个 micro turn 序列；
- [ ] 实现完整 11-turn 流式序列；
- [ ] 输出每个 token span 的起止 index；
- [ ] 检查所有边界 token；
- [ ] 检查 sequence length。

## Task 5：Attention Mask

- [ ] 实现 causal mask；
- [ ] 实现 output content mask；
- [ ] 特殊 token 保持可见；
- [ ] 当前 GT 输出不可见；
- [ ] 未来 turn 不可见；
- [ ] 编写 mask 单元测试；
- [ ] 输出小型 mask 可视化。

## Task 6：模型前向

- [ ] 加载基础 Transformer 权重；
- [ ] 接入显式 camera tokens；
- [ ] 接入 action tokens；
- [ ] 接入 stream/type/time embeddings；
- [ ] 保留 RoPE；
- [ ] 保留 PRoPE；
- [ ] 接入每个 turn 的文本 cross-attention；
- [ ] 输出 world flow；
- [ ] 输出 final world state；
- [ ] 输出 camera；
- [ ] 验证所有 shape。

## Task 7：单个 Micro-Turn 过拟合

先只训练一个 micro turn。

- [ ] 固定一个样本；
- [ ] 固定一个 action；
- [ ] 固定 noise seed；
- [ ] 运行 forward；
- [ ] 运行 backward；
- [ ] 棚查非零梯度；
- [ ] 训练 10 步；
- [ ] 训练 100 步；
- [ ] 检查 loss 是否下降。

【验收标准】

在同一个样本和固定噪声下，100 步后的 loss 必须明显低于初始 loss。

## Task 8：完整 11-Turn Rollout

- [ ] turn 0 使用 zero/null state；
- [ ] turn 1 使用 turn 0 预测；
- [ ] 依次 rollout 至最后一个 turn；
- [ ] 保留跨 turn 计算图；
- [ ] 计算所有 turn 的 loss；
- [ ] 检查 early-turn gradient；
- [ ] 训练 100 步；
- [ ] 检查总 loss 和逐 turn loss。

## Task 9：逐步增加训练步数

只有当前阶段通过后，才能增加训练步数：

```text
1 step
→ 10 steps
→ 100 steps
→ 500 steps
→ 1000 steps
→ 根据 loss 决定是否继续
```

每个阶段都记录：

```text
初始 loss
最终 loss
最低 loss
各子 loss
梯度范数
学习率
显存峰值
每步耗时
预测 latent 与 GT latent 的误差
```

不得在 loss 不下降时盲目增加训练步数。

## Task 10：最终过拟合验证

需要证明：

```text
predicted latent state ≈ ground-truth latent state
predicted camera ≈ ground-truth camera
```

至少输出：

- 每个 turn 的 state MSE；
- 每个 turn 的 camera error；
- 整体 latent cosine similarity；
- latent mean/std 对比；
- 最差 turn；
- 最好 turn；
- 最佳 checkpoint 路径。

---

# 12. Debug 要求

出现错误时必须定位根因，不要用宽泛的 try/except 隐藏错误。

重点检查：

- checkpoint key 不匹配；
- latent 维度排列错误；
- VAE 是否重复标准化；
- action/state 对齐错误；
- camera 坐标系错误；
- c2w/w2c 混淆；
- quaternion 顺序错误；
- token span 计算错误；
- mask 泄漏 ground truth；
- future turn 信息泄漏；
- noise 在 10 步中被重新采样；
- scheduler step 方向错误；
- flow target 符号错误；
- rollout 中意外 `detach()`；
- 预测 state 没有真正传给下一 turn；
- 特殊 token 被错误加入 prediction loss；
- camera loss 数值范围远大于 state loss；
- T5 被意外训练；
- VAE 被意外训练；
- 缓存与当前配置不匹配；
- loss 下降只是因为读取了 ground truth。

【必须】

开启至少以下数值检查：

```text
torch.isfinite(loss)
torch.isfinite(model_output)
torch.isfinite(grad)
gradient norm
parameter norm
latent min/max/mean/std
```

---

# 13. 配置要求

所有关键参数必须配置化，至少包括：

```yaml
base_checkpoint:
vae_checkpoint:
t5_checkpoint:
dataset_path:
cache_path:
output_dir:

num_micro_turns: 11
num_denoising_steps: 10
num_frame_per_turn:
max_time_index:

learning_rate:
weight_decay:
batch_size:
gradient_accumulation_steps:
max_grad_norm:
mixed_precision: bf16
gradient_checkpointing:
seed:

lambda_flow:
lambda_state:
lambda_camera:
lambda_translation:
lambda_rotation:
lambda_intrinsics:

save_every:
log_every:
max_steps:
teacher_forcing_ratio:
detach_between_turns: false
```

【禁止】

关键路径、loss 权重、训练步数和 token 数不得散落在代码中硬编码。

---

# 14. 每次进展汇报格式

每完成一个阶段，使用以下格式汇报：

```text
【当前任务】
Task N：任务名称

【已完成】
- 实际完成的内容

【修改文件】
- 文件路径
- 新增或修改的类/函数

【关键 Shape】
- tensor_name: shape, dtype, semantic meaning

【运行命令】
- 完整命令

【验证结果】
- 测试是否通过
- loss
- gradient norm
- 显存
- 耗时

【发现的问题】
- 已确认问题
- 尚未确认问题

【下一步】
- 下一项具体任务
```

不要只说“代码已经完成”或“训练正常”。

---

# 15. 最终交付物

最终必须提供：

1. Full-Duplex 微调模型代码；
2. 特殊 token vocabulary；
3. micro-turn sequence builder；
4. attention/prediction mask；
5. camera encoder 和 prediction head；
6. action embedding；
7. world-state Flow Matching 训练代码；
8. 10-step differentiable denoising；
9. 11-turn rollout 训练逻辑；
10. 跨 turn BPTT；
11. 数据预编码和 metadata 缓存；
12. 同步训练控制脚本；
13. best/latest checkpoint 管理；
14. checkpoint reload test；
15. mask 单元测试；
16. shape 单元测试；
17. 最小数据集训练日志；
18. 100 步 loss 曲线或数值记录；
19. 最佳 checkpoint；
20. 最终过拟合评估报告。

---

# 16. 最终验收标准

只有同时满足以下条件，任务才算完成：

- [ ] 原 `ar_diffusion_tf` checkpoint 成功加载；
- [ ] 重要基础权重没有被漏载；
- [ ] 真实最小数据完成预编码；
- [ ] 缓存可重复加载；
- [ ] 11 个动作与 micro turns 对齐；
- [ ] Full-Duplex 序列顺序正确；
- [ ] 特殊 token 位置正确；
- [ ] mask 不泄漏当前 GT 输出或未来信息；
- [ ] 文本通过 T5 和 cross-attention 注入每个 turn；
- [ ] turn 0 使用 null/zero world state；
- [ ] 后续 turn 使用上一个预测 state；
- [ ] rollout 中没有意外 detach；
- [ ] 10-step 去噪使用固定初始 noise；
- [ ] state、flow 和 camera loss 均能反向传播；
- [ ] early turn 能收到后续 loss 的梯度；
- [ ] 单 turn 过拟合 loss 明显下降；
- [ ] 完整 11-turn 训练 100 步后 loss 有下降趋势；
- [ ] checkpoint 能保存并恢复；
- [ ] `best.pt` 对应最低 loss；
- [ ] 预测 latent 明显接近 ground-truth latent；
- [ ] 所有关键结论都有代码、shape 或运行日志证据；
- [ ] 没有用 VAE Decoder 的 RGB 结果掩盖 latent 预测问题。

---

## 最终执行指令

【立即开始】

请进入环境，从 `Task 1：环境与依赖检查` 开始执行，并持续推进到最小数据集 100 步训练验证。

【不要停在方案阶段】

完成审计后必须继续：

```text
编写代码
→ 运行测试
→ 修复错误
→ 加载真实 checkpoint
→ 编码真实最小数据
→ 运行训练
→ 观察 loss
→ 保存最佳 checkpoint
```

【遇到歧义时】

优先通过以下方式解决：

1. 阅读仓库代码；
2. 检查 checkpoint；
3. 检查真实数据；
4. 运行最小实验；
5. 对照 tensor shape。

只有当某个选择会实质改变模型定义，并且无法从代码、数据或 checkpoint 中确认时，才向我提出一个具体问题。

[def]: /hyperai/home/minwm/image.png