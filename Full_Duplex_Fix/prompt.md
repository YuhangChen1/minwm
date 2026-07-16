# minWM / Wan2.1 交错式 Full-Duplex 单状态 T2V 改造任务

## 0. 角色、仓库与执行原则

你是一名负责视频生成、Diffusion Transformer、Flow Matching、因果注意力、
RoPE/PRoPE、FSDP 和训练系统的资深科研工程师。你需要在当前真实环境中完成
代码审计、实现、测试、单样本过拟合、自主推理和结果报告，而不是只提供设计
建议或伪代码。

目标仓库：

```text
/workspace/yuhang/minwm
```

新实验目录：

```text
/workspace/yuhang/minwm/Full_Duplex_Fix
```

必须遵守以下原则：

1. 先阅读并理解真实源码，再实现；不要根据文件名猜测行为。
2. 所有新增训练入口、配置、缓存、测试、日志和报告优先放在
   `Full_Duplex_Fix/` 内。
3. 不得覆盖或删除旧的 `full_duplex/` 实验；它只可作为失败经验和数据审计参考。
4. 如果确实必须修改 `Wan21/` 共享代码，修改必须最小、可测试、向后兼容，且
   不能改变原有配置的默认行为。
5. 不下载模型。所需基础模型和 checkpoint 已经在 `ckpts/` 中。
6. 不得通过缩减 Transformer 层数、空间 token stride 或偷换数据来宣称最终
   方案成功。资源缩减只能作为明确标注的诊断实验。
7. 所有结论必须来自真实运行产物。没有运行的内容必须明确写为“未验证”。
8. 不得使用 `strict=False` 静默忽略 checkpoint 不匹配。
9. 遇到错误时保留原始异常、有效配置和复现命令，不得用伪造指标代替结果。
10. 任务应持续到代码、测试、最小过拟合、自主采样和报告形成完整闭环。

开始前至少阅读：

```text
README.md
training_wan.md
Full_Duplex_Fix/proposal.md
prompt.md
full_duplex/AUDIT.md
full_duplex/FINAL_REPORT.md
full_duplex/NEW_VS_OLD_TRAINING_REPORT.md

Wan21/configs/default_config.yaml
Wan21/configs/ar_camera_tf.yaml
Wan21/model/base.py
Wan21/model/diffusion.py
Wan21/model/camera_diffusion.py
Wan21/wan_trainer/camera_ar_diffusion.py
Wan21/wan_utils/wan_wrapper.py
Wan21/wan_utils/scheduler.py
Wan21/wan_utils/dataset.py
Wan21/wan/modules/model.py
Wan21/wan/modules/causal_model.py
Wan21/wan/modules/prope.py
Wan21/pipeline/causal_diffusion_inference.py
Wan21/wan_inference.py
```

---

## 1. 最终研究目标

基于已有 Wan2.1 Action2V Teacher-Forcing AR checkpoint：

```text
ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```

实现一个新的、单 latent state 粒度的、交错式 Teacher-Forcing T2V 模型。

模型任务是：

```text
文本 prompt
+ 预先给定的相机轨迹 C0...C19
+ 每个状态的初始高斯噪声
-> 自回归生成 x0...x19
-> 冻结 Wan VAE 解码为 77 帧视频
```

这里没有用户提供的初始图像或 `x0`。`x0` 必须由模型使用文本、初始相机和
噪声生成。

准确的任务名称是：

> 文本与相机轨迹条件的单状态自回归视频生成。

如果推理时提供 `C0...C19`，它不是严格意义上“只有文本”的 T2V，而是
camera-conditioned T2V。报告中必须使用准确表述。

### 1.1 本阶段明确不做的内容

第一版禁止加入：

```text
action token
camera token
camera prediction head
camera loss
special stream token
clean/noisy role embedding
额外 world residual head
time-space memorization prior
LoRA
直接 state MSE 训练项
训练图内的多步 differentiable solver
scheduled sampling
autoregressive rollout loss
RGB/VAE perceptual loss
空间 token stride 下采样
只执行部分 Transformer blocks
```

动作标签只允许在数据预处理阶段通过仓库既有相机轨迹函数转换为 camera
trajectory；动作名称或 ID 不进入 Transformer。

这意味着当前模型只显式表达相机/视角控制。相机矩阵不能表达人物动作、物体
交互等一般 action，最终报告不得把它宣传为通用 action model。

---

## 2. 已确认的真实数据与 checkpoint

### 2.1 单条过拟合数据

视频：

```text
dataset/SmallestData/000000_right8a11/gen.mp4
```

文本与轨迹元数据：

```text
dataset/SmallestData/smllest_input.json
```

动作/帧对齐清单：

```text
dataset/SmallestData/split_4f_actions/manifest.json
```

真实数据事实：

```text
RGB frames: 77
FPS: 24
resolution: 832 x 480
frame 0: initial physical frame
frames 1..76: 19 groups, each group has 4 RGB frames
actions: right x 8, then a x 11
```

动作对齐为：

```text
action 0  -> RGB frames 1..4
...
action 18 -> RGB frames 73..76
```

提示词必须从 `smllest_input.json` 的 `caption` 字段读取，并在缓存 metadata
中保存原始字符串，不要在多个文件中复制出不同版本。

### 2.2 Wan 基础组件

```text
Wan base directory:
ckpts/Wan2.1-T2V-1.3B

Wan VAE:
ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth

UMT5 encoder:
ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth

UMT5 tokenizer:
ckpts/Wan2.1-T2V-1.3B/google/umt5-xxl

primary generator checkpoint:
ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt

secondary comparison initialization only:
ckpts/Wan21/Action2V/bidirectional/model.pt
```

第一版必须优先使用 `ar_diffusion_tf`。Bidirectional checkpoint 只作为后续
初始化消融，不得与主实验混合。

当前本地 AR checkpoint 已核验的身份为：

```text
size:   5,959,605,031 bytes
sha256: af73a86322f982cbab0446c6934f6f8dcc9f555f4d0652863baf04f4485a96dd
```

这里有一个必须显式处理的配置陷阱：当前 `Wan21/configs/ar_camera_tf.yaml`
中的 `generator_ckpt` 仍指向 `bidirectional/model.pt`，因为原配置用于从
Bidirectional checkpoint 训练 AR 阶段。这个新实验是从训练完成的 AR
checkpoint 继续微调，因此不能盲目继承该字段。Resolved config 中必须明确为：

```text
ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```

训练启动时打印绝对路径、文件大小和 SHA256，并在不匹配时立即终止。

### 2.3 Camera 数据事实

该单样本没有传感器测量的 camera ground truth。相机轨迹来自仓库既有数据
预处理代码对 `right-8, a-11` 的确定性展开。

该样本必须复用以下训练数据路径，而不是凭动作名称自行构造矩阵：

```text
Wan21/scripts/data_preprocessing/build_worldplaygen_lmdb.py::poses_from_pose_str
Wan21/wan_utils/dataset.py::build_viewmats_and_Ks
```

准确调用逻辑为：

```text
intrinsics, poses = poses_from_pose_str("right-8, a-11")
viewmats, Ks = build_viewmats_and_Ks(intrinsics, poses)
```

在该训练数据语义中：

```text
right-8: 连续 8 次 +3 degree yaw
a-11:    连续 11 次 local -X translation，每次 0.08
```

初始 identity pose 加 19 次增量得到 20 个 pose。默认归一化内参为：

```text
fx = 969.6969696969696 / 1920
fy = 969.6969696969696 / 1080
cx = 0.5
cy = 0.5
```

特别注意：`Wan21/wan_utils/camera_trajectory.py::parse_trajectory` 是推理 CLI
使用的另一套 `w*19,d*8` 星号语法。它没有 `right` 这个 action，并且 `d`
表示平移而不是本样本中 `right` 的 yaw。禁止把 `right-8` 猜测性改写成
`d*8`；这会得到长度相同但物理语义错误的轨迹。如果为了统一训练和推理而
重构解析器，必须以 `poses_from_pose_str` 的输出为 reference，逐元素验证
20 个 `viewmats/Ks`。不得自行发明新的坐标约定。

必须验证并记录：

```text
pose convention
quaternion order
w2c or c2w
first-frame normalization
intrinsics normalization
trajectory length
camera/action/frame/state alignment
```

期望最终得到 20 个 camera states：

```text
C_t = (V_t, K_t)
V_t: [B,4,4], OpenCV world-to-camera viewmat
K_t: [B,3,3]
t = 0...19
```

19 个动作增量加初始 camera `C0` 产生 `C0...C19`。如果真实工具输出的长度或
语义不同，必须先查明原因，不能静默截断或重复。

---

## 3. World state 与 VAE 数据构造

### 3.1 VAE 编码

读取原视频并得到：

```text
pixel video: [B,3,77,480,832]
```

使用仓库原版、冻结的 Wan VAE 和原版 normalization：

```text
[B,3,77,480,832]
-> [B,20,16,60,104]
```

也允许中间实现返回：

```text
[B,16,20,60,104]
```

但进入本任务的数据协议前必须统一为：

```text
X = [x0,x1,...,x19]
X: [B,20,16,60,104]
x_t: [B,16,60,104]
```

Wan VAE 的时间关系是：

```text
77 = 1 + 4 * (20 - 1)
```

因此这是 20 个物理 world states，不是 20 个 action transitions。

### 3.2 状态、初始化与转移的准确计数

模型共有：

```text
1 个 bootstrap generation target: x0
19 个 autoregressive transition targets: x1...x19
20 个 noisy prediction targets in total
```

不得把它写成“20 个状态转移”。

### 3.3 数据预编码缓存

建立独立缓存：

```text
Full_Duplex_Fix/cache/smallest_000000/
```

至少缓存：

```text
world_latents              [20,16,60,104]
prompt_embedding           [512,4096]
prompt_attention_mask
viewmats                   [20,4,4]
Ks                         [20,3,3]
caption
source video path/hash
source frame count/fps/resolution
VAE checkpoint path/hash
T5 checkpoint path/hash
tokenizer identity
AR generator checkpoint path/hash
action manifest path/hash
action names and frame ranges
camera convention
state/action/camera alignment
preprocessing config hash
cache format/version
latent dtype and statistics
```

同时写人类可读的：

```text
metadata.json
```

缓存必须具备版本失效逻辑。以下任一项变化都必须重新编码或明确拒绝旧缓存：

```text
video identity
VAE identity
T5/tokenizer identity
resolution
camera construction
action manifest
normalization
cache schema
```

缓存完成后检查：

```text
shape exact
dtype recorded
no NaN/Inf
latent mean/std/min/max
20 camera states
20 latent states
19 actions
bitwise reload equality where applicable
```

VAE 和 T5 只在预编码阶段运行。训练 graph 中不得包含 VAE 或 T5。

---

## 4. 交错式 Full-Duplex 序列协议

### 4.1 符号

对物理 world state `x_t` 定义：

```text
W_t = clean world span for x_t
N_t = noisy prediction span for x_t
```

训练时：

```text
W_t contains cached GT x_t
N_t contains a noised version of x_t
```

推理时不会把 GT `W_t` 注入模型。推理完成 `x_t` 的去噪后，模型会以
`timestep=0` 重新运行生成结果并把它写入 clean KV cache，作为后续状态的
历史。

### 4.2 唯一允许的主序列顺序

```text
N0, W0,
N1, W1,
N2, W2,
...
N18, W18,
N19, W19
```

等价单行表示：

```text
[N0,W0,N1,W1,...,N19,W19]
```

该序列包含：

```text
20 noisy spans: N0...N19
20 clean spans: W0...W19
40 total spans
```

保留 `W19` 以与 20 个 noisy states 形成完整对称形状，并作为连续流式推理进入
下一窗口时的 carry state。Mask 必须保证全部 noisy query（包括 `N19`）都不能
读取 `W19`，因此它不会泄漏到当前 20 个 Flow targets。

### 4.3 Patch token 数量

每个 state 增加单帧时间维：

```text
[B,16,60,104]
-> [B,16,1,60,104]
```

复用 checkpoint 原版 Conv3D Patch Embedding：

```text
kernel/stride = (1,2,2)
[B,16,1,60,104]
-> [B,1536,1,30,52]
-> [B,1560,1536]
```

主序列总长度：

```text
40 * 1560 = 62,400 tokens
```

第一版必须保留全部 1560 tokens/state。禁止 spatial stride、pooling、随机
采样或插值降维。

### 4.4 显式 layout metadata

不要依赖“偶数 span 是 noisy”之类的隐式约定。构造并保存/测试至少以下
metadata：

```text
span_index
role: noisy or clean
physical_time: 0...19
token_start/token_end
is_prediction_target
camera_index
rope_time_id
flow_timestep_source
```

期望 span 表：

```text
span 0  = N0,  prediction=true,  physical_time=0
span 1  = W0,  prediction=false, physical_time=0
span 2  = N1,  prediction=true,  physical_time=1
span 3  = W1,  prediction=false, physical_time=1
...
span 36 = N18, prediction=true,  physical_time=18
span 37 = W18, prediction=false, physical_time=18
span 38 = N19, prediction=true,  physical_time=19
span 39 = W19, prediction=false, physical_time=19
```

---

## 5. Flow timestep、噪声与训练目标

### 5.1 必须复用原 scheduler

使用原仓库：

```text
Wan21/wan_utils/scheduler.py::FlowMatchScheduler
```

并沿用 `ar_camera_tf.yaml` 的：

```text
num_train_timestep = 1000
timestep_shift = 5.0
training timestep weighting
```

不要自行近似 scheduler 的 timestep/sigma 映射。采样 index 后，应通过原
scheduler 获得 timestep，并使用原方法：

```text
add_noise
training_target
training_weight
```

### 5.2 Noisy span

对每个 `t=0...19` 独立采样：

```text
epsilon_t ~ N(0,I)
sigma_t
```

构造：

```text
N_t = y_t = (1-sigma_t) * x_t + sigma_t * epsilon_t
```

Flow target：

```text
v_target_t = epsilon_t - x_t
```

符号必须通过原 scheduler 单元测试确认。

### 5.3 Clean span

`W_t` 使用 clean latent `x_t`，其 diffusion timestep 采用原模型的 clean
context 约定，即 timestep 0。不得给 clean span 采样随机 noise level。

交错序列的 span-level timestep 逻辑为：

```text
N0: sigma0
W0: 0
N1: sigma1
W1: 0
...
N18: sigma18
W18: 0
N19: sigma19
W19: 0
```

必须正确扩展到每个 span 的 1560 tokens，并与 sequence parallel 重排同步。

### 5.4 训练噪声与评估噪声

训练默认使用随机 timestep 和随机噪声，保持标准 Flow Matching 目标分布。

评估必须另外使用固定 seed、固定 20 个 initial noises 和固定 solver 设置，以
便 checkpoint 之间可重复比较。训练随机性和评估固定噪声不得混为一谈。

---

## 6. Blockwise Attention Mask

### 6.1 不能使用普通三角 causal mask

普通序列因果 mask 会允许后面的 clean `W_t` 读取前面的 noisy `N_t`，并允许
未来 noisy query 读取历史 noisy span。这与真实推理 KV cache 不一致。

Mask 必须基于 `role + physical_time`，而不是只基于序列位置。

### 6.2 Clean query 规则

对 `W_t` 的任意 query token：

```text
允许读取 W_k, k <= t
禁止读取所有 N_k
```

即：

```text
W0 sees W0
W1 sees W0,W1
...
W18 sees W0...W18
```

### 6.3 Noisy query 规则

对 `N_t` 的任意 query token：

```text
允许读取 W_k, k < t
允许读取同一个 N_t span 内的所有空间 tokens
禁止读取 W_t
禁止读取 N_k, k != t
禁止读取未来 clean/noisy spans
```

具体为：

```text
N0 sees N0 only
N1 sees W0 + N1
N2 sees W0,W1 + N2
...
N19 sees W0...W18 + N19
```

`N0` 通过文本 cross-attention 和相机 `C0` 获得条件，但不能读取 GT `W0`。

### 6.4 Padding

FlexAttention 所需 padding token：

```text
不能作为有效 key
不能进入 loss
不能泄漏到真实 token
仅允许保留自己的对角线以避免数值问题（若内核需要）
```

### 6.5 必须编写的 mask 测试

使用小尺寸假数据，例如每 span 2～4 tokens，输出人类可读的二维矩阵，并逐项
断言：

```text
N0 cannot see W0
W0 cannot see N0
N1 can see W0
N1 cannot see N0
N1 cannot see W1
N1 can see all tokens inside N1
W1 can see W0 and W1
W1 cannot see any noisy span
all future spans invisible
padding invisible
```

真实 40-span layout 也必须做 index/count 断言，并明确断言任何 noisy span 都不能
读取 `W19`。

---

## 7. 3D RoPE 组织

### 7.1 保留原 Wan RoPE

原 Wan 对每个 patch token 使用：

```text
(physical_time, patch_height, patch_width)
```

每个 state 的空间坐标：

```text
height id = 0...29
width id  = 0...51
```

### 7.2 新序列的 physical time

交错 span 的 RoPE time 必须为：

```text
N0,W0,N1,W1,...,N19,W19
-> 0,0,1,1,...,19,19
```

不得错误地赋值为 `0...39`。

`N_t` 和 `W_t` 表示同一个物理 world state，因此共享同一个 RoPE time。它们
通过 latent 内容、diffusion timestep 和 attention mask 区分，不需要新增
role embedding。

### 7.3 原代码中必须解除的假设

原 Teacher-Forcing 实现把序列组织为：

```text
[all clean][all noisy]
```

并通过 `torch.chunk(..., 2)` 分别对两半应用同一套 RoPE。交错布局不能继续
依赖“前半/后半”判断。必须使用显式 layout/coordinates 对每个 token 应用正确
的 `(t,h,w)`。

训练和推理必须共享同一 physical-time 规则。推理生成 `x_t` 和随后以 timestep
0 重写 clean cache 时，都必须使用 `start_frame=t`。

---

## 8. Camera side stream 与 PRoPE

### 8.1 Camera 不是主序列 token

不建立 camera embedding 或 camera token。Camera 是与 world patch token 对齐
的 side-channel geometry。

交错序列对应 camera index：

```text
world:  N0,W0,N1,W1,...,N19,W19
camera: C0,C0,C1,C1,...,C19,C19
```

每个 `C_t` 扩展到该 span 的 1560 tokens：

```text
viewmats: [B,62400,4,4]
Ks:       [B,62400,3,3]
```

实现可使用 broadcast/index，不要求物理复制全部矩阵，但传给 attention 时必须
与当前 token slice 精确对齐。

### 8.2 保留原 PRoPE 数学

不得重新实现一套近似 camera attention。复用：

```text
Wan21/wan/modules/prope.py
```

原逻辑概念上为：

```text
P_t = lift(K_t) @ V_t
Q_p = P_t^T Q
K_p = P_t^-1 K
V_p = P_t^-1 V
```

然后使用与普通 RoPE attention 相同的 causal mask 计算 PRoPE attention，经
输出坐标校正和每层已有 `prope_o` 后，与普通 RoPE 分支相加：

```text
H_self = H_rope + prope_o(H_prope)
```

`ar_diffusion_tf` checkpoint 已包含每层训练过的 `prope_o`。必须严格加载并
记录这些参数，不得重新零初始化覆盖 checkpoint 值。

### 8.3 训练和推理 cache

原推理有普通 RoPE KV cache 和独立 PRoPE KV cache。生成 `x_t` 的多步去噪
过程中，当前位置可以被反复覆盖，但不能把每个 denoising step 当成新的时间
位置追加。

最终得到 clean `x_t` 后，必须使用：

```text
timestep = 0
RoPE time = t
camera = C_t
```

重新运行并覆盖/写入 clean RoPE KV 与 PRoPE KV，然后再生成 `x_{t+1}`。

---

## 9. 文本条件

使用冻结 UMT5：

```text
prompt string
-> tokenizer
-> frozen UMT5
-> prompt_embedding [B,512,4096]
```

padding row 必须通过原 wrapper 清零或 mask。相同 prompt 只编码一次，并从缓存
加载。

Wan 内部文本投影：

```text
[B,512,4096]
-> [B,512,1536]
```

每个 Transformer block：

```text
Q = world hidden tokens
K = text hidden
V = text hidden
```

文本不拼进 62,400-token 主序列。Self-Attention mask 不限制文本
Cross-Attention。

单样本过拟合不能证明通用文本控制能力。本阶段只要求保持真实文本路径并完成
T2V 闭环，不得把一个样本的记忆能力描述成文本泛化。

---

## 10. Transformer 主干与每层输出

### 10.1 必须复用的模块

从 AR checkpoint 复用：

```text
Conv3D Patch Embedding
time embedding
time projection / AdaLN modulation
30 Causal Wan Transformer blocks
self-attention Q/K/V/O
text cross-attention
FFN
normalization
3D RoPE frequencies
PRoPE prope_o
Flow Head
unpatchify
```

交错 layout、mask 和 token count 是运行时结构，不应新增或改变上述参数 shape。

### 10.2 中间层输出

每层输入/输出保持：

```text
H_l:     [B,62400,1536]
H_{l+1}: [B,62400,1536]
```

每层只更新 hidden states，不直接预测新 latent，也不执行一步 denoising。默认
不对中间层添加辅助 loss。

### 10.3 最终 Flow 输出

经过 30 层后，只 gather 20 个 noisy spans：

```text
[N0,N1,...,N19]
-> [B,20,1560,1536]
```

每个 token 经原 Flow Head 输出：

```text
16 channels * 1 * 2 * 2 = 64 values
```

然后 unpatchify：

```text
[B,20,1560,1536]
-> [B,20,1560,64]
-> [B,20,16,60,104]
```

clean spans 不进入 Flow Head，也没有直接监督。

---

## 11. Loss

### 11.1 单状态 Flow loss

对每个 `t=0...19`：

```text
L_t = mean(
    w(timestep_t) *
    (v_pred_t - (epsilon_t - x_t))^2
)
```

权重计算必须沿用 scheduler `training_weight`。

在固定噪声/timestep 的 teacher-forced 诊断中，可按当前 flow 定义恢复 clean
latent 估计：

```text
x_hat_t = y_t - sigma_t * v_pred_t
```

该式必须用原 scheduler 的实际 `sigma_t` 验证，不能把离散 timestep 数值直接
当作 sigma。`x_hat_t` 的 MSE/cosine 只作为诊断指标，不额外加入训练 loss。

### 11.2 初始化与转移日志

定义：

```text
L_init = L_0
L_transition = mean(L_1...L_19)
```

第一版总 loss 使用 20 个 noisy targets 等权平均：

```text
L_total = mean(L_0,L_1,...,L_19)
```

不要第一版人为放大 `L_init`。如果后续固定评估证明 `x0` 明显落后，可把
`lambda_init` 作为独立消融，但不能与基线结果混写。

### 11.3 禁止的 loss

第一版不得加入：

```text
final latent state MSE as training loss
camera loss
action classification
RGB loss
perceptual loss
old full_duplex task head loss
```

可以在评估中计算 latent MSE/cosine，但不能悄悄把它加入训练目标。

---

## 12. Checkpoint 加载与参数策略

### 12.1 Strict load

目标 checkpoint：

```text
ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```

先审计：

```text
absolute path
file size/hash
top-level keys
tensor count
parameter count
dtype
model specification
```

保持与原 wrapper 相同的 module names 和 parameter shapes，要求：

```python
load_state_dict(..., strict=True)
```

如果 FSDP wrapper 只导致已知前缀差异，可以显式规范化前缀，但规范化后必须
strict load。任何真实 missing/unexpected key 都必须报错。

### 12.2 不重新初始化

以下参数全部从 AR checkpoint 继承：

```text
patch embedding
timestep embedding/modulation
30-layer backbone
cross-attention
FFN/norm
prope_o
flow head
```

不要因为输入顺序改变就随机初始化它们。

### 12.3 第一版 train/freeze policy

```text
Wan VAE: frozen
UMT5: frozen
Wan generator: trainable, initialized from AR checkpoint
```

使用原 `ar_camera_tf.yaml` 的低学习率量级作为默认起点，例如 `2e-6`，但必须
进入配置并记录实际有效值。不得硬编码在训练循环中。

由于本实验改变 `4-state block -> 1-state block` 的注意力信息范围，第一版
优先允许完整 generator 低学习率适应，而不是重复旧实验中冻结 1.49B backbone
只训练小型新 head 的失败路径。

---

## 13. 必做：排列等价性验证

交错顺序和单状态 block 是两个独立变量，必须分开验证。

### 13.1 目的

先证明：当 attention graph 仍保持原 4-state block 语义时，单纯把：

```text
[all clean][all noisy]
```

改成：

```text
[N0,W0,N1,W1,...,N19,W19]
```

不会改变模型函数（除可解释的浮点/内核误差）。

### 13.2 测试条件

该测试临时使用完整 40 spans，包括 `W19`，并保持：

```text
num_frame_per_block = 4
same model weights
same latent
same noise
same timestep per original 4-state block
same text
same camera
same RoPE physical coordinates
same PRoPE geometry
same logical attention graph
```

原 4-state graph 应定义为：

```text
clean query in block b:
    sees clean blocks <= b

noisy query in block b:
    sees clean blocks < b
    sees all noisy tokens in the same 4-state block
```

将交错输出 gather 回原 noisy 顺序后，与原模型 flow output 比较：

```text
max absolute error
mean absolute error
relative error
```

使用 fp32 和实际 bf16/autocast 分别记录合理 tolerance。

如果不通过，不得继续把误差归因于 block size；必须检查：

```text
mask permutation
RoPE coordinates
timestep order
camera order
PRoPE order
SP reorder
output gather
padding
```

### 13.3 再切换单状态 block

排列等价性通过后，才把主实验改为：

```text
num_frame_per_block = 1
40 spans
62,400 tokens
```

此时性能变化才属于新的单状态 causal factorization，而不是错误的序列重排。

---

## 14. Teacher-Forcing 训练语义

一次训练 forward 同时放入全部 40 spans。

虽然 `W_t` 在 flat sequence 中位于 `N_t` 之后，但 mask 保证：

```text
N_t cannot see W_t
W_t cannot see N_t
N_{t+1} can see W_t
```

因此：

```text
N0 learns text/camera -> x0
N1 learns GT x0 + text/camera -> x1
...
N19 learns GT x0...x18 + text/camera -> x19
```

这是 Teacher Forcing。训练阶段不构建 20-state autoregressive rollout graph，
也不在训练 graph 中运行 50-step solver。

标准训练只在随机 timestep 上做一次 Flow prediction，20 个 target 并行完成。

训练和推理存在 GT history 与 generated history 的 exposure gap。本阶段先验证
基础数据流和单样本过拟合，scheduled sampling/rollout 必须留到基础方案成功
之后。

---

## 15. 纯 T2V 自主推理

### 15.1 不允许使用 GT latent

最终视频推理不得把任何 `x_t` GT 注入模型或替换预测。GT 只用于训练和独立
指标计算。

### 15.2 文本与 CFG

文本只编码一次。保留原 Wan 正/负文本 CFG：

```text
v = v_uncond + guidance_scale * (v_cond - v_uncond)
```

默认先使用与 AR checkpoint 匹配的原配置和 50-step solver，不要第一版改成
10-step sampler。

条件分支与无条件分支使用相同的当前 latent、已生成 clean history 和 camera
轨迹，但文本上下文不同。因此两条分支的逐层 hidden history 也不同，普通
self-attention KV cache 和 PRoPE KV cache 不能互相覆盖或错误复用。必须采用
以下任一种正确方式：

```text
1. cond/uncond 作为 batch 的两个样本同步运行，并保持 cache batch 维对齐；或
2. 为 cond/uncond 分别维护完整的 normal KV、PRoPE KV 和 cross-attention cache
```

每个状态完成后，同一个 predicted clean `x_t` 要分别在 cond/uncond 文本上下文
下执行 timestep-0 clean rerun。添加测试确认两分支 cache 不 alias、长度一致，
且 CFG 开关不会改变物理时间推进次数。

### 15.3 Bootstrap x0

```text
epsilon0 ~ N(0,I)
camera = C0
RoPE time = 0
clean KV history = empty
```

在 solver 的每个 timestep 调用完整 30-layer generator，得到 flow 并更新当前
latent，最终得到 predicted `x0`。

### 15.4 写入 clean cache

得到 `x0` 后，使用：

```text
latent = predicted x0
timestep = 0
camera = C0
RoPE time = 0
```

重新运行 generator，把 clean K/V 写入普通 KV cache 和 PRoPE KV cache。

### 15.5 生成后续状态

对 `t=1...19`：

```text
sample epsilon_t
use camera C_t
use RoPE time t
read cached clean x0...x(t-1)
run 50-step flow solver
obtain predicted x_t
rerun predicted x_t at timestep 0
append/overwrite clean RoPE and PRoPE cache at physical time t
```

多步 denoising 对同一物理时间反复调用模型时，不得把每一步追加成新的时间
token。最终 clean rerun 必须覆盖当前位置的 noisy cache 内容。

### 15.6 VAE 解码

最终得到：

```text
[x0,x1,...,x19]
-> [B,20,16,60,104]
-> [B,16,20,60,104]
```

使用冻结 Wan VAE：

```text
[B,16,20,60,104]
-> [B,3,77,480,832]
```

导出视频、contact sheet 和 provenance JSON。VAE 只在训练后评估/导出中运行，
不得进入训练 graph。

---

## 16. 单条数据过拟合实验

### 16.1 实验目的

单样本过拟合只验证：

```text
数据/状态对齐正确
交错 mask 无泄漏
RoPE/PRoPE 对齐正确
checkpoint 能适应 1-state block
Flow loss 可优化
自主 T2V sampler 可闭环
VAE decode 路径正确
```

它不证明文本泛化、camera 泛化或世界模型泛化。

### 16.2 配置

建立配置文件，例如：

```text
Full_Duplex_Fix/configs/overfit.yaml
```

至少显式记录：

```text
all model/data paths
checkpoint hashes
cache version
num_states = 20
num_noisy_spans = 20
num_clean_spans = 20
num_total_spans = 40
tokens_per_span = 1560
sequence_length = 62400
num_frame_per_block = 1
num_transformer_blocks = 30
spatial_token_stride = 1
timestep_shift
num_train_timesteps
learning rate
optimizer/betas/weight decay
batch size
gradient accumulation
mixed precision
gradient checkpointing
FSDP/SP settings
max steps
save/eval intervals
training seed
fixed evaluation seed
solver/sampling steps
guidance scale
```

### 16.3 推荐执行阶段

#### Stage 0：静态和小尺寸测试

完成 layout、mask、RoPE/camera ID、Flow 符号和 checkpoint strict-load 测试。

#### Stage 1：原 4-state 排列等价性

完成第 13 节测试，不更新权重。

#### Stage 2：1-state real forward/backward smoke test

真实单样本、40 spans、全部空间 tokens、30 blocks，完成至少一次 finite
forward/backward。记录峰值显存和耗时。

#### Stage 3：单样本过拟合

重复使用同一个 cached sample，使用随机 train noise/timestep，执行可配置的
optimizer steps。建议先运行 100 steps 检查趋势，再根据固定评估决定是否继续，
不能仅凭 training loss 盲目延长。

每步至少记录：

```text
total flow loss
init loss N0
mean transition loss N1...N19
per-state flow loss
timestep statistics
gradient norm
learning rate
step time
allocated/reserved VRAM
```

每个 eval interval 使用固定 noise 和固定 solver，运行 fresh autonomous T2V
采样或至少固定子集评估。

### 16.4 资源要求

最终忠实配置是：

```text
62,400 tokens
30 Transformer blocks
1560 tokens/state
20 noisy targets
20 clean states（其中 W19 不进入当前 noisy loss 的可见上下文）
```

优先复用原项目的 FSDP、gradient checkpointing 和必要的 sequence parallel。

如果真实 OOM：

1. 保存完整 OOM 日志和已使用显存。
2. 可用短序列/少状态做诊断，但必须标明不是最终实验。
3. 不得用 stride 8、1 block 等资源代理结果宣称本方案已经过拟合。
4. 先优化 activation checkpoint、FSDP/SP、padding 和 attention 实现，再考虑
   改变科学协议。

### 16.5 固定评估指标

必须从保存后的 checkpoint fresh reload 后评估，不得复用 stale training
tensors。

记录：

```text
overall latent MSE
overall latent cosine similarity
per-state MSE/cosine for x0...x19
prediction/target mean and std
zero-latent baseline MSE
initial checkpoint metrics
current checkpoint metrics
fixed-noise identity
solver step count
CFG scale
```

同时单独报告：

```text
x0 bootstrap quality
x1...x19 transition quality
```

自主采样必须生成完整 20 states 和 77 RGB frames。不能只报告
teacher-forced reconstruction。

### 16.6 成功判据

工程成功至少要求：

```text
strict checkpoint load passes
permutation-equivalence test passes
all tests pass
real 40-span forward/backward finite
loss and gradients finite
checkpoint save/resume/reload exact or within declared tolerance
autonomous 20-state T2V inference completes
VAE outputs 77 valid frames
```

单样本过拟合科学成功应尽量满足：

```text
fixed-eval latent MSE materially decreases from step 0
latent cosine materially increases
final MSE beats zero-latent baseline
prediction std approaches target std
decoded video visibly approaches the one training video
```

如果没有达到，不得写“predicted latent approximately equals GT”；必须准确报告
平台、噪点、幅度偏差或 sampler 问题。

---

## 17. 必须实现的测试

至少覆盖：

### 17.1 数据测试

```text
77 RGB frames
20 VAE states
19 actions
20 camera states
finite latents
cache reload equality
```

### 17.2 Layout 测试

```text
40 spans
20 noisy spans
20 clean spans
62,400 tokens
exact span order
exact noisy gather indices
W19 present and invisible to every noisy query
```

### 17.3 Mask 测试

执行第 6.5 节全部断言，保存可读矩阵。

### 17.4 RoPE/PRoPE 对齐测试

```text
N_t and W_t share physical RoPE time t
N_t and W_t use camera C_t
N19 uses time/camera 19
no span receives temporal id 20...38
token-level camera count equals sequence length
```

### 17.5 Flow 测试

```text
scheduler add_noise matches formula
training_target sign is epsilon - x
perfect target gives zero loss
Flow Head output/unpatchify shape exact
only noisy spans enter loss
```

### 17.6 Checkpoint 测试

```text
strict load missing=[] unexpected=[]
important parameters equal checkpoint tensors
all 30 prope_o modules loaded
no new random model parameters
```

### 17.7 排列等价性测试

执行第 13 节的原布局/交错布局输出比较。

### 17.8 Gradient 测试

确认：

```text
L_init has finite nonzero gradient to generator
L_transition has finite nonzero gradient to generator
Flow Head receives gradient
early/middle/late Transformer blocks receive gradient when trainable
VAE and UMT5 have no gradient
```

### 17.9 无泄漏扰动测试

在固定所有其他输入时修改 `W_t` 内容：

```text
N_t output must not change
N_{t+1} output should be allowed to change
```

修改某个历史 noisy `N_k`：

```text
N_t, t != k, must not read it through self-attention
```

注意 dropout、随机 noise 和非确定性内核，测试时固定 RNG/eval mode 并使用合理
tolerance。

### 17.10 推理 cache 测试

```text
denoising iterations do not advance physical time
clean rerun advances cache exactly once/state
normal and PRoPE cache endpoints agree
state t uses RoPE time t and camera C_t
20 generated latents concatenate correctly
VAE decode frame count is 77
```

---

## 18. 建议的新增文件与职责

具体命名可以遵循仓库风格，但职责必须清晰。推荐：

```text
Full_Duplex_Fix/
├── prompt.md
├── proposal.md
├── README.md
├── configs/
│   └── overfit.yaml
├── data.py or preencode.py
├── layout.py
├── mask.py
├── model.py
├── training.py
├── train_overfit.py
├── inference.py
├── evaluate.py
├── visualize_mask.py
├── tests/
├── cache/
└── outputs/
```

职责建议：

```text
preencode/data:
    real video/T5/camera preprocessing and cache validation

layout:
    40-span protocol, role/time/token/camera metadata

mask:
    role/time based FlexAttention mask and readable visualization

model:
    strict-loaded Wan adapter with interleaved forward,
    RoPE/PRoPE/timestep alignment and noisy gather

training:
    standard random-timestep Flow Matching, optimizer,
    logging, checkpoint, resume and fixed evaluation

inference:
    pure T2V sequential solver and clean KV/PRoPE cache update

evaluate:
    fresh checkpoint latent metrics and optional frozen VAE export
```

不要把所有逻辑塞进一个训练脚本。

---

## 19. 原 Wan 代码中需要重点处理的硬编码假设

实现时逐项审计并解除以下假设：

```text
_forward_train uses torch.cat([clean_x, noisy_x], dim=1)
self-attention detects TF from doubled sequence length
self-attention uses torch.chunk(q/k, 2)
RoPE assumes clean/noisy are two contiguous halves
camera trajectory is duplicated as two contiguous halves
timestep modulation is concatenated as two halves
Teacher-Forcing mask assumes [all clean][all noisy]
sequence parallel reorders clean/noisy halves specially
post-block output keeps x[:, x.shape[1]//2:]
head assumes the noisy half is contiguous
```

新的实现必须由显式 layout 驱动，而不是继续用 `half = seq_len // 2`。

同时保持原 inference 入口在旧配置下可用。如果修改共享代码，必须添加回归测试
证明原 `[all clean][all noisy]` 路径没有被破坏。

---

## 20. Checkpoint、日志与可复现性

每个训练 run 保存：

```text
model state
optimizer state
LR scheduler state
global step
best metric/step
Python/NumPy/PyTorch/CUDA RNG
full resolved config
base checkpoint identity
cache identity
layout version
mask version
camera convention
RoPE/PRoPE protocol
trainable/frozen parameter manifest
parameter counts
fixed evaluation noise hash
git commit/status if available
```

提供：

```text
best checkpoint
latest checkpoint
exact resume
fresh reload parity test
```

Resume 时架构、layout、checkpoint、cache、timestep 和 optimizer 参数不兼容应
明确拒绝，不能静默继续。

日志至少包含：

```text
metrics.jsonl
per_state_metrics.csv or jsonl
run_manifest.json
resolved_config.yaml/json
raw log
loss curve
evaluation.json
prediction provenance.json
```

---

## 21. 最终交付物

完成后必须提供：

1. `Full_Duplex_Fix/` 下完整实现代码和配置。
2. 单元测试及真实运行结果。
3. 人类可读的 mask 图。
4. Checkpoint strict-load 和排列等价性报告。
5. 数据/cache 审计报告。
6. 单样本过拟合 loss、per-state 指标和曲线。
7. Fresh checkpoint 自主生成的 20-state latent。
8. 冻结 VAE 解码得到的 77-frame 视频和 contact sheet。
9. 完整复现命令。
10. `Full_Duplex_Fix/FINAL_REPORT.md`，准确区分已验证事实、失败结果和未完成项。

README 中至少给出：

```text
environment/PYTHONPATH
preencode command
test command
permutation-equivalence command
overfit command
resume command
fresh inference command
evaluation command
VAE export command
artifact locations
```

---

## 22. 强制执行顺序

严格按照以下顺序推进：

```text
1. 审计真实代码、checkpoint、数据、camera 和 scheduler
2. 建立独立预编码缓存并验证
3. 实现显式 40-span layout
4. 实现并测试 role/time mask
5. 实现交错 RoPE/timestep/camera 对齐
6. 完成 strict checkpoint load
7. 完成 4-state 排列等价性测试
8. 切换 1-state block
9. 完成真实 40-span forward/backward smoke test
10. 运行单样本过拟合
11. Fresh reload 固定评估
12. 完全自主 T2V 生成 x0...x19
13. 冻结 VAE 解码 77 帧
14. 汇总测试、指标、资源和限制
```

任何阶段失败时，先定位该阶段，不得同时改变多个变量绕过问题。

---

## 23. 最终汇报要求

最终回复和报告必须明确回答：

```text
AR checkpoint 是否 strict load？
原布局与交错布局在 4-state graph 下是否等价？
40-span mask 是否通过全部无泄漏测试，且所有 noisy query 均不可见 W19？
RoPE physical time 是否为 0,0,1,1,...,19？
PRoPE camera 是否为 C0,C0,C1,C1,...,C19？
完整 62,400-token、30-block backward 是否真实运行？
哪些参数实际训练，数量是多少？
L_init 与 L_transition 如何变化？
固定自主采样 latent MSE/cosine 是否改善？
是否低于 zero-latent baseline？
预测 std 是否接近 target std？
是否生成完整 20 states 和 77 frames？
视频是否仍有噪点或结构错误？
Teacher Forcing 与自主推理差距是否仍明显？
哪些成功标准尚未达到？
```

不要以“代码已实现”代替“实验已成功”，也不要以 training loss 下降代替自主
T2V 质量。所有成功声明必须指向可复现的 checkpoint、指标和输出文件。
