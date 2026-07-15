
# 单状态 Micro-Turn AR Wan2.1 训练 Proposal

## 0. 核心目标

将原始 `ar_diffusion_tf` 的生成粒度从：

[
4\text{ 个 latent state / AR block}
]

细化为：

[
1\text{ 个 latent state / micro-turn}
]

对于一条 77 帧视频：

[
[3,77,480,832]
]

经过 Wan2.1 VAE Encoder 得到：

[
[16,20,60,104]
]

即 20 个 world state：

[
x_0,x_1,\ldots,x_{19}
]

其中：

[
x_t\in\mathbb R^{16\times60\times104}
]

若 (x_0) 是用户提供的初始世界状态，则训练和推理需要建模 19 个状态转移：

[
x_0\rightarrow x_1\rightarrow\cdots\rightarrow x_{19}
]

因此准确地说是：

> 20 个 world states、19 个单状态 AR transitions，而不是 20 个预测 transition。

---

# 1. 基本变量定义

## 1.1 World state

定义：

[
x_t\in\mathbb R^{B\times16\times60\times104}
]

表示第 (t) 个 VAE latent state。

完整序列：

[
X=[x_0,x_1,\ldots,x_{19}]
\in
\mathbb R^{B\times20\times16\times60\times104}
]

---

## 1.2 Camera state

不要把相机称为 `camera latent`，更准确的名称是：

[
C_t=(V_t,K_t)
]

其中：

[
V_t=\operatorname{viewmat}_t
\in\mathbb R^{B\times4\times4}
]

表示第 (t) 个时刻的 world-to-camera 外参矩阵；

[
K_t\in\mathbb R^{B\times3\times3}
]

表示相机内参矩阵。

如果一段视频的焦距不变，则：

[
K_0=K_1=\cdots=K_{19}
]

但相机不断运动，因此：

[
V_0,V_1,\ldots,V_{19}
]

通常不同。

---

## 1.3 Action

动作不作为离散 token 输入 Transformer。

动作只用于确定性更新相机状态：

[
C_t\xrightarrow{a_t}C_{t+1}
]

例如：

[
V_{t+1}=V_t\Delta V(a_t)
]

19 个动作增量，加上初始相机状态 (C_0)，得到：

[
C_0,C_1,\ldots,C_{19}
]

共 20 个 Camera state。

---

## 1.4 Noisy target

对于 transition：

[
x_t\rightarrow x_{t+1}
]

Flow Matching 的目标输入不是固定的 `[MASK]` token，而是 (x_{t+1}) 的加噪版本：

# [

y_{t+1,\sigma_t}

(1-\sigma_t)x_{t+1}

\sigma_t\epsilon_t
]

其中：

[
\epsilon_t\sim\mathcal N(0,I)
]

按照当前仓库采用的 Flow Matching 方向，目标速度为：

# [

v_{t+1}^{\mathrm{target}}

\epsilon_t-x_{t+1}
]

实际实现中，符号必须与原仓库 scheduler 和训练目标保持完全一致。

---

# 2. 数据预处理

## 2.1 原始视频

最小数据样本：

```text
./dataset/SmallestData/000000_right8a11/gen.mp4
```

读取并规范化后：

[
V\in\mathbb R^{3\times77\times480\times832}
]

加入 batch：

[
V\in\mathbb R^{B\times3\times77\times480\times832}
]

---

## 2.2 VAE 编码

使用原版、冻结的 Wan2.1 VAE Encoder：

[
V
\rightarrow
Z
]

得到：

[
Z\in\mathbb R^{B\times16\times20\times60\times104}
]

时间关系是：

[
77=1+4(20-1)
]

交换通道和时间维：

[
[B,16,20,60,104]
\rightarrow
[B,20,16,60,104]
]

拆成：

[
x_t\in\mathbb R^{B\times16\times60\times104},
\qquad t=0,\ldots,19
]

---

## 2.3 相机数据

LMDB 中保存：

```text
poses       [B,20,7]
intrinsics  [B,4]
```

其中：

# [

\operatorname{pose}_t

[t_x,t_y,t_z,q_x,q_y,q_z,q_w]
]

Dataset 将其转换为：

[
\operatorname{viewmats}
\in
\mathbb R^{B\times20\times4\times4}
]

以及：

[
K_s
\in
\mathbb R^{B\times20\times3\times3}
]

对应关系为：

```text
x0  ↔ viewmat0  ↔ K0
x1  ↔ viewmat1  ↔ K1
...
x19 ↔ viewmat19 ↔ K19
```

---

# 3. Micro-turn 定义

第 (t) 个 micro-turn 定义为：

# [

\mathcal M_t

\left(
x_t,
C_t,
y_{t+1,\sigma_t},
C_{t+1},
\sigma_t
\right)
]

其中：

- (x_t)：clean 历史状态；
- (C_t)：当前相机；
- (y_{t+1,\sigma_t})：下一状态的 noisy target；
- (C_{t+1})：目标相机；
- (\sigma_t)：目标状态的 Flow timestep。

具体为：

```text
micro 0：
clean x0  + camera C0
→ noisy x1 + camera C1

micro 1：
clean x1  + camera C1
→ noisy x2 + camera C2

...

micro 18：
clean x18 + camera C18
→ noisy x19 + camera C19
```

注意，不能写成：

```text
clean xt + noisy xt
```

而应当是：

```text
clean xt + noisy x(t+1)
```

因为模型预测的是下一 world state。

---

# 4. Teacher-Forcing 序列组织

将 19 个 micro-turn 放进一次 Transformer forward：

[
[
x_0^{clean},
y_{1,\sigma_0},
x_1^{clean},
y_{2,\sigma_1},
\ldots,
x_{18}^{clean},
y_{19,\sigma_{18}}
]
]

对应的 Camera 条件流为：

[
[
C_0,C_1,
C_1,C_2,
\ldots,
C_{18},C_{19}
]
]

也就是：

```text
world stream：
clean x0
noisy x1
clean x1
noisy x2
...
clean x18
noisy x19

camera stream：
C0
C1
C1
C2
...
C18
C19
```

Camera state 不作为普通 token 插入 world token 序列。

它是一条与 world span 对齐的几何条件流，通过 PRoPE 注入 Self-Attention。

---

# 5. Patch Embedding

每个 state：

[
x_t:[B,16,60,104]
]

增加时间维：

[
[B,16,1,60,104]
]

经过原版 Wan Conv3D Patch Embedding，patch size 为：

[
(1,2,2)
]

得到：

[
[B,1536,1,30,52]
]

展平：

[
X_t
\in
\mathbb R^{B\times1560\times1536}
]

因为：

[
30\times52=1560
]

同理：

[
Y_{t+1,\sigma}
\in
\mathbb R^{B\times1560\times1536}
]

一个 micro-turn 包含：

[
1560+1560=3120
]

个 token。

19 个 micro-turn 总长度：

[
19\times3120=59280
]

因此 Transformer 输入形状为：

[
\boxed{
[B,59280,1536]
}
]

它和原始 Teacher-Forcing 的 62400 token 规模接近，没有产生数量级变化。

---

# 6. Camera 与 Token 对齐

每个 world state 有 1560 个 token，这 1560 个 token 共用对应时刻的相机参数。

例如：

```text
clean x0 的1560个token
→ viewmat0 / K0

noisy x1 的1560个token
→ viewmat1 / K1

clean x1 的1560个token
→ viewmat1 / K1

noisy x2 的1560个token
→ viewmat2 / K2
```

语义上扩展为：

[
\operatorname{token_viewmats}
\in
\mathbb R^{B\times59280\times4\times4}
]

[
\operatorname{token_Ks}
\in
\mathbb R^{B\times59280\times3\times3}
]

实际代码可以用索引、`expand` 或广播实现，不一定物理复制全部矩阵。

---

# 7. PRoPE 相机注入

对于第 (t) 个相机状态，构造：

# [

P_t

\operatorname{lift}(K_t)V_t
]

其中：

- (V_t)：相机外参；
- (K_t)：相机内参；
- (P_t)：PRoPE 使用的投影几何矩阵。

Transformer 每层产生：

[
Q,K^{attn},V^{attn}
]

为了避免符号混淆：

- 相机内参写作 (K_t^{cam})；
- Attention Key 写作 (K^{attn})。

PRoPE 对 Q/K/V 做相机相关变换，概念上为：

[
Q_t^p=P_t^TQ_t
]

[
K_t^p=P_t^{-1}K_t^{attn}
]

[
V_t^p=P_t^{-1}V_t^{attn}
]

然后执行 Camera Attention。

普通 Attention 和 PRoPE Attention 并行：

# [

H_{\mathrm{attn}}

H_{\mathrm{RoPE}}

W_{\mathrm{prope}}H_{\mathrm{PRoPE}}
]

其中：

[
W_{\mathrm{prope}}
]

是原 AR checkpoint 中已经训练好的 `prope_o`。

因此：

- `viewmat/K` 提供本次具体相机几何；
- PRoPE 固定矩阵操作把几何作用到 Q/K/V；
- `prope_o` 和 Transformer 学习如何利用相机几何生成视频。

---

# 8. 两种时间信息

模型中有两种完全不同的时间。

## 8.1 Flow timestep

[
\sigma_t
]

表示 noisy target 当前有多大噪声。

它经过原 Wan timestep embedding：

[
\sigma_t
\rightarrow
e_{\sigma_t}
]

然后通过 AdaLN 的 scale、shift、gate 调制 Transformer。

它不是视频中的第几帧。

---

## 8.2 Video temporal position

每个 state 使用全局视频时间索引：

```text
x0  → temporal id 0
x1  → temporal id 1
...
x19 → temporal id 19
```

每个 state 的 1560 个 token 共享相同的 temporal id，但空间位置不同：

[
(\tau,h,w)
]

例如：

```text
x5 的 token：
temporal id = 5
height id   = 0...29
width id    = 0...51
```

普通 Wan 3D RoPE 使用这些位置表达视频先后关系。

对于同一个 physical state：

```text
noisy x1  → temporal id 1
clean x1  → temporal id 1
```

它们的角色区别由 Attention mask 和 loss mask 决定，而不是由视频时间编码决定。

---

# 9. Timestep 分配

每个 noisy target 可以采样一个独立噪声等级：

[
\sigma_0,\sigma_1,\ldots,\sigma_{18}
]

对应：

```text
noisy x1  → sigma0
noisy x2  → sigma1
...
noisy x19 → sigma18
```

因此原始采样形状可以是：

[
\sigma\in\mathbb R^{B\times19}
]

再将每个 (\sigma_t) 广播到对应的 1560 个 token。

但第一版为了最大程度复用原 checkpoint，建议优先遵循原仓库的 timestep 采样粒度和 clean-context timestep 表示方式。

也就是说：

- noisy span 使用真实 (\sigma_t)；
- clean span 使用原版 clean context 的 timestep 约定；
- 不要自行发明新的 timestep token。

---

# 10. Blockwise Causal Attention Mask

不能直接使用普通序列三角 mask，因为 packed sequence 中同时存在：

```text
noisy x1
clean x1
```

普通三角 mask 可能让后面的 clean (x_1) 读取前面的 noisy (x_1)，污染 clean history。

需要显式构造 blockwise mask。

## 10.1 Clean query 规则

对于 clean (x_t)，允许访问：

[
x_0^{clean},x_1^{clean},\ldots,x_t^{clean}
]

禁止访问所有 noisy target。

即：

```text
clean xt 可以看：
clean x0 ... clean xt

clean xt 不可以看：
任何 noisy span
未来 clean state
```

---

## 10.2 Noisy query 规则

对于 noisy (x_{t+1})，允许访问：

[
x_0^{clean},x_1^{clean},\ldots,x_t^{clean}
]

以及自己的 noisy span：

[
y_{t+1,\sigma_t}
]

禁止访问：

- clean (x_{t+1})；
- 未来 clean state；
- 其他 noisy target；
- 未来状态。

具体为：

```text
noisy x1：
可以看 clean x0
可以看 noisy x1 自己
不能看 clean x1 及未来

noisy x2：
可以看 clean x0、clean x1
可以看 noisy x2 自己
不能看 clean x2 及未来

...

noisy x19：
可以看 clean x0...clean x18
可以看 noisy x19 自己
```

这里的“看自己”是指：

> 同一个 noisy state 内部的 1560 个空间 token 可以进行双向 Self-Attention。

---

## 10.3 Mask 逻辑表

| Query span      | 可访问的 Key span                       |
| --------------- | --------------------------------------- |
| clean (x_t)     | clean (x_0\sim x_t)                     |
| noisy (x_{t+1}) | clean (x_0\sim x_t) + 自己的 noisy span |
| clean (x_t)     | 不可访问任何 noisy span                 |
| noisy (x_{t+1}) | 不可访问 clean (x_{t+1}) 及未来         |

这样可以防止 Ground Truth 泄漏。

---

# 11. 文本条件

Prompt 通过冻结的 UMT5 Encoder 得到：

[
E_{\mathrm{text}}
\in
\mathbb R^{B\times512\times4096}
]

再经过模型内部文本投影：

[
[B,512,4096]
\rightarrow
[B,512,1536]
]

文本不拼接到视频 token 主序列。

在每一层中：

```text
视频 hidden → Cross-Attention Query
文本 embedding → Cross-Attention Key / Value
```

文本负责描述：

- 场景；
- 物体；
- 人物；
- 动作语义；
- 风格与环境。

Camera PRoPE 负责目标视角几何。

---

# 12. Transformer 主干

保留原 `ar_diffusion_tf` 的主要模块：

- Wan2.1 VAE；
- UMT5 Text Encoder；
- Conv3D Patch Embedding；
- 30 层 Wan Transformer；
- 原普通时空 RoPE；
- 原 PRoPE；
- 原 timestep embedding；
- 原 AdaLN 调制；
- 原 Cross-Attention；
- 原 Flow Prediction Head；
- 原 Unpatchify。

不增加：

- action token；
- camera token；
- camera noise；
- camera prediction head；
- masked world token；
- 随机 turn embedding；
- 随机 type embedding。

主要修改只有：

1. AR 粒度从 4 latent 改为 1 latent；
2. packed sequence 布局；
3. Camera 与 token span 的重新对齐；
4. blockwise Teacher-Forcing mask；
5. loss 只在 noisy target span 上计算。

---

# 13. Transformer 输出与 Flow Head

Transformer 输入：

[
H_{\mathrm{in}}
\in
\mathbb R^{B\times59280\times1536}
]

经过 30 层：

[
H_{\mathrm{out}}
\in
\mathbb R^{B\times59280\times1536}
]

只提取 19 个 noisy target span。

每个 noisy span 长度为 1560：

[
19\times1560=29640
]

得到：

[
H_{\mathrm{noisy}}
\in
\mathbb R^{B\times29640\times1536}
]

按 19 个 target 重新组织：

[
[B,19,1560,1536]
]

分别经过原 Flow Head 和 Unpatchify：

[
[B,19,1560,1536]
\rightarrow
[B,19,16,60,104]
]

最终模型预测：

[
\hat v
\in
\mathbb R^{B\times19\times16\times60\times104}
]

对应目标：

# [

v^{target}

[
\epsilon_0-x_1,
\epsilon_1-x_2,
\ldots,
\epsilon_{18}-x_{19}
]
]

形状同样为：

[
[B,19,16,60,104]
]

---

# 14. 训练损失

主要损失只使用 Flow Matching：

# [

\mathcal L_{\mathrm{flow}}

## \frac

\sum_{t=0}^{18}
w(\sigma_t)
\left|
\hat v_{t+1}

(\epsilon_t-x_{t+1})
\right|_2^2
]

其中 (w(\sigma_t)) 优先复用原仓库的 timestep weighting。

第一版不建议加入 camera prediction loss，因为 Camera 是已知条件，不是生成目标。

第一版也不建议加入 action classification loss，因为动作已经被确定性转换为：

[
C_t\rightarrow C_{t+1}
]

---

# 15. 训练阶段设计

## Stage 0：数据与几何检查

首先确保：

1. VAE Encoder 输出：

[[B,16,20,60,104]]

2. VAE round-trip 能正确还原视频；
3. `poses [20,7]` 能正确转换为 `viewmats [20,4,4]`；
4. `intrinsics [4]` 能正确转换为 `Ks [20,3,3]`；
5. `viewmat[0]` 的参考系处理符合原仓库；
6. PRoPE 分支确实被开启；
7. 20 个 temporal id 为连续的 (0\sim19)。

---

## Stage 1：单样本过拟合

使用：

```text
./dataset/SmallestData/000000_right8a11/gen.mp4
```

进行单样本 overfit。

训练目标是确认：

- loss 可以持续下降；
- 每个 noisy target 都能恢复对应 (x_{t+1})；
- 相机右转和左移能反映到输出中；
- Flow Head 输出不是直接作为 VAE latent 解码；
- 多步 Flow solver 能得到 clean latent。

建议首先冻结：

- Wan VAE；
- UMT5；
- 大部分 Transformer block。

训练：

- 最后若干 Transformer block；
- PRoPE 的 `prope_o`；
- LayerNorm / AdaLN；
- Flow Head。

确认流程正确后，再逐渐解冻更多 Transformer 层。

---

## Stage 2：全 Transformer 微调

当单样本 overfit 成功后，可以低学习率解冻：

- 全部 30 层 Transformer；
- PRoPE 分支；
- Flow Head；
- timestep modulation。

仍然冻结：

- VAE；
- Text Encoder。

建议采用较小学习率，避免破坏原 AR checkpoint 已有的生成和相机控制能力。

---

## Stage 3：缓解训练—推理差异

Teacher Forcing 训练中，模型看到的是 GT clean history：

[
x_0,x_1,\ldots,x_t
]

推理时看到的则是自己生成的：

[
\hat x_0,\hat x_1,\ldots,\hat x_t
]

后期可以加入：

- scheduled sampling；
- 部分预测历史替换 GT history；
- rollout fine-tuning；
- 短序列自回归训练。

第一版先不加入，先验证基本数据流正确。

---

# 16. 推理流程

给定：

- 文本 prompt；
- 初始图像或初始 latent (x_0)；
- 初始相机 (C_0)；
- 连续相机动作 (a_0,\ldots,a_{18})。

第 (t) 个推理步：

## 第一步：获得动作

[
a_t
]

例如：

```text
right
a
w
l
```

---

## 第二步：更新目标相机

[
C_{t+1}=g(C_t,a_t)
]

动作字符不进入 Transformer。

---

## 第三步：初始化目标噪声

[
y_{t+1,1}\sim\mathcal N(0,I)
]

形状：

[
[B,16,1,60,104]
]

---

## 第四步：Flow Matching 去噪

在多个 Flow timestep 上重复：

# [

\hat v_{t+1}

f_\theta
\left(
y_{t+1,\sigma},
x_{\leq t},
C_{\leq t+1},
\text{text},
\sigma
\right)
]

并使用 Flow solver 更新：

# [

y_{t+1,\sigma_{\mathrm{next}}}

y_{t+1,\sigma}

(\sigma_{\mathrm{next}}-\sigma)
\hat v_{t+1}
]

最终得到：

[
\hat x_{t+1}
]

---

## 第五步：更新历史

将：

[
\hat x_{t+1},C_{t+1}
]

加入历史，继续生成下一状态。

推理过程是：

```text
已知 x0、C0
   ↓ action0
生成 x1、更新 C1
   ↓ action1
生成 x2、更新 C2
   ↓
...
   ↓ action18
生成 x19、更新 C19
```

最终得到：

[
[\hat x_0,\hat x_1,\ldots,\hat x_{19}]
]

拼接为：

[
[B,16,20,60,104]
]

经过 Wan VAE Decoder：

[
[B,16,20,60,104]
\rightarrow
[B,3,77,480,832]
]

输出 77 帧视频。

---

# 17. 成功标准

## 训练正确性

- Flow loss 稳定下降；
- 单样本可以明显 overfit；
- 预测 velocity 的数值范围正常；
- 解码前使用的是 Flow solver 最终输出，而不是 raw velocity。

## Camera 正确性

- 改变 `viewmat` 会改变输出视角；
- 固定相机时画面不出现异常整体漂移；
- `right` 动作导致合理的横向视角变化；
- `a` 动作产生平移和视差，而不只是二维平移。

## 时间正确性

- temporal id 为全局 (0\sim19)；
- noisy target 无法读取 clean target；
- clean history 无法读取 noisy span；
- 推理时状态按顺序递归生成。

---

# 19. 一张总结构图

```text
输入数据
├── video
│   └── [B,3,77,480,832]
├── prompt
│   └── B 个字符串
├── poses
│   └── [B,20,7]
└── intrinsics
    └── [B,4]
        │
        ├──────────────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      ▼
冻结 Wan2.1 VAE Encoder                              Camera 数据转换
        │                                                      │
        ▼                                                      ├── poses
latent sequence                                                │     ↓
[B,16,20,60,104]                                               │   c2w / viewmat
        │                                                      │     ↓
        │ transpose                                            │ [B,20,4,4]
        ▼                                                      │
[B,20,16,60,104]                                               └── intrinsics
        │                                                            ↓
        │ split 20 states                                       [B,20,3,3]
        ▼                                                            │
x0,x1,...,x19                                                      C0...C19
每个 [B,16,60,104]                                                │
        │                                                            │
        ├────────────────────────────────────────────────────────────┘
        │
        ▼
构造 19 个 state transition
────────────────────────────────────────────────────────────────────

transition 0：
clean x0  + camera C0
noisy x1  + camera C1

transition 1：
clean x1  + camera C1
noisy x2  + camera C2

...

transition 18：
clean x18 + camera C18
noisy x19 + camera C19
        │
        │ 对每个目标采样
        ▼
σ0,σ1,...,σ18
ε0,ε1,...,ε18 ~ N(0,I)
        │
        ▼
y(t+1,σt) = (1-σt)x(t+1) + σtεt
        │
        ▼
world state stream
[
 clean x0,
 noisy x1,
 clean x1,
 noisy x2,
 ...
 clean x18,
 noisy x19
]
        │
        ├──────────────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      ▼
Conv3D Patch Embedding                                  Camera span 对齐
每个 state：                                             每个 state：
[B,16,1,60,104]                                         viewmat [B,4,4]
        ↓                                                K       [B,3,3]
[B,1536,1,30,52]                                               │
        ↓                                                       │
[B,1560,1536]                                                  │
        │                                                       │
        ▼                                                       ▼
19 × 2 × 1560 tokens                                   广播/索引到对应 token
        │                                               viewmats：
        ▼                                               [B,59280,4,4]
world tokens：                                          Ks：
[B,59280,1536]                                          [B,59280,3,3]
        │                                                       │
        ├───────────────────┬───────────────────┬───────────────┤
        │                   │                   │               │
        ▼                   ▼                   ▼               ▼
Video temporal IDs   Flow timestep        文本 Prompt       Camera geometry
0...19               embedding            ↓ T5              viewmat/K
3D RoPE              AdaLN调制            Cross-Attn        ↓
                                                            P=lift(K)V
                                                            ↓
                                                            PRoPE
        │                   │                   │               │
        └───────────────────┴───────────────────┴───────────────┘
                                    │
                                    ▼
                     Blockwise Teacher-Forcing Mask
                     ───────────────────────────────
                     clean xt：
                     只看 clean x0...xt

                     noisy x(t+1)：
                     看 clean x0...xt
                     + 自己的 noisy span

                     禁止：
                     clean target 泄漏
                     未来状态
                     其他 noisy target
                                    │
                                    ▼
                     30 × Causal Wan Transformer Block
                     ┌────────────────────────────────┐
                     │ 普通时空 RoPE Self-Attention  │
                     │            +                  │
                     │ Camera PRoPE Self-Attention   │
                     │            +                  │
                     │ Text Cross-Attention          │
                     │            +                  │
                     │ Timestep AdaLN / FFN          │
                     └────────────────────────────────┘
                                    │
                                    ▼
                           hidden states
                         [B,59280,1536]
                                    │
                          只提取 noisy spans
                                    ▼
                         [B,29640,1536]
                                    │
                         reshape 19 targets
                                    ▼
                        [B,19,1560,1536]
                                    │
                         原 Flow Prediction Head
                                    │
                              Unpatchify
                                    ▼
                     flow_pred [B,19,16,60,104]
                                    │
                                    ▼
                     flow_target [B,19,16,60,104]
                     [
                      ε0-x1,
                      ε1-x2,
                      ...
                      ε18-x19
                     ]
                                    │
                                    ▼
                    Lflow = MSE(flow_pred, flow_target)
```

---

# 20. 最终方案一句话总结

> 本方案将原始 Wan2.1 AR checkpoint 的生成粒度从每次 4 个 latent state 细化为每次 1 个 latent state。20 个 VAE state 构成 19 个状态转移；每个 transition 使用当前 clean world state (x_t) 作为历史条件，使用下一状态的 noisy latent (y_{t+1,\sigma}) 作为 Flow Matching 目标，并分别将 (C_t) 和 (C_{t+1}) 通过 PRoPE 对齐到对应的 world token。训练时将 19 个 transition 打包到同一条 Teacher-Forcing 序列中，通过 blockwise causal mask 防止 Ground Truth 泄漏，只在 noisy target span 上计算 Flow Matching loss；推理时则逐状态接收动作、更新 Camera state，并顺序生成 (x_1\sim x_{19})。
