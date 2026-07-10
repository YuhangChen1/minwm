# minWM × Full-Duplex Interaction World Model 前期调研与方法确认

日期：2026-07-02  
本报告基于本地 `minWM/` 仓库、minWM 论文/README、Thinking Machines 博客，以及 2026 年 5-6 月公开论文/项目检索整理。

## 0. 结论先行

你的方向可以定义为：**把 Thinking Machines 的“同一时间轴上的多流 micro-turn”思想，从音频/文本/视频交互迁移到交互式世界模型中，使模型在每个微时间片同时消费 `world_state_token + action_token + time_token`，并输出下一段 `world_state_token` 或选择保持/不输出。**

我建议的主路线是：

1. **首选从 `HY15/Action2V/ar_diffusion_tf` 继续训练，而不是从 `dmd` 直接改。**  
   这是 minWM Phase 2 Stage 1 的 teacher-forcing causal AR checkpoint，已经完成了从双向 T2V 到 causal rollout 的结构转换，仍然保留真实视频监督，适合插入新的 action/time/world-state token 结构。`dmd` 是最终 4-step 实时学生，训练时只用 conditioning 做 self-rollout，不用真实视频监督；如果先改架构再从 `dmd` 接，容易把蒸馏误差和架构误差混在一起。

2. **如果目标是显式 action token，优先用 HY Action2V 线，而不是 Wan 线。**  
   本地 minWM 的 HY camera pipeline 已经有 `USE_DISCRETE_ACTION=True` 路径：从 camera viewmats 自动离散成 81 类 action label，并通过 `action_in` 注入 timestep embedding。Wan 1.3B 更轻，但本地代码主要是 ProPE camera 条件，没有同等成熟的离散 action 模块。可以先在 HY 上验证研究假设，再把轻量实现迁回 Wan。

3. **数据不必先物理重做成巨大的 token 文件。**  
   minWM 已经把视频编码成 latent，并保留逐 latent-frame 的 `viewmats/Ks`、`poses/intrinsics` 或 trajectory string。第一版 full-duplex dataset 可以做成 on-the-fly view：每个 latent frame 约对应 4 个原始视频帧，按 24 fps 估算约 167 ms，天然接近 Thinking Machines 的 200 ms micro-turn。

4. **推荐的训练顺序：**
   - A. 复现 `HY15/Action2V/ar_diffusion_tf` 推理，确认数据/轨迹/latent 对齐。
   - B. 在 Stage 1 AR checkpoint 上插入 `time_in_microturn`、显式 `action_token`、可选 `emit/hold token`，新模块 zero-init，只训新模块/LoRA/后几层。
   - C. 稳定后重新跑 causal ODE 或 causal CD；优先 `causal_cd` 作为稳定的 few-step 蒸馏入口。
   - D. 最后再用 Stage 3 asymmetric DMD self-rollout 做实时化。

## 1. minWM checkpoint 选择

### 1.1 minWM 原始阶段

minWM 是一个把 bidirectional T2V foundation model 转成 action-conditioned video world model 的开源框架。README 和训练文档把训练拆成：

- Phase 1: **Bidirectional SFT**
- Phase 2 Stage 1: **Teacher Forcing AR Diffusion**
- Phase 2 Stage 2a: **Causal ODE**
- Phase 2 Stage 2b: **Causal Consistency Distillation**
- Phase 2 Stage 3: **Asymmetric DMD with Self Rollout**

对应 checkpoint：

- `HY15/Action2V/bidirectional`
- `HY15/Action2V/ar_diffusion_tf`
- `HY15/Action2V/causal_ode`
- `HY15/Action2V/causal_cd`
- `HY15/Action2V/dmd`
- Wan 线也有 `Wan21/Action2V/{bidirectional, ar_diffusion_tf, causal_ode, causal_cd, dmd}`

来源：minWM README 与训练文档、本地仓库 `training_hunyuan.md` / `training_wan.md`，以及 [minWM GitHub](https://github.com/shengshu-ai/minWM)、[minWM arXiv 2605.30263](https://arxiv.org/abs/2605.30263)。

### 1.2 推荐 checkpoint

**主实验推荐：`HY15/Action2V/ar_diffusion_tf`**

理由：

- 它已经是 causal AR teacher-forcing 训练后的模型，和 micro-turn streaming 的方向一致。
- 它仍然用真实视频 latent 监督，适合做架构调整和对齐训练。
- 它保留 camera/action 条件路径，便于改成 `world_state_token + action_token + time_token`。
- 之后可以继续进入 `causal_ode` / `causal_cd` / `dmd`，符合 minWM 原流程。

**显式离散 action token 版本推荐：`HY-WorldPlay bidirectional_model` → minWM Stage 1，或 `HY15/Action2V/ar_diffusion_tf + USE_DISCRETE_ACTION=True`。**

本地 `training_hunyuan.md` 有一节 “Using WorldPlay Checkpoints as Init”。开启 `USE_DISCRETE_ACTION=True` 后，训练脚本会：

- 添加 `action_in` 模块；
- 从 `viewmats` 派生 81 类 action label；
- 把 action embedding 加到 timestep vector 中。

本地代码中 81 类 action 的定义是：

- translation 9 类：no-action, forward, backward, left, right, 以及对角组合；
- rotation 9 类：yaw / pitch 等；
- 最终 label = `trans_label * 9 + rotate_label`。

这和你的 `action_token` 想法直接匹配，只是现在 action 主要是 camera action。

**轻量快速原型：`Wan21/Action2V/ar_diffusion_tf`**

如果算力压力很大，Wan 1.3B 线可以作为快速验证：从 `Wan21/Action2V/ar_diffusion_tf` 接，在数据层把 trajectory/viewmats 转成 action token，再加 time token。但需要自己补一条类似 HY 的 discrete action embedding 路径。

**不建议一开始从 `dmd` 直接接。**

`dmd` 适合最后实时化。minWM 文档明确 Stage 3 DMD 训练是 conditioning-only self-rollout，不再消费真实视频 supervision。先从这里改 token 结构，容易出现动作漂移、上下文遗忘、末段崩坏，并且难判断是架构问题还是 distillation 问题。

## 2. 数据如何适配 full-duplex micro-turn

### 2.1 minWM 现有数据可复用内容

minWM HY Action2V 数据准备需要：

- `preencode_input.json`：至少包含 `image_path` / `caption` / `pose_str`；
- `videos/`：视频目录；
- preencode 后得到 `./dataset/HY15/Action2V/latents/` 和 `train_index.json`；
- 每个 latent `.pt` 内含视频 latent、text/image condition、`intrinsics`、`poses`，代码会构建 `viewmats` 和 `Ks`；
- ODE/CD 阶段另有 `Action2V_ode` latent 或复用 encoded data。

本地示例 `assets/example.json` 使用：

```json
{
  "image": "img/1.png",
  "caption": "...",
  "trajectory": "a*4,w*8,s*7"
}
```

轨迹字符串会映射到每个 latent frame 的 camera motion。`Wan21/wan_utils/camera_trajectory.py` 中每步 translation 是 `0.08`，rotation 是 `3°`；HY 线还有 `trajectory_str_to_action_labels()` 和 `discretize_poses_to_actions()`。

### 2.2 把视频 world model 改成 full-duplex token 流

Thinking Machines 的核心不是“多模态本身”，而是：

- 同一条时间轴；
- 固定 micro-turn；
- 输入流和输出流交错；
- 模型每个 micro-turn 都可以决定是否输出；
- 没有人为用户/助手回合边界。

迁移到 minWM 后，可以定义每个 micro-turn：

```text
microturn t:
  input:
    time_token(t, delta_t)
    world_state_token(z_t or z_{t-k:t})
    action_token(a_t)
    optional_condition_token(text/image/static scene)

  output target:
    next_world_state_token(z_{t+1}) 或 block z_{t+1:t+B}
    optional_emit_token / confidence / stop-or-hold
```

其中：

- `world_state_token`：minWM VAE latent patch tokens，第一版不需要新 tokenizer；
- `action_token`：先用现有 camera trajectory 或 81 类 discrete action label；
- `time_token`：micro-turn index、absolute time、`delta_t`，以及 block index；
- `state output`：下一 latent frame 或小 block；为了接近 200 ms，建议先设 **1 latent-frame ≈ 1 micro-turn**。minWM Hunyuan 配置常见 `num_frames=77`、latent time ≈ 20，因此每个 latent step 约 4 个原始帧，24 fps 下约 167 ms，和 200 ms 接近。

### 2.3 推荐的数据转换实现

第一版不要离线展开成海量 JSON。建议实现一个 dataset wrapper：

1. 读原 `train_index.json` 或 Wan LMDB。
2. 对每条样本加载：
   - `latent`: `(C, T, H, W)`
   - `viewmats/Ks` 或 `poses/intrinsics`
   - `action`: 从 `viewmats` 离散得到 `(T,)`
   - `caption/image_cond/vision_states`
3. 构造 micro-turn 序列：
   - `state_t = latent[:, t]`
   - `action_t = action[t]`
   - `time_t = t, delta_t`
   - `target = latent[:, t+1]`
4. 注意 mask：
   - `state_t/action_t/time_t` 只能 attend 到过去；
   - `target_{t+1}` 不能泄漏未来 action/state；
   - 如果用 block，block 内可 full attention，但 block 间 causal。
5. 保留原 ProPE 连续条件：
   - 离散 action token 是“语义动作”；
   - `viewmats/Ks` 是“几何连续动作”；
   - 二者一起用，比只用 81 类 action 更稳。

### 2.4 数据增强建议

- **动作 dropout / no-op 增强**：随机把一部分 action token 置为 no-action，训练模型利用状态惯性。
- **时间 jitter**：随机把 `delta_t` 设为 1 或 2 个 latent step，使模型学会非均匀交互节奏。
- **action perturbation**：对连续 pose 加小扰动，并保持离散 label 不变，提高对控制误差的鲁棒性。
- **短窗到长窗 curriculum**：先 8-12 latent frames，再 20，再更长 rollout。
- **future action masking 检查**：这是最容易出 bug 的点。full-duplex 设定下，模型只能看到当前 micro-turn 的 action，不能偷看未来 trajectory。

## 3. 类似探索、benchmark、baseline、dataset

### 3.1 最相关的模型/方法

**minWM / Causal Forcing / Causal Forcing++**

- minWM：完整开源 reproduction pipeline，支持 Wan 2.1 和 HunyuanVideo 1.5，目标是 real-time interactive video world model。
- Causal Forcing：面向 autoregressive diffusion distillation，解决高质量实时交互视频生成。
- Causal Forcing++：进一步做 scalable few-step AR diffusion distillation，是 minWM Stage 2b/Stage 3 的核心来源。

资料：

- [minWM GitHub](https://github.com/shengshu-ai/minWM)
- [minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models](https://arxiv.org/abs/2605.30263)
- [Causal Forcing](https://arxiv.org/abs/2602.02214)
- [Causal Forcing++](https://arxiv.org/abs/2605.15141)

**HY-WorldPlay / WorldPlay**

这是与你最接近的已有模型之一：action-conditioned interactive video world model，使用 camera/action 控制，并有 discrete action 模块。minWM 也明确支持用 `tencent/HY-WorldPlay` checkpoint 初始化。它的思想包括：

- Dual action representation：连续 camera representation + 离散 action label；
- Reconstituted Context Memory：长时记忆；
- Context Forcing：改善长时 rollout。

资料：

- [tencent/HY-WorldPlay Hugging Face](https://huggingface.co/tencent/HY-WorldPlay)
- [WorldPlay: Towards Long-Term Geometric Consistency for 3D World Generation](https://arxiv.org/abs/2512.14614)

**BiWM**

BiWM 强调 interactive world model 中 rollout 会退化，提出 bidirectional autoregressive learning，结合 forward KL 和 adversarial loss 做抗退化训练。它对你的项目有启发：full-duplex 世界模型会长期自回归，必须专门处理 rollout degradation。

资料：

- [BiWM: Bidirectional Autoregressive Learning for Interactive World Model](https://arxiv.org/abs/2606.07107)

**WorldCraft**

WorldCraft 把交互从 camera navigation 扩展到 object manipulation，提出 object-aware action data collection、混合训练策略和跨领域 object actions。它对应你的下一阶段：不要只做 camera action token，而要做 object/action/state 的耦合。

资料：

- [WorldCraft: From Camera Navigation to Object Manipulation in Interactive World Models](https://arxiv.org/abs/2606.12863)

**GameNGen / DIAMOND / Oasis / MineWorld**

这些是更“游戏世界模型”的 baseline，常见输入是 action + image observation，输出下一帧/下一状态。它们适合做低成本 baseline 或思想对照：

- GameNGen：用 diffusion model 模拟 DOOM gameplay。
- DIAMOND：diffusion world model for RL / visual control。
- Oasis / Open-Oasis：Minecraft 风格的 interactive generated world。
- MineWorld / MineRL 系列：可作为低维动作环境和数据来源。

资料：

- [GameNGen](https://arxiv.org/abs/2408.14837)
- [DIAMOND](https://arxiv.org/abs/2405.12399)
- [Open-Oasis GitHub](https://github.com/etched-ai/open-oasis)
- [MineRL](https://minerl.io/)

### 3.2 Benchmark / dataset

**minWM-data**

最直接的数据起点。包含 raw videos、preencode input、ODE data、negative prompt embeddings 等。适合先做 pipeline validation。

- [MIN-Lab/minWM-data](https://huggingface.co/datasets/MIN-Lab/minWM-data)

**WBench**

WBench 是交互式世界模型 benchmark，支持 unified text / pose / action condition，含多轮交互设置和较多自动指标。适合作为你的 full-duplex world model 的主 benchmark 候选。

- [WBench: Benchmarking Interactive World Models](https://huggingface.co/papers/2605.25874)

**iWorld-Bench**

iWorld-Bench 关注 interactive world model 的 action-conditioned generation，提出 Action Generation Framework，并提供大规模 synthetic action-conditioned dataset 与 test split。它适合评估“动作是否真正驱动世界状态变化”。

- [iWorld-Bench: A Benchmark for Interactive World Models](https://arxiv.org/abs/2605.19343)

**WorldMark**

WorldMark 是交互式视频世界模型 benchmark，强调 camera action / key action / combined action / unusual scene 等维度。适合补充人工评估和 stress test。

- [WorldMark: A Benchmark for Interactive Video World Models](https://arxiv.org/abs/2604.21686)

**VBench / VBench++ / WorldScore**

这些不是专门为 full-duplex 交互设计，但可作为视频质量、运动、主体一致性、物理合理性的通用评估。

- [VBench](https://github.com/Vchitect/VBench)
- [VBench++](https://github.com/Vchitect/VBench)
- [WorldScore](https://arxiv.org/abs/2504.00983)

### 3.3 Baseline 设计

建议至少做以下 baseline：

1. **minWM 原版 DMD**：`HY15/Action2V/dmd` 或 `Wan21/Action2V/dmd`，输入原 trajectory，评估动作跟随和视频质量。
2. **minWM AR teacher forcing**：`HY15/Action2V/ar_diffusion_tf`，不用新 token，仅 causal rollout。
3. **ProPE-only full-duplex**：只加入 time/micro-turn mask，不加入 discrete action token。
4. **Discrete-action-only**：不用连续 viewmats/Ks，只用 81 类 action token。
5. **Dual action representation**：continuous ProPE + discrete action token，这是主方法。
6. **No-time-token ablation**：去掉 explicit time token，验证 full-duplex 时间片编码是否有贡献。
7. **Variable-dt ablation**：固定 1 latent step vs 随机 1/2/4 latent step。

核心评估指标：

- Action adherence：预测相机轨迹的 RPE / ATE、translation/yaw/pitch 分类准确率；
- Visual quality：VBench 指标、FVD、LPIPS、CLIP similarity；
- Temporal consistency：frame-to-frame jitter、identity/scene persistence；
- Long rollout degradation：20/40/80 latent step 的质量衰减曲线；
- Latency：每 micro-turn 生成耗时、KV cache 命中率、首帧/下一帧延迟；
- Interruptibility：中途切换 action 的响应延迟和过冲程度。

## 4. 训练稳定 trick

### 4.1 架构层面

- **新模块 zero-init**：`action_in`、`time_in`、`emit_head` 都 zero-init，让初始行为尽量等价于原 minWM。
- **dual action representation**：不要只用离散 action。保留 `viewmats/Ks` 的 ProPE 连续几何约束，同时把 81 类 action 作为离散语义 token。
- **显式 `delta_t`**：micro-turn 是时间系统，不只需要 token index。加入 `delta_t` 可以支持可变时间片和跳帧。
- **KV cache streaming**：仿照 full-duplex interaction model，把历史 world state/action/time 保存在缓存中；长上下文时做 memory compression 或 keyframe memory。
- **emit/hold token 从后期再加**：第一阶段强制每个 micro-turn 输出下一 state；等稳定后再训练“是否输出/是否保持”的 head。

### 4.2 训练日程

1. **冻结大部分 backbone，只训新 token adapter / LoRA / 后几层**，跑通 5k-20k steps。
2. **teacher-forcing micro-turn 训练**：目标是 `z_t + a_t + time_t -> z_{t+1}`，不要一开始做长 self-rollout。
3. **scheduled sampling / self-forcing**：逐步把一部分 `z_t` 换成模型预测 state，降低 exposure bias。
4. **causal CD 优先于 DMD**：DMD 放最后。先用 causal CD/ODE 让 few-step student 稳定。
5. **DMD 短训小步**：minWM 文档对 Wan DMD 建议 100-200 steps；HY 也应先短训验证，不要长时间盲训。

### 4.3 数据和 mask 防错

- **所有 window 长度必须对齐 `num_frame_per_block`**。本地代码注释已经指出：21 帧会导致最后一帧 token 只能 attend text、vision self-attn 为空，末尾 denoise 会退化。full-duplex 改造必须严格裁剪到 block multiple。
- **未来 action 不能泄漏**。原始 trajectory 是完整给定的，改成 full-duplex 后要确保第 t 步只能看 `a_{\le t}`。
- **start_idx 不要长期硬编码 0**。本地 ODE dataset 有 `start_idx = 0` 用于避免 condition mismatch；研究版本应明确保留或替换为 condition-aligned random crop。
- **动作切换样本要过采样**。full-duplex 的核心价值是中途改变 action，训练集要包含 action switch / interruption / no-op / reverse control。

### 4.4 Loss 和辅助任务

- 主 loss：flow matching / diffusion denoising loss，沿用 minWM。
- 辅助 action consistency：从预测 `z_{t:t+1}` 反推 action label，和输入 `a_t` 做 CE。
- Pose consistency：如果可以从预测视频估 camera motion，用 RPE/ATE loss 或 evaluation-only metric。
- Velocity / delta latent loss：约束 `z_{t+1}-z_t`，减少 jitter。
- Long-rollout consistency：每 N 步加 keyframe/teacher consistency，防止场景漂移。
- CFG/action dropout：训练时随机 drop caption 或 action，推理时可以调 action guidance。

## 5. 项目完成 checklist

### Milestone 0: 复现和数据核验

- 下载/确认 `HY15/Action2V/ar_diffusion_tf`、`causal_cd/causal_ode`、`dmd`。
- 跑通原版 `run_infer_ar_diffusion_camera.sh` 和 `run_infer_causal_camera.sh`。
- 可视化一个样本的 `latent T`、`trajectory`、`viewmats/Ks`、81 类 action label。
- 检查所有训练 window 是否是 `num_frame_per_block` 的整数倍。

### Milestone 1: Micro-turn dataset

- 实现 `MicroTurnActionDataset` wrapper。
- 输出 batch 字段：
  - `state_latent`
  - `target_latent`
  - `action_label`
  - `viewmats/Ks`
  - `time_index`
  - `delta_t`
  - text/image condition
- 写单元测试：无未来泄漏、shape 对齐、action label 与 trajectory 对齐。

### Milestone 2: Full-duplex token adapter

- 在 HY ProPE transformer 上加入 `time_in` 和可选 `action_token_in`。
- 继承现有 `action_in`，保持 zero-init。
- 增加 attention mask：micro-turn causal，block 内可 full attention。
- 从 `HY15/Action2V/ar_diffusion_tf` 加载权重，missing key 只允许新模块。

### Milestone 3: Teacher-forcing 训练

- 冻结大部分 backbone，训 adapter/LoRA。
- 用 1 latent-frame micro-turn 跑短实验。
- ablation：no time token、no discrete action、ProPE-only、dual action。
- 每 1k-5k steps 做固定 trajectory validation。

### Milestone 4: Few-step distillation

- 使用稳定的 full-duplex AR teacher 重新生成 ODE data，或直接跑 causal CD。
- 训练 `causal_ode` / `causal_cd` 版本。
- 评估 latency 和长 rollout degradation。

### Milestone 5: DMD 实时化和 benchmark

- 从 `causal_cd` 或 `causal_ode` 初始化 DMD。
- 先短训 100-200 steps，再逐步加长。
- 在 WBench / iWorld-Bench / WorldMark 风格任务上评估：
  - camera following；
  - action switching；
  - long-horizon consistency；
  - unusual scene / no-op / reverse action。

## 6. 2026 年 5-6 月 full-duplex / follow Thinking Machines 相关工作

### 6.1 Thinking Machines Interaction Models

Thinking Machines 的 [Interaction Models: A Scalable Approach to Human-AI Collaboration](https://thinkingmachines.ai/blog/interaction-models/) 是你的动机来源。它提出：

- 不再依赖用户/助手 turn boundary；
- 使用连续 micro-turn；
- 视频、音频、文本输入和音频/文本输出沿同一时间轴交错；
- 每约 200 ms 处理一段输入，并决定是否输出；
- 模型看到的是交错 token 序列，人类感知上是并发输入/输出流。

对你的项目的关键启发是：**世界模型也不应该只接收完整 action sequence 后离线生成视频，而应在统一时间轴上持续接收 action/state 并持续更新世界状态。**

### 6.2 DuplexSLA: 双工语音-语言-动作模型

DuplexSLA 是 2026 年 5 月下旬公开的 full-duplex spoken LLM 工作。它把交互拆成三条 channel：

- user audio channel；
- agent audio channel；
- agent action channel；

并在毫秒级共享时间轴上同步。它还提出 DuplexSLA-Bench。虽然它是语音系统，不是世界模型，但 channel 设计非常适合迁移：你的版本可以变成 `state channel + action channel + output world-state channel`。

资料：

- [DuplexSLA arXiv](https://arxiv.org/abs/2605.20755)
- [DuplexSLA GitHub / demo / benchmark](https://github.com/hyzhang24/DuplexSLA)

### 6.3 DuplexOmni: full-duplex multimodal omni interaction

DuplexOmni 是 2026 年 6 月 8 日左右的 full-duplex omni model。它强调：

- interaction layer 负责理解输入并输出 channel；
- thinking layer 负责推理；
- 使用 480 ms 级别的 streaming slice；
- writer-director 数据流水线生成带时间同步的 full-duplex 数据。

这对你的项目有两个启发：

- 把 world-state dynamics 和 action-response decision 分层；
- 构造 synthetic full-duplex action/state 数据，而不是只依赖原始完整视频。

资料：

- [DuplexOmni arXiv](https://arxiv.org/abs/2606.04175)

### 6.4 Wan-Streamer: 原生流式 audio-visual generation

Wan-Streamer 是 2026 年 6 月 25 日左右的 streaming audio-visual generation 工作。它和 Thinking Machines 最像的地方是：

- 直接流式处理 visual / audio / text 输入输出 token；
- 使用 interleaved 序列统一输入输出；
- block-causal attention；
- 约 160 ms streaming unit，目标是低延迟连续生成。

它不是世界模型，但它的 block-causal mask、interleaved token design、latency 评估方法都可以借鉴到 minWM full-duplex 改造中。

资料：

- [Wan-Streamer arXiv](https://arxiv.org/abs/2606.25041)
- [Wan-Streamer 项目页](https://wan-streamer.com/)

### 6.5 Causal Forcing++ / minWM / BiWM：世界模型方向的同步趋势

这些工作不一定自称 full-duplex，但它们解决的是同一个系统瓶颈：如何让视频世界模型实时、低步数、可交互地根据 action 继续生成。

- Causal Forcing++：few-step AR diffusion distillation。
- minWM：全栈开源 reproduction。
- BiWM：处理 interactive world model 的自回归退化。

它们和 Thinking Machines 的交叉点是：**TML 解决人机交互的实时 token 编排；minWM/Causal Forcing/BiWM 解决世界状态 rollout 的实时可控生成。你的项目正好是把二者合并。**

## 7. 建议的论文定位

可以把项目标题/贡献点暂定为：

**Full-Duplex Interactive World Models via Micro-Turn State-Action-Time Tokenization**

潜在贡献：

1. **Micro-turn world model formulation**：把交互式视频世界模型表述为统一时间轴上的 state/action/time token 流。
2. **Dual action tokenization**：连续 SE(3)/ProPE + 离散 action label，在同一 micro-turn 内耦合。
3. **Streaming causal attention mask**：适配 minWM 的 block-causal attention，防未来泄漏，支持 action interruption。
4. **Stagewise distillation recipe**：从 minWM AR teacher 到 causal CD/ODE 再到 DMD 的稳定训练流程。
5. **Full-duplex world-model evaluation**：除了视频质量，还评估 action switch latency、interruptibility、long-rollout degradation。

最小可发表版本不需要一开始超越所有大模型。只要证明：

- 相比 minWM 原版，micro-turn action/time token 对中途 action 切换响应更快；
- 相比 ProPE-only，dual action token 更稳；
- 相比 no-time-token，显式 time token 在 variable-dt 或 skip/hold 设置下更好；
- few-step distillation 后仍保留 action adherence。

## 8. 参考链接

- Thinking Machines: [Interaction Models](https://thinkingmachines.ai/blog/interaction-models/)
- minWM GitHub: [shengshu-ai/minWM](https://github.com/shengshu-ai/minWM)
- minWM paper: [arXiv 2605.30263](https://arxiv.org/abs/2605.30263)
- Causal Forcing: [arXiv 2602.02214](https://arxiv.org/abs/2602.02214)
- Causal Forcing++: [arXiv 2605.15141](https://arxiv.org/abs/2605.15141)
- HY-WorldPlay: [Hugging Face](https://huggingface.co/tencent/HY-WorldPlay)
- WorldPlay: [arXiv 2512.14614](https://arxiv.org/abs/2512.14614)
- BiWM: [arXiv 2606.07107](https://arxiv.org/abs/2606.07107)
- WorldCraft: [arXiv 2606.12863](https://arxiv.org/abs/2606.12863)
- WBench: [Hugging Face paper page](https://huggingface.co/papers/2605.25874)
- iWorld-Bench: [arXiv 2605.19343](https://arxiv.org/abs/2605.19343)
- WorldMark: [arXiv 2604.21686](https://arxiv.org/abs/2604.21686)
- GameNGen: [arXiv 2408.14837](https://arxiv.org/abs/2408.14837)
- DIAMOND: [arXiv 2405.12399](https://arxiv.org/abs/2405.12399)
- Open-Oasis: [GitHub](https://github.com/etched-ai/open-oasis)
- DuplexSLA: [arXiv 2605.20755](https://arxiv.org/abs/2605.20755), [GitHub / demo / benchmark](https://github.com/hyzhang24/DuplexSLA)
- DuplexOmni: [arXiv 2606.04175](https://arxiv.org/abs/2606.04175)
- Wan-Streamer: [arXiv 2606.25041](https://arxiv.org/abs/2606.25041), [project](https://wan-streamer.com/)
