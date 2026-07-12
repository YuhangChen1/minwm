unset http_proxy
unset https_proxy
unset ftp_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY
unset ALL_PROXY
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:$PYTHONPATH"
export PYTHONNOUSERSITE=1

export HF_ENDPOINT=https://hf-mirror.com

INCLUDES=("preencode_input.json" "others/HY/Action2V/**")
for i in $(seq 0 19); do
  id=$(printf "%06d" "$i")
  INCLUDES+=("videos/${id}_*/gen.mp4")
done

hf download MIN-Lab/minWM-data \
  --repo-type dataset \
  --local-dir ./dataset_min20 \
  --include "${INCLUDES[@]}"


export PIP_CACHE_DIR=/hyperai/home/.cache/pip
export CONDA_PKGS_DIRS=/hyperai/home/.cache/conda
export HF_HOME=/hyperai/home/.cache/huggingface
export TORCH_HOME=/hyperai/home/.cache/torch
export TRITON_CACHE_DIR=/hyperai/home/.cache/triton

一步我建议：

  denoising_steps = 10 保持不变
  blocks: 1 → 4
  stride = 8 暂时保持

  具体顺序：

  1. 显式迁移 step-200 微调参数到 blocks=4，不能静默绕过 checkpoint 配置检查。
  2. 先跑 1 step，检查显存、loss、梯度和数值稳定性。
  3. 再跑 10 steps，重新生成视频。
  4. 如果 MSE/cosine 明显改善，继续到 50–100 steps。
  5. 粗粒度 flow 学好后，再将 stride 8 → 4 提升空间清晰度。

  world prior 的学习率建议从当前 0.01 降到 0.001–0.003，避免它继续放大颗粒纹理。

  目前不应直接改 stride 4，因为当前模型距离 stride-8 自身的低通参考仍很远，说明首先是只运行 1/30 Transformer blocks 导致的表达能力不足。先增加到 4 blocks，最有可能在不立即引爆
  序列长度的情况下改善人物结构、时序一致性和 flow 质量。

─ Worked for 6m 50s ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


› 这是因为我的资源太少只有单卡无法全量训练的原因吗


• 部分原因是单卡资源限制，但更准确地说，是“当前完整训练图太大”，并不是 H100 本身性能差。

  我们已经有实测证据：

   配置                                           结果
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   19 turns、stride 1、30 blocks、10-step BPTT    H100 80GB OOM，约 79GB 已占用
  ─────────────────────────────────────────────  ───────────────────────────────
   19 turns、stride 8、30 blocks                  44.18GB，约 161 秒/step
  ─────────────────────────────────────────────  ───────────────────────────────
   当前 stride 8、1 block                         11.93GB，约 13.4 秒/step

  完整训练之所以夸张，是因为同时包含：

  19 turns
  × 每 turn 10 次去噪
  × 全历史 token
  × 最长约 89,000 tokens
  × 30 Transformer blocks
  × 跨 turn 保留反向传播图

  即使 backbone 冻结，也仍要保存或重算经过这些层的激活，才能把梯度传回 camera/action/token encoder 和输出 head。

  当前噪点的直接原因

  资源限制迫使当前版本采用了两个很强的缩减：

  - blocks=1：只执行 Wan2.1 的 1/30 Transformer 深度；
  - stride=8：每个 latent 只保留 28 个空间 token，预测约 8×14 的 flow，再插值到 60×104。

  VAE 最后会把 60×104 放大到 480×832，因此 latent 中很小的误差、插值纹理和分布偏移都会被放大成明显噪点。

  相机更容易拟合，是因为相机只有 13 个数值，而且主要是低频、连续运动；world state 每 turn 有约 10 万个 latent 数值，难度完全不同。

  但单卡并不意味着无法继续提升

  我建议采用单卡友好的分阶段策略：

  阶段 1：19 turns、10 steps、stride 8、blocks 1→4
  阶段 2：保持 blocks=4，将 stride 8→4
  阶段 3：高分辨率使用 4-turn 短窗口训练
  阶段 4：定期回到 19-turn stride-8 做全局一致性训练

  这样分别学习：

  - 19-turn 粗分辨率训练：相机、动作、长期运动；
  - 短窗口高分辨率训练：人物轮廓、纹理、去噪和局部细节。

  实际 OOM 已经发生，因此后续高分辨率阶段可以合理使用 truncated BPTT，但仍保留周期性的完整 19-turn 梯度验证，不会完全退化成逐 turn teacher forcing。

  还可以加入：

  - gradient checkpointing，当前已经启用；
  - 只对最后若干 Transformer blocks 加 LoRA；
  - CPU activation offload；
  - 全分辨率局部 residual refiner；
  - 大部分步骤训练随机单个 sigma 的 flow loss，间隔执行完整 10-step state loss。

  如果增加 GPU，仅使用普通 DDP 没有明显帮助，因为 batch size 是 1，每张卡都会复制同一模型。需要 sequence/context parallel 或 pipeline parallel，才能真正分摊这条超长序列。

  所以结论是：当前质量确实间接受单卡限制影响，但不是必须换机器才能改善。下一步最合理的实验仍然是保持 10-step 和 stride 8，将 blocks=1 → 4，先利用更多已经加载好的 Wan2.1 预训练
  层。然后再用 stride 4 提升空间清晰度。

  可以测试增加，但暂时不建议把训练从 10 步直接改成 20/30 步，更不应把它当作当前噪点的主要解决方案。

  原因是当前 Flow Matching 定义为：

  xσ = (1-σ)x0 + σε
  target velocity = ε - x0

  真实 velocity 是常量；如果模型预测准确，Euler 积分理论上一两步也能到达目标。增加步数主要补偿模型的 flow 预测误差，无法恢复 stride-8 丢掉的空间信息，也无法弥补只执行 1/30 个
  Transformer blocks 的容量缺失。对应实现见 full_duplex/flow.py:29 和 full_duplex/training.py:297。

  建议先做纯推理消融，不重新训练：

   去噪步数    19-turn 模型调用次数    用途
  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━
          5                      95    判断模型是否依赖过多迭代
  ──────────  ──────────────────────  ──────────────────────────
         10                     190    当前基线
  ──────────  ──────────────────────  ──────────────────────────
         20                     380    推荐测试项
  ──────────  ──────────────────────  ──────────────────────────
         30                     570    仅在 20 步有效时测试

  必须保持：

  - 同一个 step-200 checkpoint；
  - 相同初始 noise 和 seed；
  - 相同 19 个动作；
  - 相同 VAE；
  - 只改变 sigma 网格和去噪步数。

  对比指标：

  - latent MSE；
  - cosine similarity；
  - prediction std 是否从 1.171 接近 GT 的 0.828；
  - 高频噪声；
    -轮廓连续性；
    -相机轨迹是否保持一致。

  判断标准：

  - 如果 20 步让 MSE 至少下降 5%、cosine 明显提高且视频更稳定，可以在最终推理中使用 20 步。
  - 如果改善不足 2–3%，继续加步数没有价值，应保持 10 步，把算力用于增加 Transformer blocks 和降低 spatial stride。
  - 如果 20 步反而噪声更多，说明模型预测的 vector field 不一致，增加积分步数只是在累积误差。

  训练阶段仍建议保持 10 步。20 步会使跨 turn BPTT 图接近翻倍，而下一阶段还需要测试 blocks=4 和 stride=4，不宜同时增加三个变量。

  最合适的顺序是：

  先测试 10 vs 20 步纯推理
  → 训练仍保持 10 步
  → blocks 1 → 4
  → stride 8 → 4
  → 最后再决定正式推理使用 10 或 20 步

  另外，VAE Decoder 本身没有“去噪步数”；这里增加的是 latent Flow Matching 的采样步数。

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


› 好，做纯体力消融，先告诉我  → blocks 1 → 4
    → stride 8 → 4里面的block和stride是什么意思


• 可以。这里的 block 和 stride 都是为了让 19-turn 训练先能在 H100 上运行而设置的资源缩减项；它们分别控制 Transformer 深度和视频 latent 的空间 token 密度。我先按当前代码的实际取
  样/重建方式确认一下细节，避免把它误解成视频帧 stride 或 VAE 下采样倍数。

• Explored
  └ Search spatial_token_stride|num_backbone_blocks|class Patch|patchify|unpatch|reconstruct|sampled in model.py
    Read model.py
    Read model.py

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 这里的 block 和 stride 是两个完全不同的模型容量参数，也都和去噪步数独立。

  ### Block：Transformer 的深度

  Wan2.1 基础模型共有 30 个 Transformer blocks。当前为了节省 19-turn BPTT 的显存和时间，只执行了第 1 个：

  for block in backbone.blocks[:num_backbone_blocks]:

  见 full_duplex/model.py:717。

  每个 block 都包含：

  - Full-Duplex causal self-attention；
  - RoPE 和 PRoPE；
  - 文本 cross-attention；
  - timestep conditioning；
  - FFN。

  所以：

  blocks=1：只执行第 1 层
  blocks=4：依次执行前 4 层
  blocks=30：执行完整 Wan2.1 Transformer

  增加 blocks 主要改善：

  - 动作、相机、历史状态之间的关系建模；
  - 文本语义；
  - 长时序一致性；
  - flow 预测质量。

  它不是“增加 3 个随机新层”。30 层权重都已经从原 checkpoint 严格加载，只是当前为了速度跳过了后面 29 层。当前 backbone 又是冻结的，所以执行 4 层不会增加需要训练的基础参数，但计
  算量和反向传播激活会增加。

  简单理解：

  block 数量 = 模型对当前 token 序列进行多少轮理解和加工

  ### Stride：latent 空间 token 的采样密度

  这里不是视频帧 stride，也不会改变 19 个动作或时间对齐。它只作用于每个 world latent 的空间网格。

  每个 latent state 是：

  [16, 60, 104]

  经过 2×2 patch embedding 后，得到：

  30 × 52 = 1560 个完整空间 tokens

  当前 stride=8 使用：

  embedded[:, :, 0, ::8, ::8]

  见 full_duplex/model.py:274。

  实际数量为：

  30×52 完整网格
  → 每隔 8 个位置取一个
  → 4×7
  → 28 tokens

  模型输出这 28 个 patch 后，先拼成约 8×14 的低分辨率 latent flow，再通过三线性插值恢复到 60×104。这也是当前视频容易出现大块纹理、细节不足的重要原因。

  改成 stride=4 后：

  30×52
  → 每隔 4 个位置取一个
  → 8×13
  → 104 tokens
  → 约 16×26 flow
  → 插值恢复到 60×104

  完整对比如下：

   Spatial stride    World tokens/turn    输出 flow 的有效空间网格
  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━
                8                   28                     约 8×14
  ────────────────  ───────────────────  ──────────────────────────
                4                  104                    约 16×26
  ────────────────  ───────────────────  ──────────────────────────
                2                  390                    约 30×52
  ────────────────  ───────────────────  ──────────────────────────
                1                 1560          60×104，无空间插值

  因此：

  stride 越小
  → token 越密
  → 空间细节越丰富
  → 序列越长
  → attention 计算和显存越大

  stride 从 8 降到 4，看起来只减半，但因为高度和宽度同时加密，world token 数从 28 增加到 104，接近 4 倍；attention 成本可能增长得更快。

  ### 三个参数的区别

  denoising steps：反复修正 latent 多少次
  blocks：每次修正时经过多少层 Transformer
  spatial stride：每次修正时能看到多少空间位置

  可以类比成：

  denoising steps = 修改次数
  blocks          = 每次思考的深度
  stride          = 画布的空间分辨率

  接下来的纯推理去噪消融不会改变 block 和 stride，只比较：

  blocks=1
  stride=8
  denoising steps=5/10/20/30

  这样才能单独判断“增加去噪次数”是否有效。

  之后再分别训练测试：

  实验 A：blocks=1，stride=8
  实验 B：blocks=4，stride=8
  实验 C：选择较好的 blocks，stride=4

  不能直接把现有 checkpoint 改成 stride 4 就判断效果，因为 stride 4 会启用很多训练时从未使用的新空间位置；这些位置需要经过短暂微调才公平。

  纯推理消融已经完成。结论很明确：增加去噪步数不能让当前模型变清晰。

   去噪步数    Latent MSE      Cosine    相机旋转误差
  ━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━━━━
          5      1.266417    0.415925          3.154°
  ──────────  ────────────  ──────────  ──────────────
         10      1.266472    0.413235          0.490°
  ──────────  ────────────  ──────────  ──────────────
         20      1.266896    0.410843          2.164°
  ──────────  ────────────  ──────────  ──────────────
         30      1.267201    0.409737          2.776°

  四档视频视觉上几乎没有差异，20/30 步的 cosine 和相机预测反而更差。因此保持 num_denoising_steps=10。

  视频可直接比较：

  - full_duplex/outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/denoising_ablation/prediction_steps_5.mp4
  - full_duplex/outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/denoising_ablation/prediction_steps_10.mp4
  - full_duplex/outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/denoising_ablation/prediction_steps_20.mp4
  - full_duplex/outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/denoising_ablation/prediction_steps_30.mp4

  下一步我建议：

  denoising_steps = 10 保持不变
  blocks: 1 → 4
  stride = 8 暂时保持

  具体顺序：

  1. 显式迁移 step-200 微调参数到 blocks=4，不能静默绕过 checkpoint 配置检查。
  2. 先跑 1 step，检查显存、loss、梯度和数值稳定性。
  3. 再跑 10 steps，重新生成视频。
  4. 如果 MSE/cosine 明显改善，继续到 50–100 steps。
  5. 粗粒度 flow 学好后，再将 stride 8 → 4 提升空间清晰度。

  world prior 的学习率建议从当前 0.01 降到 0.001–0.003，避免它继续放大颗粒纹理。

  目前不应直接改 stride 4，因为当前模型距离 stride-8 自身的低通参考仍很远，说明首先是只运行 1/30 Transformer blocks 导致的表达能力不足。先增加到 4 blocks，最有可能在不立即引爆
  序列长度的情况下改善人物结构、时序一致性和 flow 质量。

─ Worked for 6m 50s ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


› 这是因为我的资源太少只有单卡无法全量训练的原因吗


• 部分原因是单卡资源限制，但更准确地说，是“当前完整训练图太大”，并不是 H100 本身性能差。

  我们已经有实测证据：

   配置                                           结果
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   19 turns、stride 1、30 blocks、10-step BPTT    H100 80GB OOM，约 79GB 已占用
  ─────────────────────────────────────────────  ───────────────────────────────
   19 turns、stride 8、30 blocks                  44.18GB，约 161 秒/step
  ─────────────────────────────────────────────  ───────────────────────────────
   当前 stride 8、1 block                         11.93GB，约 13.4 秒/step

  完整训练之所以夸张，是因为同时包含：

  19 turns
  × 每 turn 10 次去噪
  × 全历史 token
  × 最长约 89,000 tokens
  × 30 Transformer blocks
  × 跨 turn 保留反向传播图

  即使 backbone 冻结，也仍要保存或重算经过这些层的激活，才能把梯度传回 camera/action/token encoder 和输出 head。

  当前噪点的直接原因

  资源限制迫使当前版本采用了两个很强的缩减：

  - blocks=1：只执行 Wan2.1 的 1/30 Transformer 深度；
  - stride=8：每个 latent 只保留 28 个空间 token，预测约 8×14 的 flow，再插值到 60×104。

  VAE 最后会把 60×104 放大到 480×832，因此 latent 中很小的误差、插值纹理和分布偏移都会被放大成明显噪点。

  相机更容易拟合，是因为相机只有 13 个数值，而且主要是低频、连续运动；world state 每 turn 有约 10 万个 latent 数值，难度完全不同。

  但单卡并不意味着无法继续提升

  我建议采用单卡友好的分阶段策略：

  阶段 1：19 turns、10 steps、stride 8、blocks 1→4
  阶段 2：保持 blocks=4，将 stride 8→4
  阶段 3：高分辨率使用 4-turn 短窗口训练
  阶段 4：定期回到 19-turn stride-8 做全局一致性训练

  这样分别学习：

  - 19-turn 粗分辨率训练：相机、动作、长期运动；
  - 短窗口高分辨率训练：人物轮廓、纹理、去噪和局部细节。

  实际 OOM 已经发生，因此后续高分辨率阶段可以合理使用 truncated BPTT，但仍保留周期性的完整 19-turn 梯度验证，不会完全退化成逐 turn teacher forcing。

  还可以加入：

  - gradient checkpointing，当前已经启用；
  - 只对最后若干 Transformer blocks 加 LoRA；
  - CPU activation offload；
  - 全分辨率局部 residual refiner；
  - 大部分步骤训练随机单个 sigma 的 flow loss，间隔执行完整 10-step state loss。

  如果增加 GPU，仅使用普通 DDP 没有明显帮助，因为 batch size 是 1，每张卡都会复制同一模型。需要 sequence/context parallel 或 pipeline parallel，才能真正分摊这条超长序列。

  所以结论是：当前质量确实间接受单卡限制影响，但不是必须换机器才能改善。下一步最合理的实验仍然是保持 10-step 和 stride 8，将 blocks=1 → 4，先利用更多已经加载好的 Wan2.1 预训练
  层。然后再用 stride 4 提升空间清晰度。


› 训练代码是哪一个？我能不能用参数直接指定step，stride 和block的数量？这三者的含义注意在注释里面提出来


• 可以直接通过命令行指定，但 step 有两种含义，必须分开：--max-steps 是优化器更新次数，--num-denoising-steps 是每个 micro-turn 内的 Flow/Euler 去噪次数。我会把这一区别，以及
  block、stride 的真实网格含义，补到训练入口的 --help 和 YAML 注释里，并验证参数确实传到训练器。

• Explored
  └ Read control_training.py
    Read train_overfit.py
    Read overfit.yaml
    Read config.py
    Search max_steps|num_denoising_steps|num_backbone_blocks|spatial_token_stride in README.md
    Read training.py, README.md


    
• 可以直接通过命令行指定。推荐训练入口是同步监控脚本 full_duplex/control_training.py:24，它会调用 full_duplex/train_overfit.py:10，核心训练循环在 full_duplex/training.py。

  参数含义：

   参数                      含义
  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   --max-steps               目标 global optimizer step。新训练中是更新次数；从 step 200 恢复并指定 250，只会再跑 50 次
  ────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────
   --num-denoising-steps     每个 micro-turn 内的 Flow/Euler 去噪次数
  ────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────
   --blocks                  每次模型调用执行前多少个 Wan Transformer blocks，范围 1–30
  ────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────
   --spatial-token-stride    latent patch 空间采样间隔，不是视频帧 stride

  例如，新建一次 4 blocks + stride 8 + 10-step denoising 的 10 次更新测试：

  eval "$(conda shell.bash hook)"
  conda activate /hyperai/home/conda_envs/minwm
  cd /hyperai/home/minwm

  python -u full_duplex/control_training.py \
    --config full_duplex/configs/overfit.yaml \
    --mode rollout \
    --run-name rollout_b4_s8_probe10 \
    --max-steps 10 \
    --learning-rate 1e-4 \
    --world-head-learning-rate-multiplier 3 \
    --world-prior-learning-rate-multiplier 30

  也可以使用更明确的别名：

  --num-backbone-blocks 4

  完整帮助：

  python -m full_duplex.control_training --help

  注释已补充到：

  - full_duplex/control_training.py:29
  - full_duplex/train_overfit.py:14
  - full_duplex/configs/overfit.yaml:15
  - full_duplex/model.py:172
  - full_duplex/README.md

  另外修复了一个记录问题：命令行覆盖 --max-steps 后，现在会正确写入 run manifest 和 checkpoint 的 training_config。

  需要注意：新训练可以直接改变 blocks/stride；普通 --resume 会拒绝这些参数与 checkpoint 不一致。也就是说，不能直接用当前 blocks=1 的 best checkpoint 加 --blocks 4 假装普通续
  训。若要继承当前 step-200 权重，需要增加一个显式、可审计的 warm-start/架构迁移入口。当前代码会主动报错，避免静默错误加载。

  语法、CLI 帮助和 8 项回归测试均已通过。

  gpt-5.6-sol max fast · /output/minwm