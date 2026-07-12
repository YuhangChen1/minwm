# Full-Duplex minWM implementation and validation report

This report covers latent-space training only. No VAE decoder or RGB recovery
is called by the implementation or evaluation tools.

## Audited ground truth

- Environment: `/hyperai/home/conda_envs/minwm`, PyTorch 2.9.1+cu128,
  NVIDIA H100 80GB, bf16 supported.
- Base checkpoint: `ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt`, top-level
  key `generator`, 885 fp32 tensors and 1,489,821,760 elements. Strict load has
  zero missing/unexpected keys and a loaded ratio of 1.0. A real cached-input
  forward is finite.
- Data: 77 RGB frames at 24 fps and 832x480. The real semantic manifest has 19
  actions (eight `right`, eleven `a`), not 11 total actions. Frames 1..76 form
  19 contiguous four-frame transitions; frame 0 is initial context.
- VAE cache: 20 states `[20,16,60,104]` float16. Turn `t` targets state `t+1`;
  turn 0 receives an all-zero state plus `NULL_WORLD_STATE`.
- Text cache: frozen UMT5-XXL `[512,4096]` bf16, 132 non-padding tokens and
  zero padding rows. The embedding is projected once per training step and
  reused in cross-attention.
- The sample has no measured camera file. Camera targets are deterministic
  repository-generated OpenCV w2c trajectories from the real action sequence,
  encoded as translation + 6D rotation + four intrinsics (13 values). Camera
  metrics therefore measure fitting of that repository trajectory, not a
  physical sensor trace.

Full evidence and original source locations are in `AUDIT.md`, while exact
identity hashes and alignment are in the cache `metadata.json` and
`outputs/smallest_000000/data_audit.json`.

## Implemented training path

- Exact ordered input/output stream boundaries, independent special-token
  embeddings, `TIME_INDEX_0..31`, visible boundary tokens, masked world/camera
  output content, and a separate prediction mask.
- One explicit camera token and action token per turn. Explicit camera tokens
  are separate from PRoPE w2c/K geometry.
- Checkpoint 3D RoPE, PRoPE, self-attention, text cross-attention, FFN,
  timestep conditioning, patch embedding and compatible world head are reused.
- A default-off, zero-initialized time×spatial flow prior is available as a
  small-data adaptation. It is indexed only by visible turn/patch coordinates,
  never by target tensors, and its construction preserves the RNG stream so
  the enabled/disabled models have the same step-0 function.
- Checkpoint-compatible Flow Matching uses
  `x_sigma=(1-sigma)*x0+sigma*epsilon`, target `epsilon-x0`, and a
  differentiable decreasing-sigma Euler update.
- Each turn uses one seeded initial noise tensor for all ten denoising steps.
  Noise identity is recorded by SHA-256 and mutation is asserted against.
- Rollout feeds predicted state and camera directly to the next turn, with no
  teacher forcing and no detach. A future-only gradient probe proves turn 0
  receives nonzero gradient from later-turn losses.
- Checkpoints contain the trainable delta over a freshly strict-loaded base,
  optimizer and scheduler state, RNG states, full configs, token/action
  vocabularies, camera representation, cache hash and base identity. Allowed
  base missing keys during delta reload are enumerated exactly; any other
  missing or unexpected key raises.
- The foreground controller records full per-turn JSON to a raw log, emits
  compact progress, updates status atomically, terminates its worker on an
  interrupt, maintains `best.pt`/`latest.pt`, and performs an exact fresh-model
  reload probe after training.

## Fidelity and resource validation

- Native 19-turn, stride-1, 30-layer, ten-step BPTT was attempted first and
  reached a real H100 OOM at 79.09 GiB in use. The exception is preserved in
  `outputs/smallest_000000/rollout_full_fidelity_19turn_1step.log`.
- The same 19 turns/all history/ten denoising steps at stride 8 and all 30
  checkpoint layers completed one forward/backward step: 44.18 GiB peak,
  160.94 seconds, finite loss, and future-turn gradient norm 0.2614.
- Full spatial fidelity and all 30 layers completed a 100-step single-turn
  overfit: total loss 3.6066 to 2.9379 and state loss 1.8916 to 1.4558.
- The monitored multi-turn overfit retains all 19 turns, all historical
  predictions, ten differentiable denoising steps, RoPE, PRoPE, text
  cross-attention and BPTT. To make hundreds of steps feasible it explicitly
  uses stride 8 and one real checkpoint block; every checkpoint records those
  reductions. The frozen 1.49B backbone is strict-loaded and 4,995,175 new
  parameters are trained, including a zero-initialized compatible residual
  world head.

## Final staged overfit metrics

Selected checkpoint:
`outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/checkpoints/best.pt`.
It is step 200 and is identical in content to the named latest checkpoint.
The exact fresh-model reload probe reports zero maximum absolute error for
both flow and camera outputs.

Training metrics (one real sample, fixed noise):

| Metric | Step 1 | Step 200 |
|---|---:|---:|
| total loss | 5.706124 | 2.534786 |
| state loss | 2.755545 | 1.266439 |
| flow loss | 2.865153 | 1.268220 |
| camera loss | 0.085427 | 0.000127 |

Total loss decreased by 55.58%; 92.46% of logged transitions decreased. The
future-only turn-0 gradient norm was 0.084259, maximum allocated GPU memory
was 11.93 GiB, and mean optimizer-step time was 13.43 seconds. The final model
has 8,190,055 trainable delta parameters over the strict 1.49B base, including
a `[49920,64]` zero-started time×spatial prior. Its staged prior LR was 0.01
for steps 1–10, 0.003 for steps 11–100, then 0.01 for steps 101–200; every
stage and optimizer moment is represented in the checkpoint history.

Fresh 19-turn checkpoint rollout (not stale training tensors):

- overall latent MSE: 1.266472; cosine similarity: 0.413235;
- prediction mean/std: 0.102214 / 1.170548; target: 0.098777 / 0.828415;
- camera translation L2: 0.012891; rotation: 0.489831 degrees;
  intrinsics RMSE: 0.007636;
- best state-MSE turn: 0 (MSE 1.204913, cosine 0.411986);
- worst state-MSE turn: 18 (MSE 1.294794, cosine 0.399385);
- RGB decoder used for these latent metrics: false.

The selected step-200 checkpoint improves on the non-prior step-500 candidate
(MSE 1.315440, cosine 0.382559, rotation 1.066 degrees) with fewer steps.
The loss curve and exact aggregate/per-turn values are `loss_curve.png`,
`loss_history.csv`, `per_turn_loss_history.csv`, and `evaluation.csv` in the
selected run directory.

All “best” values are training-set overfit metrics because the minimal dataset
contains one sample. Prediction evaluation is re-run from a fresh model loaded
from the selected checkpoint; it does not reuse stale pre-update tensors.

## Remaining scientific limitations

- The hundreds-step validation profile is a resource-reduced proxy, not a
  claim that the native 89,187-token, 30-layer training graph fits in 80GB.
- Stride 8 reconstructs a 60x104 latent from an 8x14 patch grid. The report
  therefore records both full latent error and the attainable low-pass
  projection reference.
- One-sample overfitting verifies wiring, gradients, checkpointing and
  optimization; it does not demonstrate generalization.
- Camera ground truth is deterministic action-derived geometry because no
  measured camera sequence exists in this sample.
- The selected prediction is materially closer to the target than at step 1,
  but its MSE remains above the zero-latent baseline (0.696027) and above the
  stride-8 low-pass reference (0.376614). Therefore the strict scientific claim
  `predicted latent ≈ ground truth` is not yet established; this report does not
  mark that acceptance item complete.

## On-demand RGB prediction export

After the latent-only training scope was completed, the selected step-200
checkpoint was freshly loaded for an explicit video export. The newly produced
rollout is bitwise identical to the earlier fresh-checkpoint rollout (state
tensor SHA-256
`21e2475afff0eb2c8793ea5f59ac3a00fe6662c7198ea9c9d582594d251e6033`).
No ground-truth state is used by the prediction rollout or decoded into the
result.

The frozen real Wan2.1 VAE checkpoint, SHA-256
`38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`,
decodes the 19 predicted states to 73 RGB frames according to
`1 + 4 * (19 - 1)`. The verified output is H.264/yuv420p, 832x480, 24 FPS,
73 readable frames, and 3.0417 seconds long:

`outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/video_export/prediction_step_000200.mp4`

The adjacent JSON contains the exact checkpoint, fixed-noise identity, VAE
identity, shapes, latent metrics, runtime, and GPU-memory evidence. Visual
inspection of the contact sheet shows colored high-frequency/block artifacts
without a recognizable scene. Thus this artifact proves the real end-to-end
load/rollout/VAE-decode/video-write path, but it also confirms that the current
resource-reduced overfit model has not reached useful RGB prediction quality.

## Inference-only denoising-step ablation

The step-200 checkpoint was evaluated at 5, 10, 20, and 30 Euler steps.
Model weights, `blocks=1`, spatial stride 8, all 19 turns, action sequence, VAE,
and fixed initial-noise SHA-256 were held constant. Only the inference sigma
grid changed after strict checkpoint reload.

| Steps | Latent MSE | Cosine | Predicted std | Camera translation L2 | Camera rotation (deg) |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.266417 | 0.415925 | 1.173815 | 0.049090 | 3.154300 |
| 10 | 1.266472 | 0.413235 | 1.170548 | 0.012891 | 0.489831 |
| 20 | 1.266896 | 0.410843 | 1.167856 | 0.069755 | 2.164040 |
| 30 | 1.267201 | 0.409737 | 1.166660 | 0.094575 | 2.776021 |

All four variants decode to valid 73-frame, 832x480, 24 FPS videos. Their
contact sheets are visually almost indistinguishable. Extra Euler steps do not
reduce world-state error and progressively reduce cosine similarity; both
fewer and more than the trained 10-step grid materially degrade camera error.
The selected inference setting therefore remains 10 steps. The artifacts are
not attributable to insufficient sampler iterations.

## Physical-last-block LoRA experiment

A separate LoRA path now strict-loads the base, warm-starts and freezes the
selected 8,190,055-element Full-Duplex task delta, executes all 30 Wan blocks,
and trains rank-8 adapters only in blocks 26..29. Forty attention/cross-attention
and FFN Linear modules contribute 1,458,176 trainable elements; every original
Wan and task-delta element remains frozen. The self-contained 49-MiB checkpoint
reloads with exact flow/camera output parity.

The real 19-turn, ten-denoise-step run fits at 43.42 GiB and about 100–107
seconds per optimizer step. From step 1 to 10, total loss decreases
14.598776→12.106856 and state loss 7.637802→6.351964. A fresh post-update
rollout has MSE 6.110075 and cosine 0.093171, however, versus 1.266472/0.413235
for the selected one-block baseline; camera translation/rotation also regress
to 0.498396/8.133980 degrees. The decoded video is visibly noisier.

Therefore LoRA plumbing and overfit gradients are validated, but this strict
LoRA-only architecture migration is not a quality win and is stopped at 10
steps rather than extended blindly. `LORA_REPORT.md` contains the full evidence
and identifies frozen task-module/full-depth hidden-distribution mismatch as
the next issue to address.
