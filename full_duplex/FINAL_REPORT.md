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

<!-- FINAL_METRICS -->

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
