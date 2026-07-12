# Full-Duplex Wan2.1 micro-turn fine-tuning

This directory contains the latent training implementation requested in
`prompt.md`. The training graph never calls the VAE decoder; the separate
evaluation-only `decode_predictions.py` utility can export predicted RGB video.

## Verified sample contract

The real sample has 77 RGB frames. Wan VAE temporal chunks are `1,4,4,...`,
which produce 20 cached states `[20,16,60,104]`. The checked manifest contains
19 transitions, not 11: eight `right`, then eleven `a`. Turn `t` predicts state
`t+1`; turn 0 receives a zero latent plus `NULL_WORLD_STATE`. This discrepancy
is preserved in `cache/smallest_000000/metadata.json` instead of being hidden.

## Main files

- `preencode.py`: real VAE/T5/camera/action preprocessing, identity hashes,
  invalidation, numerical checks, and bitwise reload verification.
- `tokens.py`: stable special-token IDs, exact stream protocol, spans,
  prediction mask, and causal visibility mask.
- `model.py`: strict base loading; explicit world/camera/action/noise tokens;
  type/turn embeddings; checkpoint 3D RoPE, PRoPE, self/cross attention, FFN,
  timestep conditioning and world output head; new camera encoder/head; optional
  zero-initialized parallel world residual head and optional zero-initialized
  time×spatial flow prior for small-data adaptation.
- `flow.py`: checkpoint-compatible Flow Matching interpolation, target and
  differentiable Euler step.
- `lora.py`, `train_lora.py`: zero-initialized low-rank adapters for the
  physical final Wan blocks; explicit pre-LoRA warm-start, exact resume and
  self-contained adapter checkpoints.
- `training.py`: fixed-noise 10-step denoising, autoregressive rollout without
  detach, cross-turn BPTT probe, losses, finite/gradient checks, checkpointing,
  resume, and exact reload test.
- `control_training.py`: foreground synchronous worker controller with a raw
  log and atomically updated status/summary files. Console progress is compact;
  the raw log retains every per-turn metric. Interrupts terminate the worker.
- `predict_checkpoint.py` and `evaluate_predictions.py`: deterministic
  latent/camera rollout and per-turn latent/camera evaluation; no RGB decode.
- `visualize_mask.py`, `tests/`: readable mask artifact and unit tests.

## Reproducible commands

Use the requested environment directly (equivalent to activating it):

```bash
export PYTHONPATH=/output/minwm:/output/minwm/Wan21:/output/minwm/shared
PY=/hyperai/home/conda_envs/minwm/bin/python

$PY full_duplex/audit_checkpoint.py --config full_duplex/configs/overfit.yaml
$PY full_duplex/preencode.py --config full_duplex/configs/overfit.yaml
$PY -m unittest discover -s full_duplex/tests -v
$PY full_duplex/visualize_mask.py
```

## Training controls

Use `control_training.py` for monitored foreground training. It launches
`train_overfit.py`, whose `FullDuplexTrainer` implementation is in
`training.py`. Four independent command-line controls must not be conflated:

| CLI option | Meaning |
|---|---|
| `--max-steps` | Target global optimizer step; fresh run = update count, resume = continue to this step |
| `--num-denoising-steps` | Differentiable Flow/Euler updates inside every micro-turn |
| `--blocks` / `--num-backbone-blocks` | Leading pretrained Wan Transformer blocks executed per model call |
| `--spatial-token-stride` | Spatial sampling interval on the latent patch grid; not frame stride |

For example, this requests 100 optimizer updates, 10 denoising updates per
turn, four Transformer blocks, and stride-8 spatial tokens:

```bash
$PY -u full_duplex/control_training.py \
  --mode rollout --run-name rollout_100step_10denoise_4block_stride8 \
  --max-steps 100 --num-denoising-steps 10 \
  --blocks 4 --spatial-token-stride 8 \
  --freeze-backbone --attention-pad-to-turns 19
```

These flags override the YAML for a new run and the effective values are saved
to the run manifest and checkpoint. Ordinary `--resume` deliberately rejects
changes to denoising steps, blocks, or stride; an architecture change requires
an explicit warm-start/migration rather than silently treating it as the same
training run.

## Last-block LoRA

The dedicated LoRA path executes all 30 checkpoint blocks and can adapt only
the physical final N blocks. By default the existing Full-Duplex task delta is
loaded and frozen, all original Wan parameters remain frozen, and only LoRA
A/B matrices enter the optimizer:

```bash
$PY -u -m full_duplex.train_lora \
  --warm-start RUN/checkpoints/best.pt \
  --run-name lora_last4_rank8_rollout19 \
  --max-steps 10 --num-turns 19 --num-denoising-steps 10 \
  --num-backbone-blocks 30 --spatial-token-stride 8 \
  --lora-last-blocks 4 --lora-rank 8 --lora-alpha 8 \
  --learning-rate 1e-4 --attention-pad-to-turns 19
```

`--train-task-modules` optionally updates the Full-Duplex embeddings/heads
alongside LoRA while still freezing every original Wan parameter. Exact code,
parameter manifests, commands and the first real-data result are documented in
`LORA_REPORT.md`.

Full-fidelity single-turn overfit:

```bash
$PY -u full_duplex/train_overfit.py \
  --mode single --run-name single_full_100 --max-steps 100 --freeze-backbone
```

The native-token 19-turn graph was tried first and produced a real 80-GB H100
OOM. A 30-layer, stride-8, all-history, 19-turn, 10-denoise-step run fits and is
used as the high-fidelity rollout smoke test:

```bash
$PY -u full_duplex/train_overfit.py \
  --mode rollout --run-name rollout_19turn_stride8_30blocks_fixedpad_1step \
  --max-steps 1 --freeze-backbone --spatial-token-stride 8 \
  --attention-pad-to-turns 19
```

The staged 100-step functional overfit keeps all 19 turns, all history,
10-step differentiable denoising, RoPE, PRoPE, text cross-attention and BPTT,
but executes one real checkpoint block so it can be monitored in reasonable
time. The exact reduction is stored in every checkpoint:

```bash
$PY -u full_duplex/control_training.py \
  --mode rollout --run-name rollout_19turn_stride8_1block_100step \
  --max-steps 100 --freeze-backbone --spatial-token-stride 8 --blocks 1 \
  --attention-pad-to-turns 19
```

If the frozen checkpoint head reduces MSE mainly by shrinking prediction
variance but latent cosine remains weak, enable the compatible parallel world
head. It is zero initialized, so the base head remains the exact starting
function; only its residual is learned and saved in the explicit delta:

```bash
$PY -u full_duplex/control_training.py \
  --mode rollout --run-name rollout_19turn_stride8_1block_worldhead_100step \
  --max-steps 100 --freeze-backbone --spatial-token-stride 8 --blocks 1 \
  --attention-pad-to-turns 19 --world-residual-head --learning-rate 1e-4
```

For the staged continuation, the residual world head can use its own explicit
optimizer group while all other new modules retain the base learning rate.
Resuming with a changed LR requires the opt-in flag; the loader migrates Adam
moments without silently dropping optimizer state and records both rates:

```bash
$PY -u full_duplex/control_training.py \
  --mode rollout --run-name rollout_19turn_stride8_1block_worldhead_final500 \
  --max-steps 500 --resume RUN/checkpoints/best.pt --freeze-backbone \
  --spatial-token-stride 8 --blocks 1 --attention-pad-to-turns 19 \
  --world-residual-head --learning-rate 1e-4 \
  --world-head-learning-rate-multiplier 3 \
  --override-resume-learning-rate
```

`--train-base-world-head` is a separately logged ablation switch. It only
unfreezes the checkpoint's timestep-conditioned output head while leaving the
Transformer frozen; it is not enabled in the selected stable run.

The optional `--world-time-space-prior` adds a trainable flow bias indexed only
by the visible turn index and output patch coordinate. It never reads target
content. Its table is zero initialized while preserving the global RNG state,
so enabling it leaves the step-0 model function unchanged. Its separate LR
group is controlled by `--world-prior-learning-rate-multiplier`; the feature is
disabled in the base configuration and must be enabled explicitly.

Evaluate the actual best checkpoint parameters, rather than stale pre-update
training tensors:

```bash
$PY full_duplex/predict_checkpoint.py \
  --checkpoint RUN/checkpoints/best.pt --output RUN/best_predictions.pt
$PY full_duplex/evaluate_predictions.py \
  --predictions RUN/best_predictions.pt --checkpoint RUN/checkpoints/best.pt \
  --output RUN/evaluation.json
$PY full_duplex/summarize_metrics.py --metrics RUN/metrics.jsonl
```

Export a freshly generated latent rollout through the frozen real Wan2.1 VAE
(this is intentionally separate from the training graph):

```bash
$PY -m full_duplex.predict_checkpoint \
  --checkpoint RUN/checkpoints/best.pt \
  --output RUN/video_export/predicted_latents.pt
$PY -m full_duplex.decode_predictions \
  --predictions RUN/video_export/predicted_latents.pt \
  --vae-checkpoint ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
  --project-root /output/minwm \
  --output RUN/video_export/prediction.mp4 --fps 24 --crf 18
```

The decoder asserts the exact latent and RGB shapes, rejects NaN/Inf, writes a
contact sheet and a JSON provenance manifest, and does not prepend or decode a
ground-truth context state. Nineteen predicted latent states therefore produce
73 frames under Wan's `1 + 4 * (T - 1)` temporal decode rule.

For a pure inference sampler ablation, `predict_checkpoint` can replace only
the Euler sigma grid after the original checkpoint/config compatibility checks
have succeeded:

```bash
$PY -m full_duplex.predict_checkpoint \
  --checkpoint RUN/checkpoints/best.pt \
  --num-denoising-steps 20 \
  --output RUN/denoising_ablation/predictions_steps_20.pt
```

The output manifest records both the trained and inference step counts and the
complete sigma grid. Checkpoint weights and fixed initial noise remain
unchanged.

## Attention padding

FlexAttention recompiles for each tensor length. A 19-turn rollout initially
exhausted Dynamo's specialization limit and fell back to an unfused dense score
matrix. `attention_padding_strategy: power_of_two` reduces buckets, and the
validation command pads to the masked 19-turn maximum (`2048` at stride 8), so
one compiled shape is reused. Padding tokens have turn `-1`, are invalid keys,
and only retain a private diagonal for numerical safety; they cannot leak data.

## Freeze policy

VAE and UMT5 are pre-encoded and absent from the training graph. The fidelity
default allows the generator backbone and new modules to train. The reported
small-data runs explicitly freeze the 1.49B generator and train the new special,
type, turn and action embeddings plus camera encoder/head; this decision and all
parameter counts are written to `run_manifest.json`. A separate full-backward
smoke test proves gradients through the checkpoint backbone path.
