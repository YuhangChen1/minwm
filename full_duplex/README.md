# Full-Duplex Wan2.1 micro-turn fine-tuning

This directory contains the latent-only implementation requested in `prompt.md`.
It never calls the VAE decoder and does not reconstruct RGB video.

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
  zero-initialized parallel world residual head for small-data adaptation.
- `flow.py`: checkpoint-compatible Flow Matching interpolation, target and
  differentiable Euler step.
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
