# Last-block LoRA experiment

## Scope and starting point

This experiment asks whether the strict Wan base can remain frozen while only
low-rank adapters in its physical final Transformer blocks are optimized. It
warm-starts the selected Full-Duplex task delta from:

`outputs/smallest_000000/rollout_19turn_stride8_1block_worldprior_final200/checkpoints/best.pt`

That source task delta was trained with one executed backbone block and spatial
stride 8. Its 21 task tensors (8,190,055 elements) are loaded exactly and
frozen for the default LoRA-only experiment. Optimizer/RNG/global-step state is
not inherited; the warm-start report records this architecture migration.

## Implementation

- `lora.py` implements a frozen `nn.Linear` plus
  `scaling * B(A(dropout(x)))`. `A` uses Kaiming initialization and `B` is
  exactly zero, so injection preserves the pretrained function at step zero.
- The complete 30-block Wan path is executed. `lora_last_blocks=4` selects
  physical block indices 26, 27, 28 and 29, not the end of a reduced prefix.
- Ten affine paths are adapted per selected block: self-attention q/k/v/o,
  text cross-attention q/k/v/o and FFN layers 0/2.
- Rank 8 and alpha 8 produce 1,458,176 trainable LoRA elements across 40
  wrapped Linear modules. They cover 185,689,088 frozen base elements.
- All 1,498,011,815 non-LoRA elements are frozen, including the warm-started
  Full-Duplex embeddings, camera modules, residual head and time-space prior.
- Checkpoints use
  `full_duplex_lora_delta_over_strict_base`. They contain both the frozen task
  delta and LoRA matrices (101 tensors, 9,648,231 elements), while the optimizer
  contains only LoRA parameters. A fresh model therefore reloads independently
  from the strict base without relying on the warm-start file.
- Ordinary LoRA resume restores optimizer/global step. Pre-LoRA warm-start and
  resume are explicit, mutually exclusive operations.

Unit tests prove exact zero-initial function preservation, physical-last-block
selection, loud failure for invalid targets, and LoRA gradient boundaries. The
complete suite passes 11/11 tests.

## Commands

Single-turn real-data smoke test:

```bash
python -u -m full_duplex.train_lora \
  --warm-start RUN_1BLOCK/checkpoints/best.pt \
  --run-name lora_last4_rank8_singleturn_smoke1 \
  --max-steps 1 --num-turns 1 --num-denoising-steps 10 \
  --num-backbone-blocks 30 --spatial-token-stride 8 \
  --lora-last-blocks 4 --lora-rank 8 --lora-alpha 8 \
  --learning-rate 1e-4 --attention-pad-to-turns 1
```

Full 19-turn smoke and continuation:

```bash
python -u -m full_duplex.train_lora \
  --warm-start RUN_1BLOCK/checkpoints/best.pt \
  --run-name lora_last4_rank8_rollout19_smoke1 \
  --max-steps 1 --num-turns 19 --num-denoising-steps 10 \
  --num-backbone-blocks 30 --spatial-token-stride 8 \
  --lora-last-blocks 4 --lora-rank 8 --lora-alpha 8 \
  --learning-rate 1e-4 --attention-pad-to-turns 19

python -u -m full_duplex.train_lora \
  --resume RUN_LORA_1/checkpoints/latest.pt \
  --run-name lora_last4_rank8_rollout19_10step --max-steps 10
```

## Real-data results

Single turn, one optimizer step:

- total/state/flow/camera: 5.006008 / 2.701054 / 2.304622 / 0.000332;
- LoRA gradient norm: 0.874745;
- peak allocated GPU memory: 5.97 GiB;
- elapsed time: 11.22 seconds;
- checkpoint fresh-reload flow/camera max error: 0 / 0.

Full 19-turn, all-history, ten-denoise-step BPTT:

| Metric | Zero-LoRA step 1 input/output | Step 10 input/output |
|---|---:|---:|
| total loss | 14.598776 | 12.106856 |
| state loss | 7.637802 | 6.351964 |
| flow loss | 6.829523 | 5.614003 |
| camera loss | 0.131452 | 0.140890 |
| gradient norm before clipping | 3.162993 | 7.653405 |
| peak allocated GPU memory | 43.42 GiB | 43.36 GiB |
| elapsed time per step | 107.10 s | 100.47 s |

Total and state training losses decrease by 17.07% and 16.84%, respectively.
The step-1 future-only turn-0 gradient norm is 0.109254, proving that the LoRA
path participates in the existing cross-turn BPTT graph. The step-10 fresh
reload test again reports exact zero flow/camera differences.

Fresh post-update 19-turn evaluation of the step-10 best checkpoint:

- state MSE: 6.110075;
- latent cosine: 0.093171;
- prediction mean/std: 0.471481 / 2.359191 (target std 0.828415);
- camera translation L2: 0.498396;
- camera rotation: 8.133980 degrees;
- camera intrinsics RMSE: 0.110747.

The comparison 1-block step-200 checkpoint has MSE 1.266472, cosine 0.413235,
translation L2 0.012891 and rotation 0.489831 degrees. The LoRA step-10 video is
valid H.264, 73 frames, 832x480 at 24 FPS, but visual inspection shows stronger
green high-frequency noise than the existing baseline.

## Decision

The adapters are implemented correctly and can overfit their objective, but
this specific LoRA-only continuation is not a quality improvement. The
dominant issue is an architecture-distribution migration: the frozen task
embeddings/heads/prior were learned from block-0 hidden states, then are asked
to consume block-29 hidden states. Last-four-block LoRA alone must first undo
that mismatch, and the camera objective is already regressing while world loss
dominates.

The run is deliberately stopped at 10 rather than blindly extended to 100.
The scientifically useful next variants are:

1. warm-start the same task delta but jointly train LoRA plus task-specific
   Full-Duplex modules for 10 steps, leaving all original Wan weights frozen;
2. or first train the task modules on the full 30-block path, then freeze them
   and perform the strict LoRA-only ablation;
3. only after the full-depth hidden-distribution mismatch is resolved, reduce
   spatial stride from 8 to 4 for detail.

The second option is the cleanest test of “only last-block LoRA”; the first is
the fastest likely path toward better visible fitting quality.
