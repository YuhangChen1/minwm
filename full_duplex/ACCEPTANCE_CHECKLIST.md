# `prompt.md` acceptance checklist

Status is based on code artifacts and commands that actually ran. `PASS` does
not infer facts that were unavailable in the sample.

| Acceptance item | Status | Evidence |
|---|---|---|
| Strict-load original `ar_diffusion_tf` checkpoint | PASS | `checkpoint_audit.json`: 885 tensors, missing/unexpected 0/0, ratio 1.0, finite real forward |
| Important base weights loaded | PASS | strict second load plus exact 1,489,821,760-element count |
| Real smallest-data preprocessing | PASS | 77 RGB frames → 20 real VAE latents; frozen real T5 embedding |
| Cache invalidation and repeat load | PASS | checkpoint/source hashes and `preencode_cache_hit.log`; tensor reload bitwise equal |
| Action/turn alignment | PASS with corrected count | real manifest is right×8 + a×11 = 19 transitions, not 11 total; metadata maps state `t → t+1` |
| Exact Full-Duplex stream order and special tokens | PASS | `tokens.py`, `full_sequence_layout.json`; 41 independent IDs/embeddings |
| No current-GT/future mask leakage | PASS | `test_tokens.py`, `mask_visualization.txt`, FlexAttention mask assertions |
| Frozen cached T5 cross-attention on every executed turn/block | PASS | padding rows zero, context projected once/step, Q=stream and K/V=text |
| Turn 0 zero/null state | PASS | zero latent plus distinct `NULL_WORLD_STATE` span/token |
| Later turns consume prior predictions | PASS | runtime grad-fn assertion; no ground-truth input replacement |
| No unintended detach across turns | PASS | runtime assertion and future-only turn-0 gradient norm 0.084259 |
| Fixed seeded initial noise and 10-step denoising | PASS | SHA `03c127...02ec`, mutation assertion, 10 decreasing sigma/Euler steps |
| Flow/state/camera losses backpropagate | PASS | `gradient_audit.json`: all requested paths finite and nonzero |
| Single-turn 100-step overfit decreases | PASS | total 3.606590→2.937884, state 1.891568→1.455826 |
| Full 19-turn 100+ step loss decreases | PASS | selected run total 5.706124→2.534786 by step 200 |
| Checkpoint best/latest/save/resume/reload | PASS | best step 200, complete optimizer/RNG/config metadata, reload max errors 0/0 |
| Camera prediction approaches target | PASS for repository trajectory | translation L2 0.012891, rotation 0.489831°, intrinsics RMSE 0.007636 |
| Latent prediction clearly trends toward target | PASS for trend | MSE 2.755545→1.266472 and cosine to 0.413235 |
| Strict `predicted latent ≈ ground truth` claim | NOT YET ESTABLISHED | final MSE 1.266472 remains above zero baseline 0.696027 and low-pass reference 0.376614 |
| No RGB/VAE decoder used to hide latent errors | PASS | evaluation artifact records `rgb_decoder_used: false`; decoder absent from training/evaluation |
| Physical-last-block LoRA path | PASS (experimental) | blocks 26..29, rank 8, 1,458,176 LoRA-only trainables; strict warm-start/resume/reload; 11/11 tests |
| LoRA-only quality improvement | NOT ACHIEVED | step-10 loss falls 17.07%, but fresh MSE/cosine 6.110075/0.093171 are worse than the selected baseline |

## Resource qualification

The full conceptual stream has 89,187 tokens. Native stride-1, 30-block,
19-turn BPTT was attempted first and OOMed on the 80GB H100 at 79.09 GiB in
use. A 30-block stride-8 19-turn backward passed at 44.18 GiB. Monitored
hundreds-step runs therefore retain all 19 turns/history and 10-step BPTT but
explicitly use stride 8 and one real checkpoint block; these values are stored
inside every checkpoint and are not presented as native-fidelity training.
