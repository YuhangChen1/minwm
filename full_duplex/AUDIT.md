# Wan2.1 / minWM code audit

This audit records only conclusions verified from the checked-out code or real files. Runtime shapes are appended by `preencode.py` and `audit_checkpoint.py` logs.

| Item | Evidence | Verified input → output | dtype / conclusion |
|---|---|---|---|
| VAE encoding | `Wan21/wan_utils/wan_wrapper.py`, `WanVAEWrapper.encode_to_latent`, lines 73–87; `Wan21/wan/modules/vae.py`, `WanVAE_.encode`, lines 517–543 | pixel `[B,3,F,H,W]` → normalized latent `[B,F,16,H/8,W/8]` | wrapper emits float32; cache follows original preprocessing and stores float16 |
| VAE normalization | `Wan21/wan_utils/wan_wrapper.py`, lines 56–77; `Wan21/wan/modules/vae.py`, lines 537–541 | `mu → (mu-channel_mean)/channel_std` | 16 fixed float32 means/stds; normalization occurs inside encode exactly once |
| Temporal VAE alignment | `Wan21/wan/modules/vae.py`, `WanVAE_.encode`, lines 520–535 | RGB chunks `1,4,4,...` → one latent per chunk | 77 RGB frames → 20 latent states |
| Patch embedding | `Wan21/wan/modules/causal_model.py`, `CausalWanModel.__init__`, lines 680–682; `_forward_train`, lines 1195–1207 | `[B,16,F,H,W]` → `[B,F*(H/2)*(W/2),1536]` | checkpoint weight float32 `[1536,16,1,2,2]`; training casts model to bf16 |
| UMT5-XXL | `Wan21/wan/modules/t5.py`, `umt5_xxl`, lines 456–469; `Wan21/wan_utils/wan_wrapper.py`, `WanTextEncoder.forward`, lines 37–50 | token IDs/mask `[B,512]` → prompt embedding `[B,512,4096]` | frozen; padding rows explicitly zeroed at wrapper lines 45–46 |
| Text injection | `Wan21/wan/modules/model.py`, `WanT2VCrossAttention.forward`, lines 228–263; `WanAttentionBlock.forward`, lines 418–426 | Q = video/sequence hidden `[B,L,1536]`; K,V = projected T5 context `[B,512,1536]` | cross-attention is present in every one of 30 blocks |
| 3D RoPE | `Wan21/wan/modules/model.py`, `rope_apply`, lines 50–77; self-attention lines 190–196 | Q/K `[B,L,12,128]`, grid `[B,3]=(F,H,W)` → rotary Q/K | frequency dimensions are split across time/height/width |
| PRoPE | `Wan21/wan/modules/prope.py`, `prope_qkv`, lines 61–113; `Wan21/wan/modules/causal_model.py`, `CausalWanSelfAttention.forward`; `add_prope_parameters`, lines 262–307 | Q/K/V plus per-token w2c `[B,L,4,4]`, K `[B,L,3,3]` → projective attention branch | independent explicit camera tokens are absent in the base model; `prope_o` exists in the Action2V checkpoint |
| Camera source format | `Wan21/wan_utils/dataset.py`, `CameraLatentLMDBDataset`, lines 305–398; `build_viewmats_and_Ks`, lines 401–436 | pose `[tx,ty,tz,qx,qy,qz,qw]` → viewmat `[F,4,4]`, K `[F,3,3]` | OpenCV w2c; scipy quaternion order is xyzw; poses normalized to the first frame |
| Minimal-sample camera construction | `full_duplex/preencode.py`, `_camera_data`, lines 179–190; `Wan21/scripts/data_preprocessing/build_worldplaygen_lmdb.py`, `poses_from_pose_str` | real ordered labels `right-8,a-11` → 20 w2c viewmats/Ks → camera `[20,13]` | The sample contains no measured camera file. Camera is deterministic repository trajectory output, not random data; `right` drives rotation and `a` drives translation. |
| Action source | `dataset/SmallestData/split_4f_actions/manifest.json`, lines 10–29; `dataset/SmallestData/smllest_input.json`, line 4 | `right-8,a-11` → 19 action transitions | vocabulary observed in this sample: `right`, `a`; no-op is reserved but absent |
| Flow target | `Wan21/wan_utils/scheduler.py`, `FlowMatchScheduler.add_noise/training_target`, lines 159–180; `Wan21/model/camera_diffusion.py`, lines 45–94 | `x_sigma=(1-sigma)x0+sigma*epsilon`; target `epsilon-x0` | weighted float32 MSE; sign is checkpoint-compatible |
| Checkpoint loading | `Wan21/wan_trainer/camera_ar_diffusion.py`, `Trainer.__init__`, lines 117–138 | top-level `generator` state dict → `WanDiffusionWrapper` | trainer uses strict=True; inference fallback at `Wan21/wan_inference.py:115–123` uses unsafe strict=False and is not reused here |
| Base causal mask | `Wan21/wan/modules/causal_model.py`, `_prepare_blockwise_causal_attn_mask`, lines 732–787 | per-frame block causal visibility | tokens in a block attend within block and prior blocks |
| Base teacher forcing | same file, `_prepare_teacher_forcing_mask`, lines 790–875; `_forward_train`, lines 1230–1249 | `[clean sequence, noisy sequence]` | noisy block reads prior clean blocks plus its own noisy block; this is not rollout |
| Blockwise generation | `Wan21/pipeline/causal_diffusion_inference.py`, `inference`, lines 238–332 | fixed sampled noise → scheduler loop → latent chunk; clean rerun updates KV cache | base inference defaults to 50 steps and does not predict camera |
| Freeze policy | `full_duplex/configs/overfit.yaml` plus runtime trainable-parameter log | VAE/T5 frozen; base generator and all new modules trainable by fidelity default | any runtime reduction is logged explicitly and never hidden |

## Checkpoint facts already verified

- Absolute path: `/output/minwm/ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt`.
- Size: 5,959,605,031 bytes.
- Top-level keys: exactly `generator`; no EMA, optimizer, scheduler, or trainer state.
- Generator entries: 885 tensors, 1,489,821,760 elements, all float32.
- Specification: Wan2.1 T2V 1.3B family, hidden dim 1536, FFN dim 8960, 12 heads, 30 layers, input/output channels 16, patch `(1,2,2)`, text width 4096 and length 512.
- The checkpoint includes `self_attn.prope_o.{weight,bias}` for every transformer layer.

## Data discrepancy to preserve in metadata

The prompt alternates between “11 actions/turns” and “19 actions.” The real 77-frame sample and labeled manifest contain 19 transitions: 8 `right`, then 11 `a`. Consequently the implementation uses 20 cached VAE states and 19 rollout turns. The initial state is latent 0; action turn `t` maps transition `(latent t → latent t+1)`. Per the explicit null-start protocol, the model input at turn 0 is zero plus `NULL_WORLD_STATE`, while target state is latent 1; latent 0 is retained in cache and alignment metadata as the real pre-action state.

The framework image labels a nominal 200 ms turn. The real video is 24 fps and
each action span is exactly 4 frames, so the verified data duration is
`4/24 = 166.67 ms` per turn. Training follows frame/latent alignment rather than
silently changing the video timing to the diagram label.

The prompt also names both `script_test_split/actions` and an unavailable
`/mnt/.../split_4f_actions/actions` path. The local `script_test_split` directory
does exist. Its 19 MP4 files are pairwise bitwise identical to the local
`split_4f_actions/actions` files; only the latter manifest contains the actual
`right`/`a` labels (the former says `action_00..action_18`). Therefore the
semantic manifest is used for IDs/alignment. Exact hashes and both paths are in
`full_duplex/outputs/smallest_000000/data_audit.json`.

## Runtime confirmations

- `full_duplex/outputs/smallest_000000/preencode.log`: decoded pixel `[3,77,480,832]` float32; VAE output `[20,16,60,104]` float32 before cache conversion; cache latent float16 with min/max/mean/std `-3.25/3.453125/0.094372/0.816062`; T5 `[512,4096]` bfloat16 with 132 non-padding tokens; camera `[20,13]` float32; cache reload is bitwise equal.
- `full_duplex/outputs/smallest_000000/checkpoint_audit.log`: missing `[]`, unexpected `[]`, strict loaded ratio `1.0`; real checkpoint forward inputs are latent `[1,1,16,60,104]` bf16, prompt `[1,512,4096]` bf16, w2c `[1,1,4,4]` bf16 and K `[1,1,3,3]` bf16; outputs flow/x0 `[1,1,16,60,104]` bf16, both finite; 14.69 seconds and 3.01 GiB peak allocated GPU memory.

## Implemented Full-Duplex evidence

| Item | File / class / function / lines | Verified input → output | dtype / semantic check |
|---|---|---|---|
| Special vocabulary | `full_duplex/tokens.py`, `SpecialTokenVocabulary`, lines 68–92 | 9 base names + `TIME_INDEX_0..31` → 41 distinct integer IDs | `nn.Embedding(41,1536)` fp32 master weights, trainable; initialized `normal(0,0.02)`; absent from base checkpoint by construction |
| Exact sequence and spans | `full_duplex/tokens.py`, `build_layout`, lines 95–186 | world tokens `1560`, camera tokens `1`, 19 turns → conceptual sequence length `89187` | output content mask contains exactly `19*(1560+1)=29659` bool targets; boundaries/time/null never targets |
| Causal visibility | `full_duplex/tokens.py`, `build_attention_mask`, lines 189–205; `full_duplex/model.py`, `_make_masks`, lines 464–521 | `SequenceLayout[L]` → boolean/FlexAttention `[L,L]` visibility | past all visible; same-turn non-output keys visible; current masked output keys and all future keys invisible; padding invalid |
| World/noise encoder | `full_duplex/model.py`, `FullDuplexWanModel._patchify`, lines 257–289 | `[B,1,16,60,104]` → `[B,1560,1536]` at fidelity stride 1 | reuses strict checkpoint `patch_embedding`; training autocast performs linear/attention math in bf16 |
| Camera/action encoders | `full_duplex/model.py`, `CameraEncoder`, lines 100–113; `_assemble_sequence`, lines 326–447 | camera `[B,13]` → `[B,1,1536]`; action ID `[B]` → `[B,1,1536]` | explicit tokens are distinct from per-video-token PRoPE geometry |
| Stream assembly | `full_duplex/model.py`, `_assemble_sequence`, lines 326–447 | list of chronological `DuplexTurn` objects → hidden `[B,L,1536]`, coordinates `[L,3]`, viewmats `[B,L,4,4]`, Ks `[B,L,3,3]` | current world/camera outputs are trainable mask embeddings; historical output slots contain model predictions, never GT |
| RoPE + PRoPE | `full_duplex/model.py`, `_scattered_rope`, lines 450–476; `_full_duplex_self_attention`, lines 537–600 | Q/K/V `[B,L,12,128]` plus `(turn,y,x)` and w2c/K → attention `[B,L,1536]` | base 3D frequencies and every checkpoint `prope_o` are reused; explicit camera token remains separate |
| Text cross-attention | `full_duplex/model.py`, `_block_forward`, lines 602–633; `forward`, line 635 onward | Q = stream hidden `[B,L,1536]`; K/V = cached T5 `[B,512,4096]` projected to `[B,512,1536]` | same zero-padded/masked frozen prompt embedding is reused by every executed checkpoint block and every turn |
| Outputs | `full_duplex/model.py`, `FullDuplexWanModel.forward`, line 635 onward | current masked world slots → flow `[B,1,16,60,104]`; masked camera slot → camera `[B,13]` | base world head reused; stable translation+6D rotation+intrinsics residual camera head and optional zero-initialized world residual head are new/trainable |
| Differentiable denoising | `full_duplex/flow.py`, lines 11–46; `full_duplex/training.py`, `_denoise_turn`, lines 269–340 | fixed initial epsilon `[B,1,16,60,104]` → 10 Euler updates → predicted latent same shape | checkpoint sign `epsilon-x0`; sigma decreases 1→0; no detach and initial-noise SHA/mutation assertion |
| Rollout/BPTT | `full_duplex/training.py`, `forward_loss`, lines 342–449; `train_step`, lines 458–537 | predicted state/camera at turn `t` → input at `t+1`, through 19 turns | grad-fn assertions at boundaries; future-only contribution to turn-0 gradient recovered after one total backward and asserted finite/nonzero |
| Checkpoint/reload | `full_duplex/training.py`, lines 539–873 | strict base + explicit trainable delta + optimizer/scheduler/RNG/config → restored model | all 885 intentionally frozen base keys are enumerated and checked exactly; unexpected keys must be empty; same probe output max error must be zero |
| Optimizer groups | `full_duplex/training.py`, `FullDuplexTrainer.__init__`, lines 117–184; `load_checkpoint`, lines 617–775 | named trainable parameters → default/world-head AdamW groups; old one-group state → two groups | LR changes require an explicit CLI opt-in; parameter order and full moment state are checked during migration rather than discarded |

The assembled hidden tensor is logged as fp32 because trainable embedding master
weights are fp32 and residual additions promote the stream. Under the configured
CUDA autocast, checkpoint linear/attention/FFN kernels compute in bf16. This is
reported explicitly rather than labeling the entire hidden stream bf16.
