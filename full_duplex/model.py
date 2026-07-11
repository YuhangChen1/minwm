from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from full_duplex.camera import CAMERA_DIM, camera_to_viewmats_and_Ks
from full_duplex.tokens import (
    SequenceLayout,
    SpecialTokenVocabulary,
    TokenType,
    build_layout,
)


def padded_attention_length(length: int, strategy: str = "power_of_two") -> int:
    """Choose a FlexAttention compile bucket without changing visible tokens.

    A rollout has one distinct sequence length per micro turn. Padding every
    length merely to 128 caused FlexAttention to exhaust Dynamo's recompilation
    limit and fall back to the unfused score-matrix implementation. Power-of-two
    buckets keep a 19-turn rollout to five compiled shapes; padding remains
    masked and cannot leak.
    """
    if length < 1:
        raise ValueError("attention sequence length must be positive")
    if strategy == "multiple_of_128":
        return math.ceil(length / 128) * 128
    if strategy == "power_of_two":
        return 1 << max(7, (length - 1).bit_length())
    raise ValueError(f"Unknown attention_padding_strategy={strategy!r}")


@dataclass
class DuplexTurn:
    """All visible content for one streaming micro turn.

    `world_output` and `camera_output` must be predictions for historical turns
    and must be None for the currently predicted turn.
    """

    turn_index: int
    world_input: torch.Tensor  # [B, 1, 16, H, W]
    camera_input: torch.Tensor  # [B, 13]
    action_id: torch.Tensor  # [B]
    noise_input: torch.Tensor  # [B, 1, 16, H, W]
    world_output: torch.Tensor | None = None  # [B, 1, 16, H, W]
    camera_output: torch.Tensor | None = None  # [B, 13]


@dataclass
class FullDuplexOutput:
    flow: torch.Tensor
    camera: torch.Tensor
    layout: SequenceLayout
    sequence_length: int
    hidden_is_finite: bool


@dataclass(frozen=True)
class PatchGrid:
    latent_height: int
    latent_width: int
    patch_height: int
    patch_width: int
    selected_height: int
    selected_width: int
    spatial_stride: int
    coordinates: torch.Tensor

    @property
    def token_count(self) -> int:
        return self.selected_height * self.selected_width


def normalize_generator_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict) or list(checkpoint) != ["generator"]:
        raise KeyError("Expected checkpoint with exactly one top-level key: generator")
    state = checkpoint["generator"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint['generator'] must be a state dict")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor checkpoint value: {key}")
        new_key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        if new_key in normalized:
            raise KeyError(f"Duplicate normalized key {new_key}")
        normalized[new_key] = value
    return normalized


class CameraEncoder(nn.Module):
    def __init__(self, dim: int, num_tokens: int):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.network = nn.Sequential(
            nn.LayerNorm(CAMERA_DIM),
            nn.Linear(CAMERA_DIM, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * num_tokens),
        )

    def forward(self, camera: torch.Tensor) -> torch.Tensor:
        return self.network(camera).unflatten(-1, (self.num_tokens, self.dim))


class CameraPredictionHead(nn.Module):
    """Predict a residual over the visible camera; zero init starts as persistence."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.hidden = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, CAMERA_DIM)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: torch.Tensor, input_camera: torch.Tensor) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        delta = self.output(F.silu(self.hidden(self.norm(pooled))))
        return input_camera + delta


class FullDuplexWanModel(nn.Module):
    def __init__(self, backbone: nn.Module, config: dict[str, Any], load_report: dict[str, Any]):
        super().__init__()
        self.backbone = backbone
        self.config = dict(config)
        self.load_report = load_report
        self.dim = int(backbone.dim)
        if self.dim != config["model_dim"]:
            raise ValueError(f"Configured dim {config['model_dim']} != checkpoint dim {self.dim}")
        self.vocabulary = SpecialTokenVocabulary(config["max_time_index"])
        self.special_embedding = nn.Embedding(len(self.vocabulary), self.dim)
        self.type_embedding = nn.Embedding(len(TokenType), self.dim)
        self.turn_embedding = nn.Embedding(config["max_time_index"] + 1, self.dim)
        self.action_embedding = nn.Embedding(len(config["action_vocabulary"]), self.dim)
        self.camera_encoder = CameraEncoder(self.dim, config["num_camera_tokens"])
        self.camera_head = CameraPredictionHead(self.dim)
        patch_volume = math.prod(backbone.patch_size) * backbone.out_dim
        if config.get("world_residual_head", False):
            self.world_residual_norm = nn.LayerNorm(self.dim)
            self.world_residual_head = nn.Linear(self.dim, patch_volume)
            nn.init.zeros_(self.world_residual_head.weight)
            nn.init.zeros_(self.world_residual_head.bias)
        if config.get("world_time_space_prior", False):
            _, patch_height, patch_width = backbone.patch_size
            latent_height = int(config["latent_height"])
            latent_width = int(config["latent_width"])
            if latent_height % patch_height or latent_width % patch_width:
                raise ValueError("Latent resolution must be divisible by the checkpoint patch size")
            self.max_world_patch_tokens = (
                latent_height // patch_height
            ) * (latent_width // patch_width)
            prior_entries = (config["max_time_index"] + 1) * self.max_world_patch_tokens
            # A zero-initialized optional module must not perturb initialization
            # of the other new modules in an ablation run.
            cpu_rng_state = torch.get_rng_state()
            self.world_time_space_prior = nn.Embedding(prior_entries, patch_volume)
            torch.set_rng_state(cpu_rng_state)
            nn.init.zeros_(self.world_time_space_prior.weight)
        self._initialize_new_embeddings()
        self.spatial_token_stride = int(config["spatial_token_stride"])
        self.num_backbone_blocks = int(config["num_backbone_blocks"])
        if not 1 <= self.num_backbone_blocks <= len(backbone.blocks):
            raise ValueError("num_backbone_blocks must be in [1, checkpoint layer count]")
        self.gradient_checkpointing = bool(config["gradient_checkpointing"])
        self.use_prope = bool(config["use_prope"])
        self._mask_cache: dict[tuple[Any, ...], tuple[BlockMask, BlockMask | None]] = {}
        self._patch_cache: dict[int, tuple[torch.Tensor, torch.Tensor, PatchGrid]] = {}
        self._camera_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._context_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_shape_log: dict[str, Any] | None = None

    def clear_step_cache(self) -> None:
        """Drop graph-bearing encodings at an optimizer/evaluation boundary."""
        self._patch_cache.clear()
        self._camera_cache.clear()
        self._context_cache.clear()

    def _initialize_new_embeddings(self) -> None:
        for embedding in (
            self.special_embedding,
            self.type_embedding,
            self.turn_embedding,
            self.action_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        for module in self.camera_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @classmethod
    def from_checkpoint(cls, config: dict[str, Any]) -> "FullDuplexWanModel":
        root = Path(config["project_root"])
        sys.path.insert(0, str(root / "Wan21"))
        sys.path.insert(0, str(root / "shared"))
        from wan_utils.wan_wrapper import WanDiffusionWrapper

        wrapper = WanDiffusionWrapper(
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=config["timestep_shift"],
            is_causal=True,
            local_attn_size=20,
            use_camera=True,
        )
        checkpoint = torch.load(
            config["base_checkpoint"], map_location="cpu", mmap=True, weights_only=True
        )
        state = normalize_generator_state(checkpoint)
        incompatible = wrapper.load_state_dict(state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        print(f"[full-duplex load] base missing_keys({len(missing)})={missing}")
        print(f"[full-duplex load] base unexpected_keys({len(unexpected)})={unexpected}")
        if missing or unexpected:
            raise RuntimeError("Base checkpoint mismatch is not an expected new-module mismatch")
        wrapper.load_state_dict(state, strict=True)
        base_elements = sum(value.numel() for value in wrapper.state_dict().values())
        loaded_elements = sum(value.numel() for value in state.values())
        ratio = loaded_elements / base_elements
        if ratio != 1.0:
            raise RuntimeError(f"Base loaded ratio is {ratio}, expected 1.0")
        load_report = {
            "base_missing_keys": missing,
            "base_unexpected_keys": unexpected,
            "base_loaded_parameter_ratio": ratio,
            "base_loaded_elements": loaded_elements,
            "base_checkpoint": str(Path(config["base_checkpoint"]).resolve()),
        }
        model = cls(wrapper.model, config, load_report)
        del checkpoint, state, wrapper
        return model

    def new_parameter_names(self) -> list[str]:
        prefixes = (
            "special_embedding.",
            "type_embedding.",
            "turn_embedding.",
            "action_embedding.",
            "camera_encoder.",
            "camera_head.",
            "world_residual_norm.",
            "world_residual_head.",
            "world_time_space_prior.",
        )
        return [name for name, _ in self.named_parameters() if name.startswith(prefixes)]

    def configure_trainable_parameters(
        self,
        train_backbone: bool,
        train_base_world_head: bool = False,
    ) -> dict[str, int]:
        self.backbone.requires_grad_(train_backbone)
        if train_base_world_head:
            self.backbone.head.requires_grad_(True)
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                parameter.requires_grad_(True)
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def _patchify(self, latent: torch.Tensor) -> tuple[torch.Tensor, PatchGrid]:
        if latent.ndim != 5 or latent.shape[1] != 1 or latent.shape[2] != self.backbone.in_dim:
            raise ValueError(f"Expected [B,1,{self.backbone.in_dim},H,W], got {tuple(latent.shape)}")
        cache_key = id(latent)
        cached = self._patch_cache.get(cache_key)
        if cached is not None and cached[0] is latent:
            return cached[1], cached[2]
        embedded = self.backbone.patch_embedding(latent.permute(0, 2, 1, 3, 4))
        if embedded.shape[2] != 1:
            raise ValueError("This micro-turn implementation expects one VAE state per turn")
        patch_h, patch_w = embedded.shape[-2:]
        stride = self.spatial_token_stride
        selected = embedded[:, :, 0, ::stride, ::stride]
        selected_h, selected_w = selected.shape[-2:]
        tokens = selected.flatten(2).transpose(1, 2).contiguous()
        ys = torch.arange(0, patch_h, stride, device=latent.device, dtype=torch.long)
        xs = torch.arange(0, patch_w, stride, device=latent.device, dtype=torch.long)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((torch.zeros_like(yy), yy, xx), dim=-1).reshape(-1, 3)
        grid = PatchGrid(
            latent_height=latent.shape[-2],
            latent_width=latent.shape[-1],
            patch_height=patch_h,
            patch_width=patch_w,
            selected_height=selected_h,
            selected_width=selected_w,
            spatial_stride=stride,
            coordinates=coords,
        )
        if tokens.shape[1] != grid.token_count:
            raise AssertionError("patch token/grid mismatch")
        self._patch_cache[cache_key] = (latent, tokens, grid)
        return tokens, grid

    def _camera_tokens(self, camera: torch.Tensor) -> torch.Tensor:
        cache_key = id(camera)
        cached = self._camera_cache.get(cache_key)
        if cached is not None and cached[0] is camera:
            return cached[1]
        tokens = self.camera_encoder(camera)
        self._camera_cache[cache_key] = (camera, tokens)
        return tokens

    def _unpatchify_selected(self, patches: torch.Tensor, grid: PatchGrid) -> torch.Tensor:
        batch = patches.shape[0]
        pt, ph, pw = self.backbone.patch_size
        channels = self.backbone.out_dim
        if pt != 1 or patches.shape[-1] != pt * ph * pw * channels:
            raise ValueError("Checkpoint head/patch dimensions are incompatible")
        patches = patches.view(
            batch, grid.selected_height, grid.selected_width, pt, ph, pw, channels
        )
        latent = torch.einsum("bhwtpqc->bcthpwq", patches)
        latent = latent.reshape(
            batch, channels, pt, grid.selected_height * ph, grid.selected_width * pw
        )
        if latent.shape[-2:] != (grid.latent_height, grid.latent_width):
            latent = F.interpolate(
                latent,
                size=(pt, grid.latent_height, grid.latent_width),
                mode="trilinear",
                align_corners=False,
            )
        return latent.permute(0, 2, 1, 3, 4).contiguous()

    def _special(self, name: str, batch: int, length: int, device: torch.device) -> torch.Tensor:
        ids = torch.full((batch, length), self.vocabulary.id(name), device=device, dtype=torch.long)
        return self.special_embedding(ids)

    def _assemble_sequence(
        self, turns: list[DuplexTurn]
    ) -> tuple[
        torch.Tensor,
        SequenceLayout,
        PatchGrid,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not turns:
            raise ValueError("At least one turn is required")
        absolute_turns = [turn.turn_index for turn in turns]
        if absolute_turns != sorted(absolute_turns) or len(set(absolute_turns)) != len(turns):
            raise ValueError("Turns must be unique and chronological")
        current_index = absolute_turns[-1]
        batch = turns[0].world_input.shape[0]
        device = turns[0].world_input.device
        dtype = turns[0].world_input.dtype
        patches: dict[tuple[int, str], torch.Tensor] = {}
        grid: PatchGrid | None = None
        for turn in turns:
            for field in ("world_input", "noise_input"):
                value, this_grid = self._patchify(getattr(turn, field))
                patches[(turn.turn_index, field)] = value
                if grid is None:
                    grid = this_grid
                elif this_grid.token_count != grid.token_count:
                    raise ValueError("All turn states must share a patch grid")
            if turn.turn_index != current_index:
                if turn.world_output is None or turn.camera_output is None:
                    raise ValueError("Historical output slots must contain predictions")
                patches[(turn.turn_index, "world_output")], _ = self._patchify(turn.world_output)
            elif turn.world_output is not None or turn.camera_output is not None:
                raise ValueError("Current output content must remain masked")
        assert grid is not None
        layout = build_layout(
            len(turns),
            grid.token_count,
            self.config["num_camera_tokens"],
            self.vocabulary,
            turn_indices=absolute_turns,
        )
        turn_by_index = {turn.turn_index: turn for turn in turns}
        pieces: list[torch.Tensor] = []
        coordinate_pieces: list[torch.Tensor] = []
        video_pieces: list[torch.Tensor] = []
        viewmat_pieces: list[torch.Tensor] = []
        K_pieces: list[torch.Tensor] = []
        for span in layout.spans:
            turn = turn_by_index[span.turn]
            if span.is_special:
                special_id = int(layout.special_ids[span.start])
                piece = self.special_embedding(
                    torch.full((batch, span.length), special_id, device=device, dtype=torch.long)
                )
            elif span.name == "world_input":
                piece = patches[(span.turn, "world_input")]
            elif span.name == "camera_input":
                piece = self._camera_tokens(turn.camera_input)
            elif span.name == "action_input":
                piece = self.action_embedding(turn.action_id).unsqueeze(1)
            elif span.name == "noise_input":
                piece = patches[(span.turn, "noise_input")]
            elif span.name == "world_output":
                piece = (
                    self._special("MASKED_WORLD", batch, span.length, device)
                    if span.turn == current_index
                    else patches[(span.turn, "world_output")]
                )
            elif span.name == "camera_output":
                piece = (
                    self._special("MASKED_CAMERA", batch, span.length, device)
                    if span.turn == current_index
                    else self._camera_tokens(turn.camera_output)
                )
            else:
                raise KeyError(f"Unhandled token span {span}")
            if piece.shape != (batch, span.length, self.dim):
                raise ValueError(f"Span {span.name} produced {piece.shape}")
            pieces.append(piece)

            is_video = span.token_type in (
                TokenType.WORLD_INPUT,
                TokenType.NOISE_INPUT,
                TokenType.WORLD_OUTPUT,
            )
            if is_video:
                coords = grid.coordinates.clone()
                coords[:, 0] = span.turn
                coordinate_pieces.append(coords)
                video_pieces.append(torch.ones(span.length, device=device, dtype=torch.bool))
                camera_for_geometry = (
                    turn.camera_output
                    if span.name == "world_output" and span.turn != current_index
                    else turn.camera_input
                )
                viewmat, K = camera_to_viewmats_and_Ks(camera_for_geometry)
                viewmat_pieces.append(viewmat[:, None].expand(-1, span.length, -1, -1))
                K_pieces.append(K[:, None].expand(-1, span.length, -1, -1))
            else:
                coordinate_pieces.append(
                    torch.full((span.length, 3), -1, device=device, dtype=torch.long)
                )
                video_pieces.append(torch.zeros(span.length, device=device, dtype=torch.bool))
                eye4 = torch.eye(4, device=device, dtype=dtype)[None, None]
                eye3 = torch.eye(3, device=device, dtype=dtype)[None, None]
                viewmat_pieces.append(eye4.expand(batch, span.length, -1, -1))
                K_pieces.append(eye3.expand(batch, span.length, -1, -1))

        hidden = torch.cat(pieces, dim=1)
        token_types = layout.token_types.to(device)
        turn_ids = layout.turn_ids.to(device)
        hidden = hidden + self.type_embedding(token_types)[None] + self.turn_embedding(turn_ids)[None]
        coordinates = torch.cat(coordinate_pieces, dim=0)
        video_mask = torch.cat(video_pieces, dim=0)
        viewmats = torch.cat(viewmat_pieces, dim=1)
        Ks = torch.cat(K_pieces, dim=1)
        if hidden.shape[1] != layout.sequence_length:
            raise AssertionError("assembled sequence/layout mismatch")
        return hidden, layout, grid, coordinates, video_mask, viewmats, Ks

    @staticmethod
    def _scattered_rope(
        tensor: torch.Tensor, coordinates: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        valid = coordinates[:, 0].ge(0)
        if not torch.any(valid):
            return tensor
        indices = valid.nonzero(as_tuple=False).flatten()
        coords = coordinates[indices]
        half = tensor.shape[-1] // 2
        splits = (half - 2 * (half // 3), half // 3, half // 3)
        time_freq, height_freq, width_freq = freqs.split(splits, dim=1)
        rotary = torch.cat(
            (
                time_freq[coords[:, 0]],
                height_freq[coords[:, 1]],
                width_freq[coords[:, 2]],
            ),
            dim=-1,
        ).view(1, -1, 1, half)
        selected = tensor[:, indices]
        complex_selected = torch.view_as_complex(
            selected.to(torch.float64).reshape(*selected.shape[:-1], half, 2)
        )
        rotated = torch.view_as_real(complex_selected * rotary).flatten(-2).type_as(tensor)
        output = tensor.clone()
        output[:, indices] = rotated
        return output

    def _make_masks(
        self, layout: SequenceLayout, video_mask: torch.Tensor, padded_length: int, device: torch.device
    ) -> tuple[BlockMask, BlockMask | None]:
        key = (
            tuple(int(value) for value in layout.turn_ids.tolist()),
            tuple(bool(value) for value in layout.output_content.tolist()),
            tuple(bool(value) for value in video_mask.cpu().tolist()),
            padded_length,
            str(device),
            self.use_prope,
        )
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        length = layout.sequence_length
        turns = torch.full((padded_length,), -1, device=device, dtype=torch.long)
        turns[:length] = layout.turn_ids.to(device)
        output_content = torch.ones(padded_length, device=device, dtype=torch.bool)
        output_content[:length] = layout.output_content.to(device)
        valid = torch.arange(padded_length, device=device).lt(length)
        video = torch.zeros(padded_length, device=device, dtype=torch.bool)
        video[:length] = video_mask

        def standard_mask(_b, _h, query, key_index):
            both_valid = valid[query] & valid[key_index]
            allowed = (turns[key_index] < turns[query]) | (
                (turns[key_index] == turns[query]) & ~output_content[key_index]
            )
            padding_diagonal = (~valid[query]) & (query == key_index)
            return (both_valid & allowed) | padding_diagonal

        block_mask = create_block_mask(
            standard_mask,
            B=None,
            H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            device=device,
            _compile=False,
        )
        prope_mask = None
        if self.use_prope:
            def geometry_mask(_b, _h, query, key_index):
                geometry_allowed = standard_mask(_b, _h, query, key_index) & video[query] & video[key_index]
                nongeometry_diagonal = (~video[query]) & (query == key_index)
                return geometry_allowed | nongeometry_diagonal

            prope_mask = create_block_mask(
                geometry_mask,
                B=None,
                H=None,
                Q_LEN=padded_length,
                KV_LEN=padded_length,
                device=device,
                _compile=False,
            )
        self._mask_cache[key] = (block_mask, prope_mask)
        return block_mask, prope_mask

    def _full_duplex_self_attention(
        self,
        attention: nn.Module,
        hidden: torch.Tensor,
        coordinates: torch.Tensor,
        video_mask: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        block_mask: BlockMask,
        prope_mask: BlockMask | None,
        padded_length: int,
    ) -> torch.Tensor:
        from wan.modules.causal_model import flex_attention as compiled_flex_attention

        batch, length, _ = hidden.shape
        heads, head_dim = attention.num_heads, attention.head_dim
        q = attention.norm_q(attention.q(hidden)).view(batch, length, heads, head_dim)
        k = attention.norm_k(attention.k(hidden)).view(batch, length, heads, head_dim)
        v = attention.v(hidden).view(batch, length, heads, head_dim)
        pad = padded_length - length
        if pad < 0:
            raise ValueError("padded attention length is shorter than the sequence")
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        # Match wan.modules.attention.flash_attention: RMSNorm's fp32 master
        # scale can promote Q/K, while attention kernels require Q/K/V parity.
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        padded_coords = F.pad(coordinates, (0, 0, 0, pad), value=-1)
        freqs = self.backbone.freqs.to(hidden.device)
        q_rope = self._scattered_rope(q, padded_coords, freqs)
        k_rope = self._scattered_rope(k, padded_coords, freqs)
        standard = compiled_flex_attention(
            q_rope.transpose(1, 2),
            k_rope.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
        ).transpose(1, 2)
        standard = attention.o(standard[:, :length].flatten(2))

        if self.use_prope:
            if prope_mask is None or not hasattr(attention, "prope_o"):
                raise RuntimeError("PRoPE requested without checkpoint PRoPE parameters/mask")
            from wan.modules.prope import prope_qkv

            eye4 = torch.eye(4, device=hidden.device, dtype=hidden.dtype)[None, None]
            eye3 = torch.eye(3, device=hidden.device, dtype=hidden.dtype)[None, None]
            padded_viewmats = torch.cat(
                (viewmats, eye4.expand(batch, pad, -1, -1)), dim=1
            ) if pad else viewmats
            padded_Ks = torch.cat((Ks, eye3.expand(batch, pad, -1, -1)), dim=1) if pad else Ks
            q_p, k_p, v_p, apply_output = prope_qkv(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                viewmats=padded_viewmats,
                Ks=padded_Ks,
            )
            geometry = compiled_flex_attention(q_p, k_p, v_p, block_mask=prope_mask)
            geometry = apply_output(geometry).permute(0, 2, 1, 3)[:, :length]
            geometry = geometry * video_mask[None, :, None, None]
            standard = standard + attention.prope_o(geometry.flatten(2))
        return standard

    def _block_forward(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        modulation: torch.Tensor,
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        coordinates: torch.Tensor,
        video_mask: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        block_mask: BlockMask,
        prope_mask: BlockMask | None,
        padded_length: int,
    ) -> torch.Tensor:
        e = (block.modulation[None] + modulation).chunk(6, dim=2)
        self_input = block.norm1(hidden) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        attended = self._full_duplex_self_attention(
            block.self_attn,
            self_input,
            coordinates,
            video_mask,
            viewmats,
            Ks,
            block_mask,
            prope_mask,
            padded_length,
        )
        hidden = hidden + attended * e[2].squeeze(2)
        hidden = hidden + block.cross_attn(block.norm3(hidden), context, context_lengths)
        ffn = block.ffn(block.norm2(hidden) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        return hidden + ffn * e[5].squeeze(2)

    def forward(
        self,
        turns: list[DuplexTurn],
        prompt_embedding: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        turn_timesteps: torch.Tensor,
    ) -> FullDuplexOutput:
        hidden, layout, grid, coords, video_mask, viewmats, Ks = self._assemble_sequence(turns)
        device, dtype = hidden.device, hidden.dtype
        batch = hidden.shape[0]
        absolute_turns = [turn.turn_index for turn in turns]
        if turn_timesteps.shape != (batch, len(turns)):
            raise ValueError(
                f"turn_timesteps must be {(batch, len(turns))}, got {tuple(turn_timesteps.shape)}"
            )
        from wan.modules.model import sinusoidal_embedding_1d

        time_embedding = self.backbone.time_embedding(
            sinusoidal_embedding_1d(
                self.backbone.freq_dim, turn_timesteps.flatten()
            ).type_as(hidden)
        ).unflatten(0, (batch, len(turns)))
        projected = self.backbone.time_projection(time_embedding.flatten(0, 1))
        projected = projected.unflatten(1, (6, self.dim)).unflatten(0, (batch, len(turns)))
        local_index = {absolute: local for local, absolute in enumerate(absolute_turns)}
        token_local_turns = torch.tensor(
            [local_index[int(turn)] for turn in layout.turn_ids], device=device, dtype=torch.long
        )
        modulation = projected[:, token_local_turns]

        if prompt_embedding.shape[:2] != (batch, 512) or prompt_embedding.shape[-1] != self.backbone.text_dim:
            raise ValueError(f"Unexpected prompt shape {tuple(prompt_embedding.shape)}")
        if prompt_attention_mask.shape != (batch, 512):
            raise ValueError(f"Unexpected prompt mask {tuple(prompt_attention_mask.shape)}")
        context_key = id(prompt_embedding)
        cached_context = self._context_cache.get(context_key)
        if cached_context is not None and cached_context[0] is prompt_embedding:
            context = cached_context[1]
        else:
            context = self.backbone.text_embedding(prompt_embedding)
            self._context_cache[context_key] = (prompt_embedding, context)
        context_lengths = prompt_attention_mask.sum(dim=1).to(device=device, dtype=torch.int32)
        padded_length = padded_attention_length(
            layout.sequence_length,
            self.config.get("attention_padding_strategy", "power_of_two"),
        )
        pad_to_turns = int(self.config.get("attention_pad_to_turns", 0))
        if pad_to_turns:
            if not 1 <= pad_to_turns <= self.config["num_micro_turns"]:
                raise ValueError("attention_pad_to_turns must fit the cached rollout")
            maximum_layout = build_layout(
                pad_to_turns,
                grid.token_count,
                self.config["num_camera_tokens"],
                self.vocabulary,
            )
            padded_length = max(
                padded_length,
                padded_attention_length(
                    maximum_layout.sequence_length,
                    self.config.get("attention_padding_strategy", "power_of_two"),
                ),
            )
        block_mask, prope_mask = self._make_masks(layout, video_mask, padded_length, device)

        for block in self.backbone.blocks[: self.num_backbone_blocks]:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                def run(h, m, c, block=block):
                    return self._block_forward(
                        block, h, m, c, context_lengths, coords, video_mask,
                        viewmats, Ks, block_mask, prope_mask, padded_length,
                    )
                hidden = torch.utils.checkpoint.checkpoint(
                    run, hidden, modulation, context, use_reentrant=False
                )
            else:
                hidden = self._block_forward(
                    block,
                    hidden,
                    modulation,
                    context,
                    context_lengths,
                    coords,
                    video_mask,
                    viewmats,
                    Ks,
                    block_mask,
                    prope_mask,
                    padded_length,
                )

        current_turn = absolute_turns[-1]
        world_span = layout.span(current_turn, "world_output")
        camera_span = layout.span(current_turn, "camera_output")
        world_hidden = hidden[:, world_span.start : world_span.end]
        camera_hidden = hidden[:, camera_span.start : camera_span.end]
        current_local = local_index[current_turn]
        head_e = time_embedding[:, current_local]
        head_modulation = (self.backbone.head.modulation + head_e[:, None]).chunk(2, dim=1)
        world_patches = self.backbone.head.head(
            self.backbone.head.norm(world_hidden)
            * (1 + head_modulation[1])
            + head_modulation[0]
        )
        if hasattr(self, "world_residual_head"):
            world_patches = world_patches + self.world_residual_head(
                self.world_residual_norm(world_hidden)
            )
        if hasattr(self, "world_time_space_prior"):
            spatial_ids = (
                grid.coordinates[:, 1] * grid.patch_width + grid.coordinates[:, 2]
            )
            if int(spatial_ids.max()) >= self.max_world_patch_tokens:
                raise IndexError("World patch coordinate exceeds the configured prior table")
            prior_ids = current_turn * self.max_world_patch_tokens + spatial_ids
            world_patches = world_patches + self.world_time_space_prior(prior_ids)[None]
        flow = self._unpatchify_selected(world_patches, grid)
        camera = self.camera_head(camera_hidden, turns[-1].camera_input)
        finite = bool(torch.isfinite(hidden).all().detach().item())
        if self.last_shape_log is None:
            self.last_shape_log = {
                "micro_turn_latent": list(turns[-1].noise_input.shape),
                "patchified_world_tokens": [batch, grid.token_count, self.dim],
                "flattened_sequence": list(hidden.shape),
                "prompt_context": list(context.shape),
                "flow": list(flow.shape),
                "camera": list(camera.shape),
                "dtype": str(dtype),
                "spatial_token_stride": grid.spatial_stride,
            }
            print(f"[full-duplex shapes] {self.last_shape_log}")
        return FullDuplexOutput(flow, camera, layout, layout.sequence_length, finite)
