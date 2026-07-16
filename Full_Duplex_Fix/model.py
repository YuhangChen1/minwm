from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .debug.tracer import debug_timer, trace_event, trace_gradient, trace_tensor
from .layout import InterleavedLayout, SpanRole
from .mask import create_flex_block_mask
from .rope import explicit_rope_apply


@dataclass(frozen=True)
class InterleavedOutput:
    flow: torch.Tensor
    raw_time_embedding: torch.Tensor
    layout_name: str


@lru_cache(maxsize=1)
def _flex_attention_operator():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False, mode="default")


def _pad_sequence(tensor: torch.Tensor, padding: int) -> torch.Tensor:
    if padding == 0:
        return tensor
    shape = list(tensor.shape)
    shape[1] = padding
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=1)


def _self_attention(
    attention: nn.Module,
    hidden: torch.Tensor,
    *,
    coordinates: torch.Tensor,
    frequencies: torch.Tensor,
    block_mask: Any,
    padding: int,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    debug_prefix: str,
) -> torch.Tensor:
    batch, sequence = hidden.shape[:2]
    heads = attention.num_heads
    head_dim = attention.head_dim
    with debug_timer() as timing:
        query = attention.norm_q(attention.q(hidden)).view(batch, sequence, heads, head_dim)
        key = attention.norm_k(attention.k(hidden)).view(batch, sequence, heads, head_dim)
        value = attention.v(hidden).view(batch, sequence, heads, head_dim)
    trace_event(
        "self_attention",
        f"{debug_prefix}.qkv_projection",
        tensors={"query": query, "key": key, "value": value},
        details={**timing, "heads": heads, "head_dim": head_dim},
    )
    trace_gradient("self_attention", f"{debug_prefix}.query", query)
    trace_gradient("self_attention", f"{debug_prefix}.key", key)
    trace_gradient("self_attention", f"{debug_prefix}.value", value)

    with debug_timer() as timing:
        roped_query = explicit_rope_apply(query, coordinates, frequencies).type_as(value)
        roped_key = explicit_rope_apply(key, coordinates, frequencies).type_as(value)
    trace_event(
        "rope",
        f"{debug_prefix}.explicit_rope",
        tensors={"query": roped_query, "key": roped_key},
        details={**timing, "coordinate_order": "physical_time,height,width"},
    )
    padded_query = _pad_sequence(roped_query, padding)
    padded_key = _pad_sequence(roped_key, padding)
    padded_value = _pad_sequence(value, padding)
    trace_event(
        "shape",
        f"{debug_prefix}.attention_padding",
        tensors={"query": padded_query, "key": padded_key, "value": padded_value},
        details={"padding_tokens": padding, "valid_sequence": sequence},
    )

    flex_attention = _flex_attention_operator()
    with debug_timer() as timing:
        rope_output = flex_attention(
            query=padded_query.transpose(1, 2),
            key=padded_key.transpose(1, 2),
            value=padded_value.transpose(1, 2),
            block_mask=block_mask,
        ).transpose(1, 2)[:, :sequence]
        rope_output = attention.o(rope_output.flatten(2))
    trace_event(
        "self_attention",
        f"{debug_prefix}.rope_attention_output",
        tensors={"output": rope_output},
        details={**timing, "mask": "shared interleaved FlexAttention mask"},
    )
    trace_gradient("self_attention", f"{debug_prefix}.rope_attention_output", rope_output)

    if not hasattr(attention, "prope_o"):
        raise RuntimeError("Checkpoint-compatible PRoPE projection is missing")
    from wan.modules.prope import prope_qkv

    with debug_timer() as timing:
        prope_query, prope_key, prope_value, apply_output = prope_qkv(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            viewmats=viewmats,
            Ks=Ks,
        )
    trace_event(
        "prope",
        f"{debug_prefix}.prope_qkv",
        tensors={"query": prope_query, "key": prope_key, "value": prope_value},
        details={**timing, "camera_tokens": viewmats.shape[1]},
    )
    with debug_timer() as timing:
        prope_output = flex_attention(
            query=_pad_sequence(prope_query.permute(0, 2, 1, 3), padding).transpose(1, 2),
            key=_pad_sequence(prope_key.permute(0, 2, 1, 3), padding).transpose(1, 2),
            value=_pad_sequence(prope_value.permute(0, 2, 1, 3), padding).transpose(1, 2),
            block_mask=block_mask,
        )[:, :, :sequence]
        prope_output = apply_output(prope_output).permute(0, 2, 1, 3).flatten(2)
        projected_prope = attention.prope_o(prope_output)
        merged = rope_output + projected_prope
    trace_event(
        "prope",
        f"{debug_prefix}.prope_attention_and_merge",
        tensors={
            "raw_prope": prope_output,
            "projected_prope": projected_prope,
            "merged_attention": merged,
        },
        details={**timing, "formula": "rope_output+prope_o(prope_output)"},
    )
    trace_gradient("prope", f"{debug_prefix}.projected_prope", projected_prope)
    trace_gradient("self_attention", f"{debug_prefix}.merged_attention", merged)
    return merged


def _run_block(
    block: nn.Module,
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    *,
    num_spans: int,
    tokens_per_span: int,
    coordinates: torch.Tensor,
    frequencies: torch.Tensor,
    block_mask: Any,
    padding: int,
    context: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    block_index: int,
) -> torch.Tensor:
    debug_prefix = f"transformer.block_{block_index:02d}"
    trace_event(
        "transformer",
        f"{debug_prefix}.input",
        tensors={"hidden": hidden},
        details={"block_index": block_index},
    )
    if modulation.shape[1] != num_spans:
        raise ValueError("Frame-level modulation does not match layout")
    with debug_timer() as timing:
        chunks = (block.modulation.unsqueeze(1) + modulation).chunk(6, dim=2)
        normalized = block.norm1(hidden).unflatten(1, (num_spans, tokens_per_span))
        attention_input = (normalized * (1 + chunks[1]) + chunks[0]).flatten(1, 2)
    trace_event(
        "modulation",
        f"{debug_prefix}.self_attention_modulation",
        tensors={"modulation": modulation, "attention_input": attention_input},
        details={**timing, "chunks": "shift,scale,gate,ffn_shift,ffn_scale,ffn_gate"},
    )
    attended = _self_attention(
        block.self_attn,
        attention_input,
        coordinates=coordinates,
        frequencies=frequencies,
        block_mask=block_mask,
        padding=padding,
        viewmats=viewmats,
        Ks=Ks,
        debug_prefix=debug_prefix,
    )
    with debug_timer() as timing:
        hidden = hidden + (
            attended.unflatten(1, (num_spans, tokens_per_span)) * chunks[2]
        ).flatten(1, 2)
    trace_event(
        "residual",
        f"{debug_prefix}.self_attention_residual",
        tensors={"hidden": hidden},
        details=timing,
    )

    with debug_timer() as timing:
        cross_output = block.cross_attn(block.norm3(hidden), context, None)
        hidden = hidden + cross_output
    trace_event(
        "cross_attention",
        f"{debug_prefix}.text_cross_attention",
        tensors={"context": context, "cross_output": cross_output, "hidden": hidden},
        details={**timing, "text_tokens": context.shape[1]},
    )
    trace_gradient("cross_attention", f"{debug_prefix}.cross_output", cross_output)
    with debug_timer() as timing:
        normalized_ffn = block.norm2(hidden).unflatten(1, (num_spans, tokens_per_span))
        ffn_input = (normalized_ffn * (1 + chunks[4]) + chunks[3]).flatten(1, 2)
        ffn_output = block.ffn(ffn_input)
        hidden = hidden + (
            ffn_output.unflatten(1, (num_spans, tokens_per_span)) * chunks[5]
        ).flatten(1, 2)
    trace_event(
        "ffn",
        f"{debug_prefix}.ffn",
        tensors={"ffn_input": ffn_input, "ffn_output": ffn_output},
        details={**timing, "dimensions": "1536->8960->1536"},
    )
    trace_gradient("ffn", f"{debug_prefix}.ffn_output", ffn_output)
    if num_spans == 40:
        role_hidden = hidden.unflatten(1, (num_spans, tokens_per_span))
        trace_event(
            "transformer",
            f"{debug_prefix}.clean_noisy_state_samples",
            tensors={
                "N0": role_hidden[:, 0],
                "W0": role_hidden[:, 1],
                "N19": role_hidden[:, 38],
                "W19": role_hidden[:, 39],
            },
            details={
                "purpose": "compare noisy and clean hidden-state evolution across layers",
                "span_shape": [hidden.shape[0], tokens_per_span, hidden.shape[-1]],
            },
        )
    return trace_tensor(
        "transformer",
        f"{debug_prefix}.output",
        hidden,
        details={"next": f"block_{block_index + 1:02d}" if block_index < 29 else "output_head"},
        backward=True,
    )


class InterleavedWanAdapter(nn.Module):
    """Parameter-free runtime adapter around a strict-loaded Wan generator."""

    def __init__(
        self,
        generator: nn.Module,
        *,
        layout: InterleavedLayout | None = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.layout = layout or InterleavedLayout.main()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self._mask_cache: dict[tuple[str, str, int | None], tuple[Any, int]] = {}
        self._validate_model()

    @property
    def backbone(self) -> nn.Module:
        return self.generator.model

    @property
    def scheduler(self):
        return self.generator.scheduler

    def _validate_model(self) -> None:
        model = self.backbone
        expected = {
            "dim": 1536,
            "num_heads": 12,
            "num_layers": 30,
            "in_dim": 16,
            "out_dim": 16,
        }
        for name, value in expected.items():
            if getattr(model, name) != value:
                raise ValueError(f"Wan {name} must be {value}, got {getattr(model, name)}")
        if tuple(model.patch_size) != (1, 2, 2):
            raise ValueError(f"Unexpected Wan patch size: {model.patch_size}")
        prope_layers = sum(hasattr(block.self_attn, "prope_o") for block in model.blocks)
        if prope_layers != 30:
            raise ValueError(f"Expected 30 PRoPE projections, got {prope_layers}")

    def _block_mask(self, layout: InterleavedLayout, device: torch.device) -> tuple[Any, int]:
        key = (layout.name, device.type, device.index)
        if key not in self._mask_cache:
            self._mask_cache[key] = create_flex_block_mask(layout, device=device)
        return self._mask_cache[key]

    def _assemble_span_latents(
        self,
        noisy_states: torch.Tensor,
        clean_states: torch.Tensor,
        layout: InterleavedLayout,
    ) -> torch.Tensor:
        spans = []
        for span in layout.spans:
            source = noisy_states if span.role == SpanRole.NOISY else clean_states
            if span.physical_time >= source.shape[1]:
                raise ValueError(
                    f"{span.label} requires state {span.physical_time}, source has {source.shape[1]}"
                )
            spans.append(source[:, span.physical_time])
        assembled = torch.stack(spans, dim=1)
        trace_event(
            "layout",
            "layout.assemble_interleaved_spans",
            tensors={
                "noisy_states": noisy_states,
                "clean_states": clean_states,
                "interleaved": assembled,
            },
            details={
                "layout": layout.name,
                "labels": layout.labels(),
                "shape_change": "two [B,20,C,H,W] sources -> [B,40,C,H,W]",
            },
        )
        return assembled

    def _patchify(
        self,
        span_latents: torch.Tensor,
        layout: InterleavedLayout,
    ) -> torch.Tensor:
        batch, spans, channels, height, width = span_latents.shape
        with debug_timer() as timing:
            flat = span_latents.reshape(batch * spans, channels, height, width).unsqueeze(2)
        trace_event(
            "shape",
            "embedding.patch_input_reshape",
            tensors={"patch_input": flat},
            details={**timing, "shape_change": "[B,S,C,H,W] -> [B*S,C,1,H,W]"},
        )
        with debug_timer() as timing:
            embedded_grid = self.backbone.patch_embedding(flat)
        trace_event(
            "embedding",
            "embedding.patch_conv3d",
            tensors={"embedded_grid": embedded_grid},
            details={**timing, "kernel_stride": [1, 2, 2]},
        )
        with debug_timer() as timing:
            embedded = embedded_grid.flatten(2).transpose(1, 2)
        if embedded.shape[1] != layout.tokens_per_span:
            raise ValueError(f"Unexpected patch token count: {embedded.shape[1]}")
        hidden = embedded.unflatten(0, (batch, spans)).flatten(1, 2)
        trace_event(
            "shape",
            "embedding.patch_tokens_flatten",
            tensors={"hidden": hidden},
            details={
                **timing,
                "tokens_per_span": layout.tokens_per_span,
                "shape_change": "[B*S,1536,1,30,52] -> [B,S*1560,1536]",
            },
        )
        return hidden

    @staticmethod
    def _span_timesteps(
        noisy_timesteps: torch.Tensor,
        layout: InterleavedLayout,
    ) -> torch.Tensor:
        values = []
        for span in layout.spans:
            if span.role == SpanRole.NOISY:
                values.append(noisy_timesteps[:, span.physical_time])
            else:
                values.append(torch.zeros_like(noisy_timesteps[:, 0]))
        span_timesteps = torch.stack(values, dim=1)
        trace_event(
            "timestep",
            "timestep.interleave_noisy_and_clean",
            tensors={"span_timesteps": span_timesteps},
            details={"pattern": "[sigma_0,0,sigma_1,0,...,sigma_19,0]"},
        )
        return span_timesteps

    @staticmethod
    def _expand_camera(
        camera: torch.Tensor,
        layout: InterleavedLayout,
    ) -> torch.Tensor:
        indices = torch.tensor(
            [span.camera_index for span in layout.spans],
            dtype=torch.long,
            device=camera.device,
        )
        selected = camera.index_select(1, indices)
        return selected.repeat_interleave(layout.tokens_per_span, dim=1)

    def _unpatchify_noisy(
        self,
        values: torch.Tensor,
        num_states: int,
        layout: InterleavedLayout,
    ) -> torch.Tensor:
        batch = values.shape[0]
        patch_t, patch_h, patch_w = self.backbone.patch_size
        channels = self.backbone.out_dim
        grid_h = layout.patch_height
        grid_w = layout.patch_width
        values = values.view(
            batch,
            num_states,
            grid_h,
            grid_w,
            patch_t,
            patch_h,
            patch_w,
            channels,
        )
        values = torch.einsum("bfhwpqrc->bcfphqwr", values)
        values = values.reshape(
            batch,
            channels,
            num_states * patch_t,
            grid_h * patch_h,
            grid_w * patch_w,
        )
        return values.permute(0, 2, 1, 3, 4).contiguous()

    def forward(
        self,
        *,
        noisy_states: torch.Tensor,
        clean_states: torch.Tensor,
        noisy_timesteps: torch.Tensor,
        prompt_embedding: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        layout: InterleavedLayout | None = None,
    ) -> InterleavedOutput:
        layout = layout or self.layout
        batch = noisy_states.shape[0]
        if noisy_states.shape != (batch, 20, 16, 60, 104):
            raise ValueError(f"Expected noisy [B,20,16,60,104], got {noisy_states.shape}")
        if noisy_timesteps.shape != (batch, 20):
            raise ValueError(f"Expected timesteps [B,20], got {noisy_timesteps.shape}")
        if prompt_embedding.shape != (batch, 512, 4096):
            raise ValueError(f"Expected prompt [B,512,4096], got {prompt_embedding.shape}")
        if viewmats.shape != (batch, 20, 4, 4) or Ks.shape != (batch, 20, 3, 3):
            raise ValueError("Camera tensors must be [B,20,4,4] and [B,20,3,3]")
        trace_event(
            "model",
            "model.forward_inputs",
            tensors={
                "noisy_states": noisy_states,
                "clean_states": clean_states,
                "noisy_timesteps": noisy_timesteps,
                "prompt_embedding": prompt_embedding,
                "viewmats": viewmats,
                "Ks": Ks,
            },
            details={"layout": layout.name, "num_spans": layout.num_spans},
        )

        span_latents = self._assemble_span_latents(noisy_states, clean_states, layout)
        hidden = self._patchify(span_latents, layout)
        if hidden.shape[1] != layout.sequence_length:
            raise RuntimeError(f"Layout/patchify mismatch: {hidden.shape[1]} vs {layout.sequence_length}")
        hidden = trace_tensor(
            "embedding",
            "embedding.hidden_0",
            hidden,
            details={"destination": "transformer.block_00"},
            backward=True,
        )

        from wan.modules.model import sinusoidal_embedding_1d

        span_timesteps = self._span_timesteps(noisy_timesteps, layout)
        with debug_timer() as timing:
            sinusoidal_time = sinusoidal_embedding_1d(
                self.backbone.freq_dim, span_timesteps.flatten()
            ).type_as(hidden)
            raw_time = self.backbone.time_embedding(sinusoidal_time).unflatten(
                0, (batch, layout.num_spans)
            )
        trace_event(
            "timestep",
            "timestep.embedding",
            tensors={"sinusoidal": sinusoidal_time, "raw_time": raw_time},
            details={**timing, "frequency_dimension": self.backbone.freq_dim},
        )
        with debug_timer() as timing:
            modulation = self.backbone.time_projection(raw_time).unflatten(
                2, (6, self.backbone.dim)
            )
        trace_event(
            "timestep",
            "timestep.modulation_projection",
            tensors={"modulation": modulation},
            details={**timing, "six_channels": "shift,scale,gate repeated for attention/ffn"},
        )
        trace_gradient("timestep", "timestep.modulation", modulation)
        with debug_timer() as timing:
            context = self.backbone.text_embedding(prompt_embedding)
        trace_event(
            "text",
            "text.wan_projection",
            tensors={"t5_embedding": prompt_embedding, "context": context},
            details={**timing, "shape_change": "4096 -> 1536"},
        )
        trace_gradient("text", "text.context", context)
        with debug_timer() as timing:
            token_viewmats = self._expand_camera(viewmats, layout)
            token_Ks = self._expand_camera(Ks, layout)
        trace_event(
            "camera",
            "camera.expand_to_tokens",
            tensors={"token_viewmats": token_viewmats, "token_Ks": token_Ks},
            details={
                **timing,
                "mapping": "[C0,C0,C1,C1,...,C19,C19], each span repeated 1560 tokens",
            },
        )
        with debug_timer() as timing:
            coordinates = layout.token_coordinates.to(hidden.device)
        if self.backbone.freqs.device != hidden.device:
            self.backbone.freqs = self.backbone.freqs.to(hidden.device)
        frequencies = self.backbone.freqs
        trace_event(
            "rope",
            "rope.coordinates_and_frequencies",
            tensors={"coordinates": coordinates, "frequencies": frequencies},
            details={**timing, "coordinate_shape": "[sequence,time-height-width]"},
        )
        with debug_timer() as timing:
            block_mask, padding = self._block_mask(layout, hidden.device)
        trace_event(
            "mask",
            "mask.create_flex_block_mask",
            details={
                **timing,
                "valid_sequence": layout.sequence_length,
                "padded_sequence": layout.sequence_length + padding,
                "padding": padding,
                "noisy_rule": "N_t reads W_<t and N_t",
                "clean_rule": "W_t reads W_<=t",
            },
        )

        def execute(block: nn.Module, current: torch.Tensor, block_index: int) -> torch.Tensor:
            return _run_block(
                block,
                current,
                modulation,
                num_spans=layout.num_spans,
                tokens_per_span=layout.tokens_per_span,
                coordinates=coordinates,
                frequencies=frequencies,
                block_mask=block_mask,
                padding=padding,
                context=context,
                viewmats=token_viewmats,
                Ks=token_Ks,
                block_index=block_index,
            )

        for block_index, block in enumerate(self.backbone.blocks):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(
                    lambda current, selected=block, selected_index=block_index: execute(
                        selected, current, selected_index
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = execute(block, hidden, block_index)

        with debug_timer() as timing:
            noisy_indices = layout.noisy_token_indices.to(hidden.device)
            noisy_hidden = hidden.index_select(1, noisy_indices)
        noisy_span_indices = torch.tensor(
            [span.span_index for span in layout.noisy_spans],
            dtype=torch.long,
            device=hidden.device,
        )
        noisy_raw_time = raw_time.index_select(1, noisy_span_indices)
        trace_event(
            "selection",
            "output.select_noisy_tokens",
            tensors={"noisy_indices": noisy_indices, "noisy_hidden": noisy_hidden},
            details={**timing, "discarded": "all clean output positions"},
        )
        with debug_timer() as timing:
            head_values = self.backbone.head(noisy_hidden, noisy_raw_time.unsqueeze(2))
        trace_event(
            "head",
            "output.flow_head",
            tensors={"head_values": head_values, "noisy_raw_time": noisy_raw_time},
            details={**timing, "projection": "1536 -> 64=(1*2*2*16)"},
        )
        trace_gradient("head", "output.head_values", head_values)
        with debug_timer() as timing:
            flow = self._unpatchify_noisy(
                head_values,
                len(layout.noisy_spans),
                layout,
            )
        flow = trace_tensor(
            "head",
            "output.unpatchified_flow",
            flow,
            details={**timing, "shape_change": "[B,31200,64] -> [B,20,16,60,104]"},
            backward=True,
        )
        return InterleavedOutput(
            flow=flow,
            raw_time_embedding=noisy_raw_time,
            layout_name=layout.name,
        )

    def parameter_manifest(self) -> dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "adapter_only": 0,
        }
