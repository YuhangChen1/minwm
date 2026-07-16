from __future__ import annotations

import math
from typing import Any

import torch

from .layout import InterleavedLayout, SpanRole


def padded_sequence_length(sequence_length: int, multiple: int = 128) -> int:
    return math.ceil(sequence_length / multiple) * multiple


def dense_token_mask(layout: InterleavedLayout, padding: int = 0) -> torch.Tensor:
    valid_length = layout.sequence_length
    total_length = valid_length + padding
    mask = torch.zeros((total_length, total_length), dtype=torch.bool)
    span_mask = layout.span_visibility_matrix()
    token_spans = layout.token_span_indices
    mask[:valid_length, :valid_length] = span_mask[token_spans[:, None], token_spans[None, :]]
    if padding:
        indices = torch.arange(valid_length, total_length)
        mask[indices, indices] = True
    return mask


def create_flex_block_mask(
    layout: InterleavedLayout,
    *,
    device: torch.device | str,
    pad_to_multiple: int = 128,
    compile_mask: bool = False,
) -> tuple[Any, int]:
    from torch.nn.attention.flex_attention import create_block_mask

    valid_length = layout.sequence_length
    total_length = padded_sequence_length(valid_length, pad_to_multiple)
    pad = total_length - valid_length

    token_roles = torch.full(
        (total_length,), int(SpanRole.PADDING), dtype=torch.long, device=device
    )
    token_times = torch.full((total_length,), -1, dtype=torch.long, device=device)
    token_spans = torch.full((total_length,), -1, dtype=torch.long, device=device)
    token_roles[:valid_length] = layout.token_roles.to(device)
    token_times[:valid_length] = layout.token_times.to(device)
    token_spans[:valid_length] = layout.token_span_indices.to(device)
    block_states = layout.attention_block_states

    def mask_mod(batch_idx, head_idx, query_idx, key_idx):
        del batch_idx, head_idx
        query_valid = query_idx < valid_length
        key_valid = key_idx < valid_length
        query_role = token_roles[query_idx]
        key_role = token_roles[key_idx]
        query_block = torch.div(token_times[query_idx], block_states, rounding_mode="floor")
        key_block = torch.div(token_times[key_idx], block_states, rounding_mode="floor")

        clean_rule = (
            (query_role == int(SpanRole.CLEAN))
            & (key_role == int(SpanRole.CLEAN))
            & (key_block <= query_block)
        )
        noisy_rule = (query_role == int(SpanRole.NOISY)) & (
            (
                (key_role == int(SpanRole.CLEAN))
                & (key_block < query_block)
            )
            | (
                (key_role == int(SpanRole.NOISY))
                & (key_block == query_block)
            )
        )
        valid_rule = query_valid & key_valid & (clean_rule | noisy_rule)
        padding_diagonal = (~query_valid) & (~key_valid) & (query_idx == key_idx)
        return valid_rule | padding_diagonal

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_length,
        KV_LEN=total_length,
        device=device,
        _compile=compile_mask,
    )
    return block_mask, pad


def readable_span_mask(layout: InterleavedLayout) -> str:
    labels = layout.labels()
    matrix = layout.span_visibility_matrix()
    width = max(3, max(map(len, labels)))
    rows = [" " * (width + 1) + " ".join(label.rjust(width) for label in labels)]
    for label, values in zip(labels, matrix):
        cells = ["1" if bool(value) else "." for value in values]
        rows.append(label.rjust(width) + " " + " ".join(cell.rjust(width) for cell in cells))
    return "\n".join(rows)
