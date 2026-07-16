from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Iterable, Literal

import torch


class SpanRole(IntEnum):
    CLEAN = 0
    NOISY = 1
    PADDING = 2


@dataclass(frozen=True)
class Span:
    span_index: int
    role: SpanRole
    physical_time: int
    token_start: int
    token_end: int
    is_prediction_target: bool
    camera_index: int
    rope_time_id: int
    flow_timestep_source: str

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    @property
    def label(self) -> str:
        prefix = "N" if self.role == SpanRole.NOISY else "W"
        return f"{prefix}{self.physical_time}"


class InterleavedLayout:
    def __init__(
        self,
        spans: Iterable[Span],
        *,
        tokens_per_span: int,
        patch_height: int,
        patch_width: int,
        attention_block_states: int,
        name: str,
    ) -> None:
        self.spans = tuple(spans)
        self.tokens_per_span = int(tokens_per_span)
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.attention_block_states = int(attention_block_states)
        self.name = name
        self._validate()

    @classmethod
    def main(
        cls,
        tokens_per_span: int = 1560,
        patch_height: int = 30,
        patch_width: int = 52,
    ) -> "InterleavedLayout":
        spec: list[tuple[SpanRole, int]] = []
        for physical_time in range(20):
            spec.append((SpanRole.NOISY, physical_time))
            spec.append((SpanRole.CLEAN, physical_time))
        return cls._from_spec(
            spec,
            tokens_per_span=tokens_per_span,
            patch_height=patch_height,
            patch_width=patch_width,
            attention_block_states=1,
            name="main_40_span",
        )

    @classmethod
    def full_teacher_forcing(
        cls,
        order: Literal["interleaved", "original"],
        *,
        attention_block_states: int = 4,
        num_states: int = 20,
        tokens_per_span: int = 1560,
        patch_height: int = 30,
        patch_width: int = 52,
    ) -> "InterleavedLayout":
        if order == "interleaved":
            spec = [
                item
                for physical_time in range(num_states)
                for item in (
                    (SpanRole.NOISY, physical_time),
                    (SpanRole.CLEAN, physical_time),
                )
            ]
        elif order == "original":
            spec = [
                *((SpanRole.CLEAN, t) for t in range(num_states)),
                *((SpanRole.NOISY, t) for t in range(num_states)),
            ]
        else:
            raise ValueError(f"Unsupported order: {order}")
        return cls._from_spec(
            spec,
            tokens_per_span=tokens_per_span,
            patch_height=patch_height,
            patch_width=patch_width,
            attention_block_states=attention_block_states,
            name=f"full_{order}_{attention_block_states}_state_block",
        )

    @classmethod
    def _from_spec(
        cls,
        spec: Iterable[tuple[SpanRole, int]],
        **kwargs,
    ) -> "InterleavedLayout":
        tokens_per_span = int(kwargs["tokens_per_span"])
        spans = []
        for index, (role, physical_time) in enumerate(spec):
            spans.append(
                Span(
                    span_index=index,
                    role=role,
                    physical_time=physical_time,
                    token_start=index * tokens_per_span,
                    token_end=(index + 1) * tokens_per_span,
                    is_prediction_target=role == SpanRole.NOISY,
                    camera_index=physical_time,
                    rope_time_id=physical_time,
                    flow_timestep_source=(
                        f"noisy_timestep[{physical_time}]"
                        if role == SpanRole.NOISY
                        else "clean_zero"
                    ),
                )
            )
        return cls(spans, **kwargs)

    def _validate(self) -> None:
        if self.tokens_per_span != self.patch_height * self.patch_width:
            raise ValueError("tokens_per_span must equal patch_height * patch_width")
        if self.attention_block_states <= 0:
            raise ValueError("attention_block_states must be positive")
        for index, span in enumerate(self.spans):
            if span.span_index != index:
                raise ValueError("span indices must be contiguous")
            if span.token_start != index * self.tokens_per_span:
                raise ValueError("span token ranges must be contiguous")
            if span.token_count != self.tokens_per_span:
                raise ValueError("all spans must have the same token count")
            if span.camera_index != span.physical_time:
                raise ValueError("camera and physical time must align")
            if span.rope_time_id != span.physical_time:
                raise ValueError("RoPE and physical time must align")

    @property
    def num_spans(self) -> int:
        return len(self.spans)

    @property
    def sequence_length(self) -> int:
        return self.num_spans * self.tokens_per_span

    @cached_property
    def noisy_spans(self) -> tuple[Span, ...]:
        return tuple(span for span in self.spans if span.role == SpanRole.NOISY)

    @cached_property
    def clean_spans(self) -> tuple[Span, ...]:
        return tuple(span for span in self.spans if span.role == SpanRole.CLEAN)

    @cached_property
    def noisy_token_indices(self) -> torch.Tensor:
        return torch.cat(
            [torch.arange(span.token_start, span.token_end) for span in self.noisy_spans]
        )

    @cached_property
    def token_span_indices(self) -> torch.Tensor:
        return torch.arange(self.num_spans).repeat_interleave(self.tokens_per_span)

    @cached_property
    def token_roles(self) -> torch.Tensor:
        roles = torch.tensor([int(span.role) for span in self.spans], dtype=torch.long)
        return roles.repeat_interleave(self.tokens_per_span)

    @cached_property
    def token_times(self) -> torch.Tensor:
        times = torch.tensor([span.physical_time for span in self.spans], dtype=torch.long)
        return times.repeat_interleave(self.tokens_per_span)

    @cached_property
    def token_camera_indices(self) -> torch.Tensor:
        cameras = torch.tensor([span.camera_index for span in self.spans], dtype=torch.long)
        return cameras.repeat_interleave(self.tokens_per_span)

    @cached_property
    def token_coordinates(self) -> torch.Tensor:
        height = torch.arange(self.patch_height).view(-1, 1).expand(-1, self.patch_width)
        width = torch.arange(self.patch_width).view(1, -1).expand(self.patch_height, -1)
        spatial = torch.stack((height.reshape(-1), width.reshape(-1)), dim=-1)
        coordinates = []
        for span in self.spans:
            time = torch.full((self.tokens_per_span, 1), span.rope_time_id, dtype=torch.long)
            coordinates.append(torch.cat((time, spatial), dim=-1))
        return torch.cat(coordinates, dim=0)

    def span_block(self, span: Span) -> int:
        return span.physical_time // self.attention_block_states

    def can_attend(self, query: Span, key: Span) -> bool:
        query_block = self.span_block(query)
        key_block = self.span_block(key)
        if query.role == SpanRole.CLEAN:
            return key.role == SpanRole.CLEAN and key_block <= query_block
        if query.role == SpanRole.NOISY:
            prior_clean = key.role == SpanRole.CLEAN and key_block < query_block
            same_noisy_block = key.role == SpanRole.NOISY and key_block == query_block
            return prior_clean or same_noisy_block
        return False

    def span_visibility_matrix(self) -> torch.Tensor:
        matrix = torch.zeros((self.num_spans, self.num_spans), dtype=torch.bool)
        for query in self.spans:
            for key in self.spans:
                matrix[query.span_index, key.span_index] = self.can_attend(query, key)
        return matrix

    def labels(self) -> list[str]:
        return [span.label for span in self.spans]
