from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import torch


BASE_SPECIAL_TOKENS = (
    "INPUT_STREAM_START",
    "INPUT_STREAM_END",
    "OUTPUT_STREAM_START",
    "OUTPUT_STREAM_END",
    "MODALITY_END",
    "NOISE_END",
    "NULL_WORLD_STATE",
    "MASKED_WORLD",
    "MASKED_CAMERA",
)


class TokenType(IntEnum):
    SPECIAL = 0
    WORLD_INPUT = 1
    CAMERA_INPUT = 2
    ACTION_INPUT = 3
    NOISE_INPUT = 4
    WORLD_OUTPUT = 5
    CAMERA_OUTPUT = 6


@dataclass(frozen=True)
class TokenSpan:
    turn: int
    name: str
    start: int
    end: int
    token_type: TokenType
    is_output_content: bool = False
    is_special: bool = False

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class SequenceLayout:
    spans: list[TokenSpan]
    turn_ids: torch.Tensor
    token_types: torch.Tensor
    special_ids: torch.Tensor
    output_content: torch.Tensor
    prediction_mask: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return int(self.turn_ids.numel())

    def span(self, turn: int, name: str) -> TokenSpan:
        matches = [span for span in self.spans if span.turn == turn and span.name == name]
        if len(matches) != 1:
            raise KeyError(f"Expected one span ({turn}, {name}), found {len(matches)}")
        return matches[0]


class SpecialTokenVocabulary:
    """Stable, explicit IDs for every independently trainable special token."""

    def __init__(self, max_time_index: int):
        if max_time_index < 0:
            raise ValueError("max_time_index must be non-negative")
        names = list(BASE_SPECIAL_TOKENS)
        names.extend(f"TIME_INDEX_{index}" for index in range(max_time_index + 1))
        self.name_to_id = {name: index for index, name in enumerate(names)}
        self.id_to_name = {index: name for name, index in self.name_to_id.items()}
        self.max_time_index = max_time_index

    def __len__(self) -> int:
        return len(self.name_to_id)

    def id(self, name: str) -> int:
        return self.name_to_id[name]

    def time_id(self, turn: int) -> int:
        if not 0 <= turn <= self.max_time_index:
            raise IndexError(f"turn {turn} exceeds max_time_index={self.max_time_index}")
        return self.id(f"TIME_INDEX_{turn}")

    def as_dict(self) -> dict[str, int]:
        return dict(self.name_to_id)


def build_layout(
    num_turns: int,
    num_world_tokens: int,
    num_camera_tokens: int,
    vocabulary: SpecialTokenVocabulary,
    turn_indices: list[int] | None = None,
) -> SequenceLayout:
    """Build the exact protocol layout without placing any ground-truth content."""
    if num_turns < 1 or num_world_tokens < 1 or num_camera_tokens < 1:
        raise ValueError("turn/world/camera token counts must be positive")

    if turn_indices is None:
        turn_indices = list(range(num_turns))
    if len(turn_indices) != num_turns or len(set(turn_indices)) != num_turns:
        raise ValueError("turn_indices must contain one unique absolute index per turn")
    if turn_indices != sorted(turn_indices):
        raise ValueError("turn_indices must be chronological")
    if turn_indices[-1] > vocabulary.max_time_index:
        raise ValueError("turn index exceeds vocabulary")

    spans: list[TokenSpan] = []
    turn_ids: list[int] = []
    token_types: list[int] = []
    special_ids: list[int] = []
    output_content: list[bool] = []
    prediction: list[bool] = []
    cursor = 0

    def append(
        turn: int,
        name: str,
        length: int,
        token_type: TokenType,
        *,
        special_name: str | None = None,
        is_output: bool = False,
        is_prediction: bool = False,
    ) -> None:
        nonlocal cursor
        is_special = special_name is not None
        spans.append(TokenSpan(turn, name, cursor, cursor + length, token_type, is_output, is_special))
        turn_ids.extend([turn] * length)
        token_types.extend([int(token_type)] * length)
        sid = vocabulary.id(special_name) if special_name is not None else -1
        special_ids.extend([sid] * length)
        output_content.extend([is_output] * length)
        prediction.extend([is_prediction] * length)
        cursor += length

    for turn in turn_indices:
        append(turn, "input_stream_start", 1, TokenType.SPECIAL, special_name="INPUT_STREAM_START")
        if turn == 0:
            append(turn, "null_world_state", 1, TokenType.SPECIAL, special_name="NULL_WORLD_STATE")
        append(turn, "world_input", num_world_tokens, TokenType.WORLD_INPUT)
        append(turn, "world_input_end", 1, TokenType.SPECIAL, special_name="MODALITY_END")
        append(turn, "camera_input", num_camera_tokens, TokenType.CAMERA_INPUT)
        append(turn, "camera_input_end", 1, TokenType.SPECIAL, special_name="MODALITY_END")
        append(turn, "action_input", 1, TokenType.ACTION_INPUT)
        append(turn, "action_input_end", 1, TokenType.SPECIAL, special_name="MODALITY_END")
        append(turn, "noise_input", num_world_tokens, TokenType.NOISE_INPUT)
        append(turn, "noise_end", 1, TokenType.SPECIAL, special_name="NOISE_END")
        append(turn, "input_stream_end", 1, TokenType.SPECIAL, special_name="INPUT_STREAM_END")
        append(turn, "output_stream_start", 1, TokenType.SPECIAL, special_name="OUTPUT_STREAM_START")
        append(
            turn,
            "world_output",
            num_world_tokens,
            TokenType.WORLD_OUTPUT,
            is_output=True,
            is_prediction=True,
        )
        append(turn, "world_output_end", 1, TokenType.SPECIAL, special_name="MODALITY_END")
        append(
            turn,
            "camera_output",
            num_camera_tokens,
            TokenType.CAMERA_OUTPUT,
            is_output=True,
            is_prediction=True,
        )
        append(turn, "camera_output_end", 1, TokenType.SPECIAL, special_name="MODALITY_END")
        append(turn, "output_stream_end", 1, TokenType.SPECIAL, special_name="OUTPUT_STREAM_END")
        append(turn, "time_index", 1, TokenType.SPECIAL, special_name=f"TIME_INDEX_{turn}")

    return SequenceLayout(
        spans=spans,
        turn_ids=torch.tensor(turn_ids, dtype=torch.long),
        token_types=torch.tensor(token_types, dtype=torch.long),
        special_ids=torch.tensor(special_ids, dtype=torch.long),
        output_content=torch.tensor(output_content, dtype=torch.bool),
        prediction_mask=torch.tensor(prediction, dtype=torch.bool),
    )


def build_attention_mask(layout: SequenceLayout) -> torch.Tensor:
    """Return [query,key] visibility with no current-GT or future leakage.

    Historical output content is visible because rollout replaces those slots with
    predictions. Current-turn output slots are placeholders and are never keys.
    Every special boundary token in the current/past turns remains visible.
    """
    turn_q = layout.turn_ids[:, None]
    turn_k = layout.turn_ids[None, :]
    past = turn_k < turn_q
    same = turn_k == turn_q
    current_non_output = same & ~layout.output_content[None, :]
    visible = past | current_non_output
    future = turn_k > turn_q
    if torch.any(visible & future):
        raise AssertionError("future visibility leak")
    return visible


def render_mask(mask: torch.Tensor, indices: Iterable[int] | None = None) -> str:
    if indices is None:
        indices = range(mask.shape[0])
    selected = list(indices)
    header = "    " + "".join(str(index % 10) for index in selected)
    rows = [header]
    for query in selected:
        glyphs = "".join("#" if bool(mask[query, key]) else "." for key in selected)
        rows.append(f"{query:03d} {glyphs}")
    return "\n".join(rows)
