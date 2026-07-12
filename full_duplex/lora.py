from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# These are the affine transforms inside a WanAttentionBlock. PRoPE's
# zero-started projection is deliberately left unchanged: the LoRA experiment
# adapts the checkpoint's ordinary self-attention, text cross-attention and FFN
# paths while retaining the already trained geometric path.
DEFAULT_LORA_TARGETS = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable low-rank residual.

    The residual is ``scaling * B(A(dropout(x)))``. ``B`` is initialized to
    exact zeros, so injection preserves the pretrained model function at step
    zero even though ``A`` uses a standard Kaiming initialization.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0,1), got {dropout}")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base).__name__}")

        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base_output = self.base(value)
        low_rank = F.linear(F.linear(self.dropout(value), self.lora_A), self.lora_B)
        return base_output + low_rank * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, out_features={self.base.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}"
        )


@dataclass(frozen=True)
class LoRAInjectionReport:
    total_backbone_blocks: int
    adapted_block_indices: tuple[int, ...]
    target_paths: tuple[str, ...]
    adapted_linear_count: int
    lora_parameter_count: int
    base_parameter_count_covered: int
    rank: int
    alpha: float
    dropout: float
    initialization: str = "A=Kaiming-uniform, B=zeros (exact zero initial residual)"

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["adapted_block_indices"] = list(self.adapted_block_indices)
        result["target_paths"] = list(self.target_paths)
        return result


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, separator, name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    if name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(name)] = replacement
    else:
        setattr(parent, name, replacement)


def inject_lora_into_last_blocks(
    blocks: nn.ModuleList,
    *,
    last_blocks: int,
    rank: int,
    alpha: float,
    dropout: float,
    target_paths: Iterable[str] = DEFAULT_LORA_TARGETS,
) -> LoRAInjectionReport:
    """Inject LoRA into the physical last ``last_blocks`` Wan blocks.

    This function intentionally selects indices relative to the complete
    checkpoint ModuleList, not relative to a resource-reduced execution prefix.
    The caller must therefore execute the complete block list.
    """

    total_blocks = len(blocks)
    if not 1 <= last_blocks <= total_blocks:
        raise ValueError(
            f"lora_last_blocks must be within [1,{total_blocks}], got {last_blocks}"
        )
    targets = tuple(target_paths)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("LoRA target paths must be non-empty and unique")

    adapted_indices = tuple(range(total_blocks - last_blocks, total_blocks))
    adapted_count = 0
    lora_parameters = 0
    covered_parameters = 0
    for block_index in adapted_indices:
        block = blocks[block_index]
        for path in targets:
            module = block.get_submodule(path)
            if isinstance(module, LoRALinear):
                raise RuntimeError(f"LoRA already injected at block {block_index} path {path}")
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"LoRA target block {block_index} path {path} is "
                    f"{type(module).__name__}, expected nn.Linear"
                )
            replacement = LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            _replace_submodule(block, path, replacement)
            adapted_count += 1
            lora_parameters += replacement.lora_A.numel() + replacement.lora_B.numel()
            covered_parameters += sum(parameter.numel() for parameter in module.parameters())

    return LoRAInjectionReport(
        total_backbone_blocks=total_blocks,
        adapted_block_indices=adapted_indices,
        target_paths=targets,
        adapted_linear_count=adapted_count,
        lora_parameter_count=lora_parameters,
        base_parameter_count_covered=covered_parameters,
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
    )


def lora_parameter_names(module: nn.Module) -> list[str]:
    return sorted(
        name
        for name, _parameter in module.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    )


def set_lora_trainable(module: nn.Module, trainable: bool = True) -> None:
    names = set(lora_parameter_names(module))
    for name, parameter in module.named_parameters():
        if name in names:
            parameter.requires_grad_(trainable)


def assert_zero_lora_residual(module: nn.Module) -> None:
    found = 0
    for child in module.modules():
        if isinstance(child, LoRALinear):
            found += 1
            if torch.count_nonzero(child.lora_B).item() != 0:
                raise AssertionError("Fresh LoRA B matrix is not exactly zero")
    if found == 0:
        raise AssertionError("No LoRALinear modules were found")
