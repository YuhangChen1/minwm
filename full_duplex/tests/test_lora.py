from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from full_duplex.lora import (
    DEFAULT_LORA_TARGETS,
    LoRALinear,
    assert_zero_lora_residual,
    inject_lora_into_last_blocks,
    lora_parameter_names,
)


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class _Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_attn = _Attention(dim)
        self.cross_attn = _Attention(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))


class LoRATest(unittest.TestCase):
    def test_zero_initial_residual_and_gradient_boundary(self) -> None:
        torch.manual_seed(7)
        base = nn.Linear(6, 5)
        value = torch.randn(2, 3, 6)
        expected = base(value).detach()
        wrapped = LoRALinear(base, rank=2, alpha=2.0, dropout=0.0)
        actual = wrapped(value)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(wrapped.base.weight.requires_grad)
        actual.square().mean().backward()
        self.assertIsNone(wrapped.base.weight.grad)
        self.assertIsNotNone(wrapped.lora_B.grad)
        self.assertGreater(float(wrapped.lora_B.grad.norm()), 0.0)

    def test_only_physical_last_blocks_are_adapted(self) -> None:
        blocks = nn.ModuleList([_Block(8, 16) for _ in range(5)])
        report = inject_lora_into_last_blocks(
            blocks,
            last_blocks=2,
            rank=2,
            alpha=2.0,
            dropout=0.0,
        )
        self.assertEqual(report.adapted_block_indices, (3, 4))
        self.assertEqual(report.adapted_linear_count, 2 * len(DEFAULT_LORA_TARGETS))
        for index, block in enumerate(blocks):
            for path in DEFAULT_LORA_TARGETS:
                if index >= 3:
                    self.assertIsInstance(block.get_submodule(path), LoRALinear)
                else:
                    self.assertIsInstance(block.get_submodule(path), nn.Linear)
        names = lora_parameter_names(blocks)
        self.assertEqual(len(names), 2 * 2 * len(DEFAULT_LORA_TARGETS))
        self.assertTrue(all(name.startswith(("3.", "4.")) for name in names))
        assert_zero_lora_residual(blocks)

    def test_invalid_target_fails_loudly(self) -> None:
        blocks = nn.ModuleList([_Block(8, 16)])
        with self.assertRaisesRegex(AttributeError, "no attribute"):
            inject_lora_into_last_blocks(
                blocks,
                last_blocks=1,
                rank=2,
                alpha=2.0,
                dropout=0.0,
                target_paths=("self_attn.missing",),
            )


if __name__ == "__main__":
    unittest.main()
