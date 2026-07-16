import sys
import unittest
from pathlib import Path

import torch

from Full_Duplex_Fix.rope import explicit_rope_apply


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Wan21"))


class ExplicitRoPETest(unittest.TestCase):
    def test_matches_original_contiguous_rope(self) -> None:
        from wan.modules.model import rope_apply, rope_params

        torch.manual_seed(7)
        batch, frames, height, width, heads, head_dim = 1, 3, 2, 4, 2, 12
        tensor = torch.randn(batch, frames * height * width, heads, head_dim)
        complex_dim = head_dim // 2
        frequencies = torch.cat(
            (
                rope_params(32, 2 * (complex_dim - 2 * (complex_dim // 3))),
                rope_params(32, 2 * (complex_dim // 3)),
                rope_params(32, 2 * (complex_dim // 3)),
            ),
            dim=1,
        )
        coordinates = torch.tensor(
            [
                (t, h, w)
                for t in range(frames)
                for h in range(height)
                for w in range(width)
            ]
        )
        expected = rope_apply(tensor, torch.tensor([[frames, height, width]]), frequencies)
        actual = explicit_rope_apply(tensor, coordinates, frequencies)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_duplicate_physical_time_gets_same_rotation(self) -> None:
        from wan.modules.model import rope_params

        head_dim = 12
        complex_dim = head_dim // 2
        frequencies = torch.cat(
            (
                rope_params(32, 2 * (complex_dim - 2 * (complex_dim // 3))),
                rope_params(32, 2 * (complex_dim // 3)),
                rope_params(32, 2 * (complex_dim // 3)),
            ),
            dim=1,
        )
        token = torch.randn(1, 1, 2, head_dim)
        tensor = token.repeat(1, 2, 1, 1)
        coordinates = torch.tensor([[5, 1, 2], [5, 1, 2]])
        output = explicit_rope_apply(tensor, coordinates, frequencies)
        torch.testing.assert_close(output[:, 0], output[:, 1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
