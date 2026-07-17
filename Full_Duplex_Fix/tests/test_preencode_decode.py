import unittest
from unittest.mock import patch

import numpy as np
import torch

from Full_Duplex_Fix.preencode import _decode_video


class _NativeBatch:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def asnumpy(self) -> np.ndarray:
        return self.value


class _Reader:
    def __init__(self, batch) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 77

    def get_avg_fps(self) -> float:
        return 24.0

    def get_batch(self, _indices):
        return self.batch


class DecodeVideoBridgeTest(unittest.TestCase):
    def _decode(self, batch) -> torch.Tensor:
        reader = _Reader(batch)
        config = {
            "video_path": "unused.mp4",
            "target_width": 3,
            "target_height": 2,
            "source_frames": 77,
            "source_fps": 24,
        }
        with patch("Full_Duplex_Fix.preencode.decord.VideoReader", return_value=reader):
            pixels, _ = _decode_video(config)
        return pixels

    def test_native_and_torch_decord_bridges_match(self) -> None:
        frames = torch.arange(77 * 2 * 3 * 3, dtype=torch.uint8).reshape(77, 2, 3, 3)
        native = self._decode(_NativeBatch(frames.numpy()))
        torch_bridge = self._decode(frames)
        torch.testing.assert_close(native, torch_bridge)
        self.assertEqual(tuple(torch_bridge.shape), (3, 77, 2, 3))


if __name__ == "__main__":
    unittest.main()
