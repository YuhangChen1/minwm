from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from full_duplex.tokens import SpecialTokenVocabulary, build_layout


class RealCacheShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cache_dir = cls.root / "full_duplex/cache/smallest_000000"
        cls.cache = torch.load(cache_dir / "tensors.pt", map_location="cpu", weights_only=True)
        cls.metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))

    def test_real_cache_shapes_and_finiteness(self) -> None:
        self.assertEqual(tuple(self.cache["world_state_latents"].shape), (20, 16, 60, 104))
        self.assertEqual(tuple(self.cache["prompt_embedding"].shape), (512, 4096))
        self.assertEqual(tuple(self.cache["prompt_attention_mask"].shape), (512,))
        self.assertEqual(tuple(self.cache["camera"].shape), (20, 13))
        self.assertEqual(tuple(self.cache["action_ids"].shape), (19,))
        for tensor in self.cache.values():
            if torch.is_floating_point(tensor):
                self.assertTrue(bool(torch.isfinite(tensor).all()))
        self.assertEqual(self.metadata["video"]["original_frame_count"], 77)
        self.assertEqual(self.metadata["world_state"]["latent_frame_count"], 20)
        self.assertEqual(self.metadata["num_micro_turns"], 19)

    def test_full_protocol_shape(self) -> None:
        layout = build_layout(19, 1560, 1, SpecialTokenVocabulary(31))
        self.assertEqual(layout.sequence_length, 89187)
        self.assertEqual(int(layout.prediction_mask.sum()), 19 * 1561)


if __name__ == "__main__":
    unittest.main()
