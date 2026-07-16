import json
import unittest
from pathlib import Path

import torch

from Full_Duplex_Fix.data import load_cached_sample


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "Full_Duplex_Fix/cache/smallest_000000"


class CachedDataTest(unittest.TestCase):
    @unittest.skipUnless(CACHE_PATH.is_dir(), "real preencode cache is not present")
    def test_real_single_sample_contract(self) -> None:
        sample = load_cached_sample(CACHE_PATH)
        self.assertEqual(sample.world_latents.shape, (20, 16, 60, 104))
        self.assertTrue(torch.isfinite(sample.world_latents).all())
        self.assertEqual(sample.viewmats.shape, (20, 4, 4))
        self.assertEqual(sample.Ks.shape, (20, 3, 3))
        self.assertEqual(len(sample.metadata["action_alignment"]), 19)
        self.assertEqual(sample.metadata["video"]["frame_count"], 77)
        reloaded = torch.load(CACHE_PATH / "tensors.pt", map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(sample.world_latents, reloaded["world_latents"]))
        with (CACHE_PATH / "metadata.json").open("r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["preprocessing_hash"], sample.metadata["preprocessing_hash"])


if __name__ == "__main__":
    unittest.main()
