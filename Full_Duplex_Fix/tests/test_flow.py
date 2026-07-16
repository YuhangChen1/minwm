import sys
import unittest
from pathlib import Path

import torch

from Full_Duplex_Fix.flow import (
    flow_matching_losses,
    flow_to_clean,
    sample_flow_training_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Wan21"))


class FlowTest(unittest.TestCase):
    def setUp(self) -> None:
        from wan_utils.scheduler import FlowMatchScheduler

        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

    def test_target_and_clean_recovery(self) -> None:
        clean = torch.randn(1, 20, 2, 3, 4)
        batch = sample_flow_training_batch(
            clean, self.scheduler, generator=torch.Generator().manual_seed(9)
        )
        torch.testing.assert_close(batch.targets, batch.noise - clean)
        recovered = flow_to_clean(
            batch.noisy_latents, batch.targets, batch.timesteps, self.scheduler
        )
        torch.testing.assert_close(recovered, clean, rtol=2e-5, atol=2e-5)

    def test_perfect_prediction_has_zero_loss(self) -> None:
        prediction = torch.randn(2, 20, 3, 4, 5)
        weights = torch.ones(2, 20)
        losses = flow_matching_losses(prediction, prediction, weights)
        self.assertEqual(losses.total.item(), 0.0)
        self.assertEqual(losses.init.item(), 0.0)
        self.assertEqual(losses.transition.item(), 0.0)
        self.assertEqual(tuple(losses.per_state.shape), (20,))


if __name__ == "__main__":
    unittest.main()
