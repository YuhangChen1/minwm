from __future__ import annotations

import unittest

import torch

from full_duplex.flow import add_flow_noise, denoising_sigmas, flow_step, flow_target


class FlowMatchingTest(unittest.TestCase):
    def test_checkpoint_sign_and_euler_solution(self) -> None:
        clean = torch.tensor([2.0])
        noise = torch.tensor([-1.0])
        target = flow_target(clean, noise)
        self.assertEqual(target.item(), -3.0)
        sigma = torch.tensor(0.75)
        sample = add_flow_noise(clean, noise, sigma)
        torch.testing.assert_close(sample, torch.tensor([-0.25]))
        recovered = flow_step(sample, target, sigma, torch.tensor(0.0))
        torch.testing.assert_close(recovered, clean)

    def test_shifted_schedule_is_fixed_and_decreasing(self) -> None:
        first = denoising_sigmas(10, 5.0, device=torch.device("cpu"))
        second = denoising_sigmas(10, 5.0, device=torch.device("cpu"))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first[0].item(), 1.0)
        self.assertEqual(first[-1].item(), 0.0)
        self.assertTrue(bool(torch.all(first[:-1] > first[1:])))


if __name__ == "__main__":
    unittest.main()
