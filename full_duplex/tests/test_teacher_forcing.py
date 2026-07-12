from __future__ import annotations

import unittest

import torch

from full_duplex.teacher_forcing_training import (
    TeacherForcedTransitionTrainer,
    previous_ground_truth_world_input,
)


class TeacherForcedTransitionTest(unittest.TestCase):
    def test_world_input_is_zero_then_exact_previous_ground_truth(self) -> None:
        states = torch.arange(20 * 2 * 3 * 4, dtype=torch.float32).reshape(20, 2, 3, 4)

        turn_zero = previous_ground_truth_world_input(states, 0)
        self.assertEqual(tuple(turn_zero.shape), (1, 1, 2, 3, 4))
        self.assertEqual(torch.count_nonzero(turn_zero).item(), 0)

        for turn in range(1, 19):
            actual = previous_ground_truth_world_input(states, turn)
            expected = states[turn : turn + 1].unsqueeze(0)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.data_ptr(), expected.data_ptr())
            target = states[turn + 1 : turn + 2].unsqueeze(0)
            self.assertFalse(torch.equal(actual, target))

    def test_invalid_transition_index_fails(self) -> None:
        states = torch.zeros(20, 2, 3, 4)
        with self.assertRaises(IndexError):
            previous_ground_truth_world_input(states, -1)
        with self.assertRaises(IndexError):
            previous_ground_truth_world_input(states, 19)
        with self.assertRaises(ValueError):
            previous_ground_truth_world_input(states.unsqueeze(0), 0)

    def test_historical_predictions_have_no_cross_turn_graph(self) -> None:
        prediction = torch.randn(1, 1, 2, 3, 4, requires_grad=True) * 2
        camera_prediction = torch.randn(1, 13, requires_grad=True) * 2
        history = TeacherForcedTransitionTrainer._detached_history_turn(
            turn_index=3,
            world_input=torch.randn(1, 1, 2, 3, 4),
            camera_input=torch.randn(1, 13),
            action_id=torch.tensor([1]),
            noise_input=torch.randn(1, 1, 2, 3, 4),
            world_output=prediction,
            camera_output=camera_prediction,
        )
        for tensor in (
            history.world_input,
            history.camera_input,
            history.noise_input,
            history.world_output,
            history.camera_output,
        ):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

    def test_sequential_backward_matches_mean_loss_gradient(self) -> None:
        weight_combined = torch.tensor(2.0, requires_grad=True)
        combined = torch.stack(
            [(weight_combined * value - target).square() for value, target in ((1.0, 3.0), (4.0, -2.0))]
        ).mean()
        combined.backward()

        weight_sequential = torch.tensor(2.0, requires_grad=True)
        for value, target in ((1.0, 3.0), (4.0, -2.0)):
            ((weight_sequential * value - target).square() / 2).backward()
        torch.testing.assert_close(weight_sequential.grad, weight_combined.grad)


if __name__ == "__main__":
    unittest.main()
