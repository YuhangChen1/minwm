from __future__ import annotations

import contextlib
import io
import unittest

import torch

from full_duplex.model import padded_attention_length
from full_duplex.tokens import (
    BASE_SPECIAL_TOKENS,
    SpecialTokenVocabulary,
    build_attention_mask,
    build_layout,
    render_mask,
)


class TokenProtocolTest(unittest.TestCase):
    def test_attention_compile_buckets(self) -> None:
        self.assertEqual(padded_attention_length(99), 128)
        self.assertEqual(padded_attention_length(198), 256)
        self.assertEqual(padded_attention_length(297), 512)
        self.assertEqual(padded_attention_length(1881), 2048)
        self.assertEqual(padded_attention_length(4695), 8192)
        self.assertEqual(padded_attention_length(129, "multiple_of_128"), 256)

    def test_special_vocabulary_is_independent_and_stable(self) -> None:
        vocabulary = SpecialTokenVocabulary(max_time_index=3)
        self.assertEqual(len(set(vocabulary.as_dict().values())), len(vocabulary))
        self.assertTrue(all(name in vocabulary.as_dict() for name in BASE_SPECIAL_TOKENS))
        self.assertEqual([vocabulary.time_id(i) for i in range(4)], list(range(9, 13)))

    def test_exact_protocol_and_masks(self) -> None:
        vocabulary = SpecialTokenVocabulary(max_time_index=3)
        layout = build_layout(2, num_world_tokens=2, num_camera_tokens=1, vocabulary=vocabulary)
        names0 = [span.name for span in layout.spans if span.turn == 0]
        self.assertEqual(names0, [
        "input_stream_start",
        "null_world_state",
        "world_input",
        "world_input_end",
        "camera_input",
        "camera_input_end",
        "action_input",
        "action_input_end",
        "noise_input",
        "noise_end",
        "input_stream_end",
        "output_stream_start",
        "world_output",
        "world_output_end",
        "camera_output",
        "camera_output_end",
        "output_stream_end",
        "time_index",
        ])
        names1 = [span.name for span in layout.spans if span.turn == 1]
        self.assertNotIn("null_world_state", names1)

        mask = build_attention_mask(layout)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, (layout.sequence_length, layout.sequence_length))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print(render_mask(mask))

        turn0_query = layout.span(0, "world_output").start
        turn1_query = layout.span(1, "world_output").start
        turn0_time = layout.span(0, "time_index").start
        turn1_input = layout.span(1, "world_input").start
        turn1_world_gt = layout.span(1, "world_output").start
        turn1_camera_gt = layout.span(1, "camera_output").start
        turn1_boundary = layout.span(1, "output_stream_end").start

    # Future turns are invisible to earlier queries.
        self.assertFalse(bool(mask[turn0_query, turn1_input]))
    # Every historical token, including prior predictions and boundaries, is visible.
        self.assertTrue(bool(mask[turn1_query, turn0_query]))
        self.assertTrue(bool(mask[turn1_query, turn0_time]))
    # Current inputs and visible boundary tokens are visible.
        self.assertTrue(bool(mask[turn1_query, turn1_input]))
        self.assertTrue(bool(mask[turn1_query, turn1_boundary]))
    # Current GT output content is represented by masks and never becomes a key.
        self.assertFalse(bool(mask[turn1_query, turn1_world_gt]))
        self.assertFalse(bool(mask[turn1_query, turn1_camera_gt]))
    # Prediction mask includes content only, never special tokens.
        expected_predictions = 2 * (2 + 1)
        self.assertEqual(int(layout.prediction_mask.sum()), expected_predictions)
        self.assertFalse(bool(torch.any(layout.prediction_mask & layout.special_ids.ge(0))))

        captured = output.getvalue()
        self.assertIn("#", captured)
        self.assertIn(".", captured)


if __name__ == "__main__":
    unittest.main()
