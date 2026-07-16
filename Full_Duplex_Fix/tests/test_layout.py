import unittest

import torch

from Full_Duplex_Fix.layout import InterleavedLayout, SpanRole


class LayoutTest(unittest.TestCase):
    def test_main_layout_exact_contract(self) -> None:
        layout = InterleavedLayout.main()
        self.assertEqual(layout.num_spans, 40)
        self.assertEqual(layout.sequence_length, 62400)
        self.assertEqual(len(layout.noisy_spans), 20)
        self.assertEqual(len(layout.clean_spans), 20)
        self.assertEqual(layout.labels()[:5], ["N0", "W0", "N1", "W1", "N2"])
        self.assertEqual(layout.labels()[-4:], ["N18", "W18", "N19", "W19"])
        self.assertIn("W19", layout.labels())
        self.assertEqual(layout.noisy_token_indices.numel(), 20 * 1560)
        self.assertEqual(
            [span.physical_time for span in layout.spans],
            [value for time in range(20) for value in (time, time)],
        )
        self.assertEqual(
            [span.camera_index for span in layout.spans],
            [value for time in range(20) for value in (time, time)],
        )
        self.assertEqual(layout.token_camera_indices.numel(), 62400)
        self.assertTrue(torch.equal(layout.token_coordinates[0], torch.tensor([0, 0, 0])))
        self.assertTrue(
            torch.equal(layout.token_coordinates[1560], torch.tensor([0, 0, 0]))
        )
        self.assertTrue(
            torch.equal(layout.token_coordinates[-1], torch.tensor([19, 29, 51]))
        )
        for span in layout.spans:
            self.assertEqual(span.camera_index, span.physical_time)
            self.assertEqual(span.rope_time_id, span.physical_time)
            self.assertEqual(span.is_prediction_target, span.role == SpanRole.NOISY)

    def test_single_state_visibility(self) -> None:
        layout = InterleavedLayout.main(tokens_per_span=2, patch_height=1, patch_width=2)
        spans = {span.label: span for span in layout.spans}
        self.assertTrue(layout.can_attend(spans["N0"], spans["N0"]))
        self.assertFalse(layout.can_attend(spans["N0"], spans["W0"]))
        self.assertTrue(layout.can_attend(spans["W0"], spans["W0"]))
        self.assertFalse(layout.can_attend(spans["W0"], spans["N0"]))
        self.assertTrue(layout.can_attend(spans["N1"], spans["W0"]))
        self.assertFalse(layout.can_attend(spans["N1"], spans["N0"]))
        self.assertFalse(layout.can_attend(spans["N1"], spans["W1"]))
        self.assertTrue(layout.can_attend(spans["W1"], spans["W0"]))
        self.assertTrue(layout.can_attend(spans["W1"], spans["W1"]))
        self.assertFalse(layout.can_attend(spans["N19"], spans["W19"]))
        self.assertTrue(layout.can_attend(spans["W19"], spans["W0"]))
        self.assertTrue(layout.can_attend(spans["W19"], spans["W19"]))
        for noisy in layout.noisy_spans:
            self.assertFalse(layout.can_attend(noisy, spans["W19"]))
        for query in layout.spans:
            for key in layout.spans:
                if key.physical_time > query.physical_time:
                    self.assertFalse(layout.can_attend(query, key))

    def test_four_state_equivalence_layouts_have_same_semantics(self) -> None:
        interleaved = InterleavedLayout.full_teacher_forcing(
            "interleaved", tokens_per_span=2, patch_height=1, patch_width=2
        )
        original = InterleavedLayout.full_teacher_forcing(
            "original", tokens_per_span=2, patch_height=1, patch_width=2
        )
        self.assertEqual(set(interleaved.labels()), set(original.labels()))
        for query_label in interleaved.labels():
            for key_label in interleaved.labels():
                query_i = next(span for span in interleaved.spans if span.label == query_label)
                key_i = next(span for span in interleaved.spans if span.label == key_label)
                query_o = next(span for span in original.spans if span.label == query_label)
                key_o = next(span for span in original.spans if span.label == key_label)
                self.assertEqual(
                    interleaved.can_attend(query_i, key_i),
                    original.can_attend(query_o, key_o),
                )


if __name__ == "__main__":
    unittest.main()
