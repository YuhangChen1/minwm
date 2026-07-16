import unittest

import torch

from Full_Duplex_Fix.layout import InterleavedLayout
from Full_Duplex_Fix.mask import dense_token_mask, padded_sequence_length


class MaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = InterleavedLayout.main(
            tokens_per_span=2, patch_height=1, patch_width=2
        )
        self.mask = dense_token_mask(self.layout)

    def _range(self, label: str) -> range:
        span = next(span for span in self.layout.spans if span.label == label)
        return range(span.token_start, span.token_end)

    def _all(self, query: str, key: str) -> bool:
        q = list(self._range(query))
        k = list(self._range(key))
        return bool(self.mask[q][:, k].all())

    def _none(self, query: str, key: str) -> bool:
        q = list(self._range(query))
        k = list(self._range(key))
        return not bool(self.mask[q][:, k].any())

    def test_required_visibility(self) -> None:
        self.assertTrue(self._all("N0", "N0"))
        self.assertTrue(self._none("N0", "W0"))
        self.assertTrue(self._all("W0", "W0"))
        self.assertTrue(self._none("W0", "N0"))
        self.assertTrue(self._all("N1", "W0"))
        self.assertTrue(self._all("N1", "N1"))
        self.assertTrue(self._none("N1", "N0"))
        self.assertTrue(self._none("N1", "W1"))
        self.assertTrue(self._all("W1", "W0"))
        self.assertTrue(self._all("W1", "W1"))
        self.assertTrue(self._none("N19", "W19"))
        self.assertTrue(self._all("W19", "W0"))
        self.assertTrue(self._all("W19", "W19"))
        for noisy in (span.label for span in self.layout.noisy_spans):
            self.assertTrue(self._none("W1", noisy))
            self.assertTrue(self._none(noisy, "W19"))

    def test_padding_only_sees_own_diagonal(self) -> None:
        mask = dense_token_mask(self.layout, padding=3)
        valid = self.layout.sequence_length
        self.assertFalse(mask[:valid, valid:].any())
        self.assertFalse(mask[valid:, :valid].any())
        self.assertTrue(torch.equal(mask[valid:, valid:], torch.eye(3, dtype=torch.bool)))

    def test_real_padding_length(self) -> None:
        self.assertEqual(padded_sequence_length(62400), 62464)


if __name__ == "__main__":
    unittest.main()
