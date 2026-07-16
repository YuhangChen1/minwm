import unittest

import torch

from Full_Duplex_Fix.checkpoint import normalize_generator_state_dict


class CheckpointNormalizationTest(unittest.TestCase):
    def test_only_known_wrapper_prefixes_are_removed(self) -> None:
        tensor = torch.tensor([1.0])
        normalized = normalize_generator_state_dict(
            {
                "generator": {
                    "_checkpoint_wrapped_module.model._fsdp_wrapped_module.weight": tensor,
                    "_orig_mod.model.bias": tensor,
                }
            }
        )
        self.assertEqual(set(normalized), {"model.weight", "model.bias"})
        self.assertIs(normalized["model.weight"], tensor)


if __name__ == "__main__":
    unittest.main()
