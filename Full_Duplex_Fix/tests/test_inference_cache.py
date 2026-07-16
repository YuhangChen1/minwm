import unittest

import torch

from Full_Duplex_Fix.inference import AutoregressiveSampler


class _FakeGenerator(torch.nn.Module):
    def forward(
        self,
        *,
        noisy_image_or_video,
        kv_cache,
        current_start,
        prope_kv_cache,
        **kwargs,
    ):
        del kwargs
        endpoint = current_start + 2
        for cache in (kv_cache, prope_kv_cache):
            for layer in cache:
                layer["global_end_index"].fill_(endpoint)
                layer["local_end_index"].fill_(endpoint)
        return torch.zeros_like(noisy_image_or_video), noisy_image_or_video


class _FakeScheduler:
    timesteps = torch.tensor([900.0, 100.0])

    @staticmethod
    def step(flow, timestep, latent, return_dict=False):
        del flow, timestep, return_dict
        return (latent,)


class _FakeSampler(AutoregressiveSampler):
    def _scheduler(self):
        return _FakeScheduler()


class InferenceCacheTest(unittest.TestCase):
    def test_denoising_overwrites_and_clean_rerun_advances_once(self) -> None:
        config = {
            "num_states": 20,
            "tokens_per_span": 2,
            "num_transformer_blocks": 2,
            "num_heads": 1,
            "head_dim": 1,
            "local_attention_states": 20,
            "text_length": 512,
            "guidance_scale": 3.0,
            "sampling_steps": 2,
        }
        sampler = _FakeSampler(_FakeGenerator(), config, device="cpu", dtype=torch.float32)
        noise = torch.randn(1, 20, 16, 60, 104)
        generated, provenance = sampler.sample(
            initial_noises=noise,
            positive_prompt=torch.zeros(1, 512, 4096),
            negative_prompt=torch.zeros(1, 512, 4096),
            viewmats=torch.eye(4).view(1, 1, 4, 4).repeat(1, 20, 1, 1),
            Ks=torch.eye(3).view(1, 1, 3, 3).repeat(1, 20, 1, 1),
            show_progress=False,
        )
        torch.testing.assert_close(generated, noise)
        self.assertEqual(provenance["normal_and_prope_cache_end"], 40)
        self.assertFalse(provenance["ground_truth_latents_used"])


if __name__ == "__main__":
    unittest.main()
