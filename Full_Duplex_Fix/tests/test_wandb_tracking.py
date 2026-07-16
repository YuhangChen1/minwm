import argparse
import unittest

from Full_Duplex_Fix.wandb_tracking import (
    _public_wandb_config,
    add_wandb_arguments,
    flatten_wandb_metrics,
    wandb_overrides_from_args,
)


class WandbTrackingTest(unittest.TestCase):
    def test_public_config_keeps_model_tokens_but_removes_credentials(self) -> None:
        public = _public_wandb_config(
            {
                "tokens_per_span": 1560,
                "t5_tokenizer": "/models/umt5",
                "WANDB_API_KEY": "do-not-log",
                "service_api_key": "do-not-log-either",
            }
        )
        self.assertEqual(public["tokens_per_span"], 1560)
        self.assertEqual(public["t5_tokenizer"], "/models/umt5")
        self.assertNotIn("WANDB_API_KEY", public)
        self.assertNotIn("service_api_key", public)

    def test_training_metrics_include_each_state(self) -> None:
        payload = flatten_wandb_metrics(
            "train_step",
            {
                "step": 17,
                "loss": 0.25,
                "loss_init": 0.5,
                "loss_transition": 0.2,
                "per_state_loss": [float(index) for index in range(20)],
                "gradient_norm": 1.25,
            },
            step=17,
        )
        self.assertEqual(payload["trainer/global_step"], 17)
        self.assertEqual(payload["trainer/event"], "train_step")
        self.assertEqual(payload["train/loss"], 0.25)
        self.assertEqual(payload["train/per_state_loss/state_00"], 0.0)
        self.assertEqual(payload["train/per_state_loss/state_19"], 19.0)
        self.assertNotIn("train/per_state_loss", payload)

    def test_initial_evaluation_uses_eval_namespace(self) -> None:
        payload = flatten_wandb_metrics(
            "initial_evaluation",
            {
                "step": 0,
                "latent_mse": 0.75,
                "per_state_cosine": [0.1, 0.2],
                "evaluation_kind": "fixed",
            },
            step=0,
        )
        self.assertEqual(payload["eval/latent_mse"], 0.75)
        self.assertEqual(payload["eval/per_state_cosine/state_01"], 0.2)
        self.assertEqual(payload["eval/is_initial"], 1)
        self.assertNotIn("eval/evaluation_kind", payload)

    def test_cli_overrides_are_explicit_only(self) -> None:
        parser = argparse.ArgumentParser()
        add_wandb_arguments(parser)
        defaults = wandb_overrides_from_args(parser.parse_args([]))
        self.assertEqual(defaults, {})

        args = parser.parse_args(
            [
                "--no-wandb",
                "--wandb-mode",
                "offline",
                "--wandb-run-name",
                "smoke",
                "--wandb-tags",
                "one, two",
            ]
        )
        self.assertEqual(
            wandb_overrides_from_args(args),
            {
                "wandb_enabled": False,
                "wandb_mode": "offline",
                "wandb_run_name": "smoke",
                "wandb_tags": ["one", "two"],
            },
        )


if __name__ == "__main__":
    unittest.main()
