"""CPU tests with production SparseActionDiT/SparseMoT and synthetic data only.

PYTHONPATH=src python scripts/test_training_rtc.py -v
Seed 42; no raw data, checkpoints, downloads, GPU, or robot access.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from torch import nn

from fasterwam.models.wan22.fasterwam import FasterWAM
from fasterwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fasterwam.models.wan22.sparse_action_dit import SparseActionDiT
from fasterwam.models.wan22.sparse_mot import SparseMoT
from fasterwam.models.wan22.training_rtc import (
    TrainingRTCConfig,
    TrainingRTCPrefix,
    prepare_training_actions,
    suffix_loss_per_sample,
)
from fasterwam.models.wan22.wan_video_dit import WanVideoDiT
from fasterwam.serve.widowxai_policy import WidowXAIFasterWAMPolicy


def tiny_model(config=None, checkpointing=False):
    video = WanVideoDiT(
        hidden_dim=24, in_dim=1, ffn_dim=48, out_dim=1, text_dim=8, freq_dim=8,
        eps=1e-6, patch_size=(1, 1, 1), num_heads=2, attn_head_dim=12,
        num_layers=2, has_image_input=False, seperated_timestep=True,
        video_attention_mask_mode="first_frame_causal",
    )
    action = SparseActionDiT(
        hidden_dim=16, action_dim=14, ffn_dim=32, text_dim=8, freq_dim=8,
        eps=1e-6, num_heads=2, attn_head_dim=12, num_layers=2,
        noncondition_num_heads=1, condition_layers=[0],
    )
    mot = SparseMoT(
        {"video": video, "action": action}, condition_layers=[0],
        mot_checkpoint_mixed_attn=checkpointing, video_kv_fusion="interval_weighted_sum",
    )
    vae = nn.Module()
    vae.temporal_downsample_factor, vae.upsampling_factor = 4, 8
    vae.model = SimpleNamespace(z_dim=1)
    model = FasterWAM(
        video, action, mot, vae, text_dim=8, proprio_dim=14,
        mot_action_video_attention_layers=[0], training_rtc=config,
    )
    # Bypass only the pretrained VAE. All action/video DiT and MoT paths are real.
    model._encode_video_latents = lambda value, **kw: value[:, :1, ::4, ::8, ::8].detach()
    model._encode_input_image_latents_tensor = lambda input_image, **kw: (
        input_image[:, :1, None, ::8, ::8].detach()
    )
    return model


class TrainingRTCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_delay_distribution_and_padding(self):
        config = TrainingRTCConfig(8)
        action = torch.zeros(20000, 8, 14)
        draws = config.sample_delays(action)
        self.assertGreater((draws == 0).sum(), (draws == 1).sum())
        self.assertTrue(((draws >= 0) & (draws < 8)).all())
        expected = torch.softmax(-torch.arange(8).float(), 0)
        observed = torch.bincount(draws, minlength=8).float() / len(draws)
        torch.testing.assert_close(observed, expected, atol=0.01, rtol=0)
        padding = torch.tensor([[False, True, True], [True, True, True]])
        delays = TrainingRTCConfig(3).sample_delays(torch.zeros(2, 3, 14), padding)
        self.assertEqual(delays.tolist(), [0, 0])
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                TrainingRTCConfig(value)
        with self.assertRaises(ValueError):
            TrainingRTCConfig(9).sample_delays(torch.zeros(1, 8, 14))

    def test_clean_prefix_token_times_and_loss_mask(self):
        scheduler = WanContinuousFlowMatchScheduler()
        action, noise = torch.randn(2, 4, 14), torch.randn(2, 4, 14)
        timestep = torch.tensor([400., 700.])
        config = TrainingRTCConfig(4)
        with patch.object(TrainingRTCConfig, "sample_delays", return_value=torch.tensor([0, 2])):
            noisy, times, prefix = prepare_training_actions(action, noise, timestep, scheduler, config)
        torch.testing.assert_close(noisy[1, :2], action[1, :2], rtol=0, atol=0)
        torch.testing.assert_close(noisy[0], scheduler.add_noise(action, noise, timestep)[0])
        self.assertEqual(times.tolist(), [[400., 400., 400., 400.], [0., 0., 700., 700.]])
        prediction = torch.randn_like(action, requires_grad=True)
        padding = torch.tensor([[False, False, False, True], [False, False, False, True]])
        loss = suffix_loss_per_sample(prediction, noise - action, prefix, padding).sum()
        loss.backward()
        self.assertEqual(prediction.grad[1, :2].count_nonzero(), 0)
        self.assertEqual(prediction.grad[:, 3].count_nonzero(), 0)
        self.assertGreater(prediction.grad[1, 2].abs().sum(), 0)
        all_pad = suffix_loss_per_sample(prediction, action, prefix, torch.ones_like(prefix))
        torch.testing.assert_close(all_pad, torch.zeros(2))
        ordinary, times, mask = prepare_training_actions(
            action, noise, timestep, scheduler, TrainingRTCConfig()
        )
        torch.testing.assert_close(ordinary, scheduler.add_noise(action, noise, timestep), rtol=0, atol=0)
        torch.testing.assert_close(times, timestep, rtol=0, atol=0)
        self.assertFalse(mask.any())

    def test_scalar_and_token_timesteps_are_compatible(self):
        expert = tiny_model().action_expert.eval()
        action, context = torch.randn(2, 8, 14), torch.randn(2, 3, 8)
        times = torch.tensor([400., 700.])
        scalar = expert(action, times, context)
        token = expert(action, times[:, None].expand(2, 8), context)
        torch.testing.assert_close(scalar, token)
        mixed = times[:, None].expand(2, 8).clone()
        mixed[:, :2] = 0
        result = expert(action, mixed, context)
        self.assertFalse(torch.equal(scalar, result))
        result.square().mean().backward()
        self.assertIsNotNone(expert.time_embedding[0].weight.grad)
        with self.assertRaises(ValueError):
            expert(action, torch.zeros(2, 7), context)

    def test_training_backward_checkpoint_and_disabled_regression(self):
        sample = {
            "video": torch.randn(2, 3, 5, 16, 16), "action": torch.randn(2, 8, 14),
            "proprio": torch.randn(2, 8, 14), "context": torch.randn(2, 3, 8),
            "context_mask": torch.ones(2, 3, dtype=torch.bool),
            "action_is_pad": torch.tensor([[False] * 8, [False] * 5 + [True] * 3]),
            "image_is_pad": torch.zeros(2, 5, dtype=torch.bool),
        }
        model = tiny_model(TrainingRTCConfig(4), checkpointing=True)
        before = model.action_expert.head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            loss, metrics = model.training_loss(sample)
            self.assertTrue(torch.isfinite(loss))
            losses.append(float(loss.detach()))
            self.assertIn("rtc_prefix_steps", metrics)
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        print(f"[synthetic-training] seed=42 optimizer_steps=3 losses={losses}")
        self.assertFalse(torch.equal(before, model.action_expert.head.weight))
        original = tiny_model()
        # No extra parameters or state-dict keys; old weight checkpoints still load.
        self.assertEqual(model.state_dict().keys(), original.state_dict().keys())
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "weights.pt")
            model.save_checkpoint(path, step=3)
            original.load_checkpoint(path)
        original.eval()
        model.eval()
        model.training_rtc = TrainingRTCConfig()
        torch.manual_seed(123)
        baseline = original.training_loss(sample)[0]
        torch.manual_seed(123)
        disabled = model.training_loss(sample)[0]
        torch.testing.assert_close(disabled, baseline, rtol=0, atol=0)

    def test_both_production_inference_paths_fix_prefix_every_step(self):
        for name in ("infer_action", "infer_action_one_pass_future_cache"):
            with self.subTest(path=name):
                model = tiny_model(TrainingRTCConfig(4)).eval().requires_grad_(False)
                previous = torch.randn(1, 4, 14)
                prefix = TrainingRTCPrefix(previous, 2)
                args = {
                    "prompt": None, "input_image": torch.randn(1, 3, 16, 16),
                    "action_horizon": 8, "num_video_frames": 5, "proprio": torch.zeros(14),
                    "context": torch.randn(3, 8), "context_mask": torch.ones(3, dtype=torch.bool),
                    "num_inference_steps": 3, "seed": 42,
                }
                seen = []
                real = model.action_expert.pre_dit

                def capture(*a, seen=seen, real=real, **kw):
                    seen.append((kw["action_tokens"].clone(), kw["timestep"].clone()))
                    return real(*a, **kw)

                with patch.object(model.action_expert, "pre_dit", side_effect=capture):
                    result = getattr(model, name)(**args, training_rtc_prefix=prefix)["action"]
                self.assertEqual(len(seen), 3)
                for latent, times in seen:
                    torch.testing.assert_close(latent[:, :2], previous[:, :2], rtol=0, atol=0)
                    self.assertEqual(times[:, :2].count_nonzero(), 0)
                    self.assertTrue((times[:, 2:] > 0).all())
                torch.testing.assert_close(result[:2], previous[0, :2], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(result).all())
                plain = getattr(model, name)(**args)["action"]
                for empty in (TrainingRTCPrefix(previous[:, :0], 9), TrainingRTCPrefix(previous, 0)):
                    torch.testing.assert_close(
                        getattr(model, name)(**args, training_rtc_prefix=empty)["action"], plain
                    )
                model.training_rtc = TrainingRTCConfig()
                with self.assertRaises(ValueError):
                    getattr(model, name)(**args, training_rtc_prefix=prefix)

    def test_factory_and_serving_configuration(self):
        from serve_widowxai_policy import parse_args

        from fasterwam.runtime import create_fasterwam

        with patch.object(FasterWAM, "from_wan22_pretrained") as factory:
            create_fasterwam(
                model_id="local", tokenizer_model_id="local", video_dit_config={},
                action_dit_config={}, action_scheduler={"train_shift": 5, "infer_shift": 5, "num_train_timesteps": 1000},
                training_rtc={"simulated_delay": 8},
            )
        self.assertEqual(factory.call_args.kwargs["training_rtc"], TrainingRTCConfig(8))
        config = Path(__file__).resolve().parents[1] / "configs/serve/train.yaml"
        with patch.object(sys, "argv", ["serve", "--serving-config", str(config)]):
            args = parse_args()
        self.assertTrue(args.train_rtc)
        self.assertFalse(args.rtc_config.enabled)
        with patch.object(sys, "argv", ["serve", "--serving-config", str(config), "--no-train-rtc"]):
            self.assertFalse(parse_args().train_rtc)

    def test_training_config_and_serving_context(self):
        directory = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(config_dir=directory, version_base="1.3"):
            cfg = compose(config_name="train", overrides=["task=widowxai_fasterwam_2wrist_training_rtc"])
        self.assertEqual(cfg.model.training_rtc.simulated_delay, 8)
        self.assertEqual(cfg.data.train.num_frames, 33)
        self.assertEqual(list(cfg.data.train.video_size), [384, 320])
        self.assertIsNone(cfg.keep_last_n_states)
        policy = WidowXAIFasterWAMPolicy.__new__(WidowXAIFasterWAMPolicy)
        policy.supports_train, policy.training_rtc_max_delay_steps = True, 7
        policy.action_horizon, policy.num_video_frames, policy.num_inference_steps, policy.seed = 8, 5, 3, 42
        policy.context, policy.context_mask = torch.zeros(3, 8), torch.ones(3, dtype=torch.bool)
        policy.preprocessor = SimpleNamespace(
            prepare_observation=lambda obs: (torch.zeros(3, 16, 16), torch.zeros(14)),
            normalize_previous_actions=lambda actions, qpos: torch.from_numpy(actions - qpos),
            denormalize_action=lambda action, proprio: action.numpy(),
        )
        received = []

        def infer(training_rtc_prefix=None):
            received.append(training_rtc_prefix)
            return {"action": torch.zeros(8, 14)}

        import numpy as np

        policy._infer_fn = infer
        policy.select_action(
            {"qpos": np.ones(14, dtype=np.float32)},
            chunk_context={"guidance_method": "train", "previous_actions": np.full((4, 14), 3, dtype=np.float32),
                           "inference_delay_steps": 2},
        )
        self.assertIsInstance(received[0], TrainingRTCPrefix)
        torch.testing.assert_close(received[0].previous_actions, torch.full((1, 4, 14), 2.))
        policy.supports_train = False
        with self.assertRaises(ValueError):
            policy.select_action({}, chunk_context={"guidance_method": "train"})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
