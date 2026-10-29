"""Training-time RTC in sigma: 1 (noise) -> 0 (clean) convention.

The dataset/processor still supplies ordinary normalized expert trajectories.
A sampled clean prefix conditions the flow; only valid suffix tokens incur loss.
Reference: real-time-chunking/src/model.py, simulated_delay branch.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingRTCConfig:
    # Exclusive upper bound, matching the reference: d in [0, simulated_delay).
    # None preserves ordinary flow-matching training and checkpoint structure.
    simulated_delay: int | None = None

    def __post_init__(self):
        if self.simulated_delay is not None and (
            type(self.simulated_delay) is not int or self.simulated_delay < 1
        ):
            raise ValueError("training_rtc.simulated_delay must be a positive integer or null")

    @property
    def enabled(self):
        return self.simulated_delay is not None

    def sample_delays(self, action, action_is_pad=None):
        horizon = action.shape[1]
        if not self.enabled:
            return torch.zeros(action.shape[0], device=action.device, dtype=torch.long)
        if self.simulated_delay > horizon:
            raise ValueError("training_rtc.simulated_delay must not exceed the action horizon")
        # Stable equivalent of exp(arange(D)[::-1]); includes d=0 for bootstrap.
        probabilities = torch.softmax(
            -torch.arange(self.simulated_delay, device=action.device, dtype=torch.float32), dim=0
        )
        delays = torch.multinomial(probabilities, action.shape[0], replacement=True)
        if action_is_pad is not None:
            # The loader replicates episode tails. Always leave a real suffix
            # target if any exists, and never condition on replicated padding.
            valid = ~action_is_pad.bool()
            first_pad = torch.where(
                valid, horizon, torch.arange(horizon, device=action.device)
            ).amin(dim=1)
            delays = torch.minimum(delays, (first_pad - 1).clamp_min(0))
        return delays


def prepare_training_actions(action, noise, timestep, scheduler, config, action_is_pad=None):
    """Return noisy actions, scalar/token timesteps, and the clean prefix mask."""
    noisy = scheduler.add_noise(action, noise, timestep)
    if not config.enabled:
        return noisy, timestep, torch.zeros(action.shape[:2], device=action.device, dtype=torch.bool)
    delays = config.sample_delays(action, action_is_pad)
    prefix = torch.arange(action.shape[1], device=action.device)[None, :] < delays[:, None]
    noisy = torch.where(prefix[..., None], action, noisy)
    # Reference t=1 for clean prefix becomes sigma=0 here.
    token_timestep = torch.where(prefix, 0, timestep[:, None])
    return noisy, token_timestep, prefix


def suffix_loss_per_sample(prediction, target, prefix, action_is_pad=None):
    """Keep the existing mean-over-dimensions convention, mask prefix and pads."""
    token_loss = (prediction.float() - target.float()).square().mean(dim=-1)
    valid = ~prefix
    if action_is_pad is not None:
        valid = valid & ~action_is_pad.bool()
    return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


@dataclass(frozen=True)
class TrainingRTCPrefix:
    previous_actions: torch.Tensor  # [B,L,D], rebased/normalized for this request
    inference_delay_steps: int

    def __post_init__(self):
        if type(self.inference_delay_steps) is not int or self.inference_delay_steps < 0:
            raise ValueError("inference_delay_steps must be a nonnegative integer")
        if self.previous_actions.ndim != 3 or not torch.isfinite(self.previous_actions).all():
            raise ValueError("Training RTC prefix must be a finite [B,L,D] tensor")

    def clamp(self, latent):
        previous = self.previous_actions.to(device=latent.device, dtype=latent.dtype)
        if (
            latent.ndim != 3
            or previous.shape[0] != latent.shape[0]
            or previous.shape[2] != latent.shape[2]
            or previous.shape[1] > latent.shape[1]
        ):
            raise ValueError("Training RTC prefix and action latent shapes do not match")
        length = min(self.inference_delay_steps, previous.shape[1])
        if length >= latent.shape[1]:
            raise ValueError("Training RTC must leave at least one action to generate")
        mask = torch.arange(latent.shape[1], device=latent.device)[None, :] < length
        result = latent.clone()
        result[:, :length] = previous[:, :length]
        return result, mask

    def prepare(self, latent, timestep):
        latent, mask = self.clamp(latent)
        return latent, torch.where(mask, 0, timestep[:, None])
