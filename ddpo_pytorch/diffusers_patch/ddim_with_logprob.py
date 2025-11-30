# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/schedulers/scheduling_ddim.py
# with the following modifications:
# - It computes and returns the log prob of `prev_sample` given the UNet prediction.
# - Instead of `variance_noise`, it takes `prev_sample` as an optional argument. If `prev_sample` is provided,
#   it uses it to compute the log prob.
# - Timesteps can be a batched torch.Tensor.

from typing import Optional, Tuple, Union

import math
import torch

from diffusers.utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler


def _left_broadcast(t, shape):
    """Left-broadcast a tensor to a target shape.

    This helper reshapes `t` by appending singleton dimensions on the right and
    then uses `.broadcast_to` to match the given `shape`. This is useful for
    broadcasting per-timestep scalars (e.g., alphas) to match a full sample
    tensor shape.

    Args:
        t: Input tensor to broadcast. Typically 1D or scalar-like in this code.
        shape: Target shape to broadcast to (usually `sample.shape`).

    Returns:
        A view of `t` broadcast to `shape`.

    Raises:
        AssertionError: If `t.ndim > len(shape)`.
    """
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    """Compute DDIM variance between two timesteps.

    This follows the DDIM variance formula used for sampling, based on the
    cumulative product of alphas at the current and previous timesteps.

    Args:
        self: A `DDIMScheduler` instance providing `alphas_cumprod` and
            `final_alpha_cumprod`.
        timestep: Current timestep(s). Can be a scalar or a 1D tensor of
            timesteps (typically on CPU or GPU).
        prev_timestep: Previous timestep(s), aligned with `timestep`. Negative
            values are mapped to `final_alpha_cumprod`.

    Returns:
        A tensor of variances for each (prev_timestep, timestep) pair. The
        shape matches that of the gathered alpha products (usually 1D and later
        broadcast to the sample shape).
    """
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(
        timestep.device
    )
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def ddim_step_with_logprob(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    """Perform a DDIM sampling step and compute log-probability of the sample.

    This function is a patched version of the original DDIM scheduler step:
    in addition to computing the next sample (`prev_sample`), it also computes
    the log probability of that `prev_sample` under the Gaussian transition
    parameterized by the UNet prediction at the current timestep.

    The function supports:
    * Batched timesteps (`timestep` as a tensor).
    * Passing in an explicit `prev_sample` instead of sampling noise, in which
      case only the log-probability of that sample is computed.
    * The usual `prediction_type` modes used by `diffusers`
      (`"epsilon"`, `"sample"`, `"v_prediction"`).

    Args:
        self: A `DDIMScheduler` instance. Must have `num_inference_steps` set via
            `set_timesteps` before calling this function.
        model_output: Direct output from the diffusion model at the current
            timestep. Shape typically matches `sample`.
        timestep: Current discrete timestep(s) in the diffusion chain. Can be a
            scalar integer or a 1D tensor of timesteps; the implementation
            handles batched timesteps.
        sample: Current noisy sample `x_t` at the given `timestep`.
        eta: Weight of the stochasticity in the DDIM step. `eta = 0` yields a
            deterministic DDIM update; `eta > 0` adds noise (stochastic DDIM).
        use_clipped_model_output: If `True`, recompute `pred_epsilon` from the
            (possibly clipped) predicted original sample `x_0`. This mirrors the
            behavior in Glide and the original `diffusers` DDIM scheduler.
        generator: Optional PyTorch random number generator used to sample the
            noise term if `prev_sample` is not provided.
        prev_sample: Optional pre-computed previous sample `x_{t-1}`. If
            provided, no new noise is sampled; instead, this tensor is used to
            compute the log-probability under the Gaussian with mean
            `prev_sample_mean` and standard deviation `std_dev_t`.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]:
            * `prev_sample`: The predicted (or provided) sample at the previous
              timestep `x_{t-1}`, cast to the same dtype as `sample`.
            * `log_prob`: The log probability of `prev_sample` under the
              Gaussian transition from `x_t` to `x_{t-1}`, averaged over all
              non-batch dimensions. Shape is `(batch_size,)`.

    Raises:
        ValueError: If `self.num_inference_steps` has not been set via
            `set_timesteps`, or if both `generator` and `prev_sample` are
            provided at the same time.
        ValueError: If `self.config.prediction_type` is not one of
            `"epsilon"`, `"sample"`, or `"v_prediction"`.
    """
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep - self.config.num_train_timesteps // self.num_inference_steps
    )
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(
        sample.device
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (
            beta_prod_t**0.5
        ) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob
