#!/usr/bin/env python3
"""
Generate a single diffusion trajectory under a given policy
(using the JPEG compressibility reward) and save all intermediate
images + scores for visualization.

Usage:

    # Stable Diffusion (no DDPO weights)
    python visualize/visualize_traj.py --checkpoint None

    # DDPO trajectory (with a given checkpoint)
    python visualize/visualize_traj.py \
        --checkpoint logs/<run>/checkpoints/checkpoint_XX

Outputs (depending on checkpoint):

  trajectory_sd/ or trajectory_ddpo/
    step_000.png, step_001.png, ...
    scores.npy        # shape (T+1,)
    timesteps.npy     # shape (T,)
    prompt.txt
"""

import os
import argparse
import numpy as np
from PIL import Image
import imp

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel

import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob


# ---------------------------------------------------------------------
# CONSTANT SETTINGS
# ---------------------------------------------------------------------

# Number of DDIM steps to visualize (should match or be <= training num_steps).
NUM_INFERENCE_STEPS = 50

# Optional: fix a specific prompt (set to None to sample via prompt_fn)
# FIXED_PROMPT = None  # e.g. "a golden retriever running in a field"
FIXED_PROMPT = "a golden retriever running in a field"


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------


def sanity_check_vanilla_sd(prompt: str, output_dir: str):
    """Quick check: does vanilla SD v1-4 produce a sensible image for this prompt?"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    ).to(device)
    pipe.safety_checker = None

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
        out = pipe(prompt, num_inference_steps=50, guidance_scale=7.5)
    img = out.images[0]
    out_path = os.path.join(output_dir, "vanilla_sd.png")
    img.save(out_path)
    print("Saved vanilla SD image to", out_path)


def load_config():
    """Load the compressibility config from config/dgx.py."""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.dirname(this_dir)
    dgx_path = os.path.join(repo_root, "config", "dgx.py")

    if not os.path.exists(dgx_path):
        raise FileNotFoundError(f"Could not find dgx.py at {dgx_path}")

    dgx = imp.load_source("dgx", dgx_path)
    config = dgx.compressibility()
    return config


def load_pipeline_and_weights(config, device, dtype, checkpoint_dir):
    """Load SD pipeline and apply trained weights (LoRA or full UNet)."""
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )

    # Disable safety checker & use DDIM scheduler
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Move models to device
    pipeline.vae.to(device, dtype=dtype)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.unet.to(device, dtype=dtype)

    if checkpoint_dir is None:
        # Plain Stable Diffusion
        return pipeline

    if config.use_lora:
        # LoRA case: `train.py` saved attention processors in `checkpoint_dir`
        print(f"Loading LoRA attention processors from {checkpoint_dir}")
        pipeline.unet.load_attn_procs(checkpoint_dir)
    else:
        # Full UNet case: `train.py` saved under subfolder 'unet'
        print(f"Loading full UNet weights from {checkpoint_dir}/unet")
        unet = UNet2DConditionModel.from_pretrained(checkpoint_dir, subfolder="unet")
        pipeline.unet = unet.to(device, dtype=dtype)

    return pipeline


def generate_trajectory(checkpoint_dir: str | None):
    """
    Generate and save a single compressibility trajectory.

    If checkpoint_dir is None → output to 'trajectory_sd'.
    Otherwise → output to 'trajectory_ddpo'.
    """
    output_dir = "trajectory_sd" if checkpoint_dir is None else "trajectory_ddpo"
    os.makedirs(output_dir, exist_ok=True)

    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match the inference dtype logic from train.py as closely as possible
    inference_dtype = torch.float32
    if getattr(config, "mixed_precision", None) == "fp16":
        inference_dtype = torch.float16
    elif getattr(config, "mixed_precision", None) == "bf16":
        inference_dtype = torch.bfloat16

    pipeline = load_pipeline_and_weights(
        config, device=device, dtype=inference_dtype, checkpoint_dir=checkpoint_dir
    )
    pipeline.set_progress_bar_config(disable=True)

    # -------------------------------------------------------------------------
    # Prompt
    # -------------------------------------------------------------------------
    if FIXED_PROMPT is not None:
        prompt = FIXED_PROMPT
        prompt_metadata = {}
    else:
        prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
        prompt, prompt_metadata = prompt_fn(**config.prompt_fn_kwargs)

    print(f"Using prompt: {prompt!r}")
    with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        f.write(prompt + "\n")
        f.write(str(prompt_metadata) + "\n")

    if checkpoint_dir is None:
        # Just for sanity: also dump a vanilla SD image with the same prompt
        sanity_check_vanilla_sd(prompt, output_dir)

    # encode positive prompt
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    prompt_ids = tokenizer(
        [prompt],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids.to(device)
    prompt_embeds = text_encoder(prompt_ids)[0]  # (1, seq, dim)

    # encode negative prompt (empty string), and repeat to match batch_size=1
    neg_ids = tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids.to(device)
    neg_embed = text_encoder(neg_ids)[0]  # (1, seq, dim)
    sample_neg_prompt_embeds = neg_embed  # repeat(1,1,1) is the same

    pipeline.unet.eval()

    # -------------------------------------------------------------------------
    # Run the denoising process with logprobs and save latents
    # -------------------------------------------------------------------------
    num_inference_steps = int(NUM_INFERENCE_STEPS)
    print(f"Running {num_inference_steps} DDIM steps...")

    with torch.autocast(
        device_type=device.type,
        dtype=inference_dtype,
        enabled=(inference_dtype != torch.float32),
    ):
        images_final, _, all_latents, _ = pipeline_with_logprob(
            pipeline,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=config.sample.guidance_scale,
            eta=config.sample.eta,
            output_type="pt",
        )

    # all_latents: list of length T+1; each element (B, C, H, W)
    latents = torch.stack(all_latents, dim=0)[:, 0]  # (T+1, C, H, W)

    timesteps = pipeline.scheduler.timesteps.detach().cpu().numpy()  # (T,)
    assert latents.shape[0] == len(timesteps) + 1

    # -------------------------------------------------------------------------
    # Decode each latent to an image in [0, 1]
    # -------------------------------------------------------------------------
    decoded_images = []
    vae = pipeline.vae
    print("Decoding latents to images...")
    for t_idx in range(latents.shape[0]):
        latent = latents[t_idx].unsqueeze(0)  # (1, C, H, W)
        with torch.no_grad():
            img = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
        img = img.clamp(0, 1).cpu()  # (1, 3, H, W)
        decoded_images.append(img[0])

    decoded_images = torch.stack(decoded_images, dim=0)  # (T+1, 3, H, W)

    # -------------------------------------------------------------------------
    # Compute JPEG compressibility scores at each step
    # -------------------------------------------------------------------------
    print("Computing JPEG compressibility scores...")
    reward_fn = ddpo_pytorch.rewards.jpeg_compressibility()
    scores, _ = reward_fn(
        decoded_images,
        [prompt] * decoded_images.shape[0],
        [{}] * decoded_images.shape[0],
    )
    # scores: np.ndarray (T+1,)

    # -------------------------------------------------------------------------
    # Save images and scores for slider demo
    # -------------------------------------------------------------------------
    print(f"Saving images and scores to {output_dir}/ ...")
    images_uint8 = (
        decoded_images.numpy().transpose(0, 2, 3, 1) * 255
    ).astype(np.uint8)  # (T+1, H, W, C)

    for idx in range(images_uint8.shape[0]):
        img = Image.fromarray(images_uint8[idx])
        img.save(os.path.join(output_dir, f"step_{idx:03d}.png"))

    np.save(os.path.join(output_dir, "scores.npy"), scores)
    np.save(os.path.join(output_dir, "timesteps.npy"), timesteps)

    print("Done.")
    print(f"  Prompt saved to:      {os.path.join(output_dir, 'prompt.txt')}")
    print(f"  Scores saved to:      {os.path.join(output_dir, 'scores.npy')}")
    print(f"  Timesteps saved to:   {os.path.join(output_dir, 'timesteps.npy')}")
    print(f"  Images saved as:      {os.path.join(output_dir, 'step_XXX.png')}")


# ---------------------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SD / DDPO JPEG-compressibility trajectory."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to checkpoint directory (e.g. logs/.../checkpoint_XX). "
            "Use the literal 'None' to use plain Stable Diffusion."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_arg = args.checkpoint
    # Treat the literal string "None" as Python None
    checkpoint_dir = None if ckpt_arg == "None" else ckpt_arg
    generate_trajectory(checkpoint_dir)
