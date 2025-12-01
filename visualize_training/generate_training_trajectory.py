#!/usr/bin/env python3
"""
generate_training_trajectory.py

For a fixed prompt, sample one *final* denoised image from a sequence
of DDPO checkpoints and compute the JPEG-compressibility reward.

This produces a "training trajectory" like the llama teaser:
one image per checkpoint, showing how the learned policy evolves.

Outputs (example):
  training_trajectory_llama/
    frame_000_baseline.png
    frame_001_ckpt000.png
    frame_002_ckpt005.png
    ...
    scores.npy          # score per frame, same order as images
    checkpoints.npy     # checkpoint id per frame (or -1 for baseline)
"""

import os
import imp
import numpy as np
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel

import ddpo_pytorch.rewards

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

# Directory of a single DDPO run (the one with checkpoints/ inside)
RUN_DIR = "logs/2025.11.29_09.59.54"

CHECKPOINT_IDS = [0, 5, 10, 20, 40, 60, 80, 98]
PROMPT = "a golden retriever running in a field"
OUTPUT_DIR = "training_trajectory"

# Include a pure Stable Diffusion baseline as the first frame
USE_BASELINE_FIRST = True

# Fixed random seed so all checkpoints see the same noise
GENERATOR_SEED = 0


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def load_config():
    """Load the compressibility config from config/dgx.py."""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.dirname(this_dir)
    dgx_path = os.path.join(repo_root, "config", "dgx.py")

    if not os.path.exists(dgx_path):
        raise FileNotFoundError(f"Could not find dgx.py at {dgx_path}")

    dgx = imp.load_source("dgx", dgx_path)
    return dgx.compressibility()


def build_base_pipeline(config, device, dtype):
    """
    Build a StableDiffusionPipeline with DDIM scheduler and move everything
    (except possibly LoRA weights) to the desired device/dtype.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
    )

    # disable safety checker and switch to DDIM
    pipe.safety_checker = None
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # move to device
    pipe.vae.to(device, dtype=dtype)
    pipe.text_encoder.to(device, dtype=dtype)
    pipe.unet.to(device, dtype=dtype)

    return pipe


def load_checkpoint_into_pipeline(pipe, config, checkpoint_dir_or_none):
    """
    Load DDPO weights into the pipeline's UNet.

    - If checkpoint_dir_or_none is None:
        leave pipeline as pure SD (baseline).
    - If config.use_lora is True:
        load LoRA attention processors from checkpoint dir.
    - Else:
        load a full UNet from checkpoint dir.
    """
    if checkpoint_dir_or_none is None:
        print("Using baseline Stable Diffusion (no DDPO checkpoint).")
        return

    ckpt = checkpoint_dir_or_none

    if config.use_lora:
        print(f"Loading LoRA attention processors from {ckpt}")
        pipe.unet.load_attn_procs(ckpt)
    else:
        print(f"Loading full UNet from {ckpt}/unet")
        unet = UNet2DConditionModel.from_pretrained(ckpt, subfolder="unet")
        pipe.unet = unet.to(pipe.device, dtype=next(pipe.unet.parameters()).dtype)


def sample_image_and_score(pipe, prompt, num_steps, guidance_scale, eta, device):
    """
    Run a *full* DDIM denoising pass for a single image, returning:
        - image tensor in [0,1], shape (3, H, W)
        - JPEG-compressibility reward (scalar float)
    """
    reward_fn = ddpo_pytorch.rewards.jpeg_compressibility()
    generator = torch.Generator(device=device).manual_seed(GENERATOR_SEED)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if pipe.unet.dtype == torch.float16 else pipe.unet.dtype,
        enabled=(pipe.unet.dtype != torch.float32),
    ):
        out = pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=generator,
            output_type="pt",  # tensor in [0,1]
        )

    # diffusers returns either a dataclass or a tuple, handle both
    images = getattr(out, "images", out[0])  # (B,3,H,W)
    img = images[0].detach().cpu()          # (3,H,W), in [0,1]

    scores, _ = reward_fn(
        img.unsqueeze(0),        # (1,3,H,W)
        [prompt],
        [{}],
    )
    score = float(scores[0])
    return img, score


def save_image_tensor(img, path):
    """Save a (3,H,W) float tensor in [0,1] as PNG."""
    arr = (img.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # match train.py mixed-precision logic
    inference_dtype = torch.float32
    if getattr(config, "mixed_precision", None) == "fp16":
        inference_dtype = torch.float16
    elif getattr(config, "mixed_precision", None) == "bf16":
        inference_dtype = torch.bfloat16

    pipe = build_base_pipeline(config, device=device, dtype=inference_dtype)
    pipe.set_progress_bar_config(disable=True)

    scores = []
    ckpt_ids_record = []
    frame_idx = 0

    # 1) Optional baseline (pure SD, no DDPO checkpoints)
    if USE_BASELINE_FIRST:
        load_checkpoint_into_pipeline(pipe, config, checkpoint_dir_or_none=None)
        img, score = sample_image_and_score(
            pipe,
            PROMPT,
            num_steps=config.sample.num_steps,
            guidance_scale=config.sample.guidance_scale,
            eta=config.sample.eta,
            device=device,
        )
        out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:03d}_baseline.png")
        save_image_tensor(img, out_path)
        print(f"Saved baseline frame to {out_path} with score {score:.2f}")
        scores.append(score)
        ckpt_ids_record.append(-1)
        frame_idx += 1

    # 2) Sequence of checkpoints
    for ckpt_id in CHECKPOINT_IDS:
        ckpt_dir = os.path.join(RUN_DIR, "checkpoints", f"checkpoint_{ckpt_id}")
        if not os.path.isdir(ckpt_dir):
            print(f"WARNING: checkpoint dir not found: {ckpt_dir}, skipping.")
            continue

        load_checkpoint_into_pipeline(pipe, config, checkpoint_dir_or_none=ckpt_dir)

        img, score = sample_image_and_score(
            pipe,
            PROMPT,
            num_steps=config.sample.num_steps,
            guidance_scale=config.sample.guidance_scale,
            eta=config.sample.eta,
            device=device,
        )

        out_path = os.path.join(
            OUTPUT_DIR, f"frame_{frame_idx:03d}_ckpt{ckpt_id:03d}.png"
        )
        save_image_tensor(img, out_path)
        print(f"Saved frame for checkpoint {ckpt_id} to {out_path} with score {score:.2f}")

        scores.append(score)
        ckpt_ids_record.append(ckpt_id)
        frame_idx += 1

    # 3) Save scores + checkpoint ids
    scores = np.array(scores, dtype=np.float32)
    ckpt_ids_record = np.array(ckpt_ids_record, dtype=np.int32)
    np.save(os.path.join(OUTPUT_DIR, "scores.npy"), scores)
    np.save(os.path.join(OUTPUT_DIR, "checkpoints.npy"), ckpt_ids_record)
    print("Done.")
    print("  Scores saved to:", os.path.join(OUTPUT_DIR, "scores.npy"))
    print("  Checkpoint ids saved to:", os.path.join(OUTPUT_DIR, "checkpoints.npy"))


if __name__ == "__main__":
    main()
