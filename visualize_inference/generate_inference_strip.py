#!/usr/bin/env python3
"""
generate_sd_vs_ddpo_three_row_strip.py

Create a 3-row trajectory strip:
  - Row 1: baseline Stable Diffusion (SD)
  - Row 2: DDPO-tuned SD (params set A)
  - Row 3: DDPO-tuned SD (params set B, e.g. trajectory_ddpo_failed)

Each column is a chosen timestep; images are:
  trajectory_sd/step_XXX.png
  trajectory_ddpo/step_XXX.png
  trajectory_ddpo_failed/step_XXX.png

Labels under each image show:
  "t = <step>   score <value>"

Each row also has a title above it.
"""

import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
SD_DIR = "trajectory_sd"
DDPO_A_DIR = "trajectory_ddpo"          # params set A
DDPO_B_DIR = "trajectory_ddpo_failed"   # params set B (failed)

OUTPUT_PATH = "doc/sd_vs_ddpoAB_strip.png"

# How many timesteps (columns) to show (will be spread across the trajectory)
NUM_TIMESTAMPS = 10

# Font sizes (increase these to make text larger)
LABEL_FONT_SIZE = 36   # for "t=..., score=..." under images
TITLE_FONT_SIZE = 38   # for row titles

# Layout
LABEL_HEIGHT = 48      # vertical space reserved for label under each image
TITLE_HEIGHT = 48      # vertical space reserved for the row title
PADDING_TOP = 30
PADDING_BOTTOM = 30
ROW_GAP = 20           # vertical gap between rows
BACKGROUND_COLOR = (255, 255, 255)

FONT_PATH = "times.ttf"

# Row titles
ROW_TITLES = [
    "",
    "",
    "",
]
# ---------------------------------------------------------------------
# SCORES (from your HTML; index = step)
# ---------------------------------------------------------------------
SD_SCORES = np.array([
    -232.786, -233.221, -232.285, -231.828, -232.059, -231.882, -233.878,
    -233.792, -236.26, -234.848, -236.262, -235.815, -233.405, -232.052,
    -232.064, -230.476, -228.553, -225.372, -223.301, -223.085, -220.833,
    -217.448, -215.226, -210.753, -209.441, -204.467, -204.307, -202.296,
    -196.657, -192.842, -187.468, -182.996, -178.545, -172.802, -166.766,
    -161.688, -155.573, -148.075, -140.966, -134.559, -128.321, -120.655,
    -111.958, -102.837, -92.47, -79.549, -68.529, -58.562, -48.288,
    -39.102, -38.822
])

DDPO_A_SCORES = np.array([
    -230.99, -231.122, -230.593, -226.397, -223.578, -218.42, -215.582,
    -207.236, -202.597, -196.979, -194.727, -186.664, -183.291, -178.289,
    -170.544, -165.401, -156.979, -152.397, -142.629, -138.551, -134.805,
    -128.568, -123.117, -117.888, -111.354, -103.226, -95.839, -88.927,
    -82.321, -75.064, -70.05, -63.27, -57.696, -52.161, -47.143, -42.127,
    -37.876, -33.434, -29.54, -25.57, -22.529, -19.936, -17.465, -15.294,
    -13.912, -12.897, -11.589, -10.34, -9.988, -9.623, -9.461
])

DDPO_B_SCORES = np.array([
    -234.984, -233.661, -228.066, -224.236, -222.564, -222.19, -222.009, 
    -221.808, -221.675, -221.378, -221.319, -221.248, -220.977, -220.832, 
    -220.842, -220.569, -220.44, -220.178, -219.983, -219.834, -219.809, 
    -219.504, -219.393, -219.313, -219.102, -219.058, -218.932, -218.806, 
    -218.74, -218.739, -218.598, -218.419, -218.518, -218.374, -218.37, -218.158, 
    -218.057, -218.061, -217.944, -217.892, -217.852, -217.814, -217.613, 
    -217.682, -217.553, -217.538, -217.448, -217.468, -217.483, -217.439, -217.395])


def text_size(draw, text, font):
    """
    Safe text measurement that works across Pillow versions.
    Tries textbbox → multiline_textbbox → textsize.
    """
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font, align="center")
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    if hasattr(draw, "multiline_textbbox"):
        bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    lines = text.split("\n")
    widths, heights = zip(*(draw.textsize(line, font=font) for line in lines))
    return max(widths), sum(heights)


def available_steps_from_dir(directory):
    """
    Parse filenames like 'step_000.png' → integer step index.
    Returns sorted list of steps.
    """
    steps = []
    for f in os.listdir(directory):
        m = re.match(r"step_(\d+)\.png$", f)
        if m:
            steps.append(int(m.group(1)))
    steps = sorted(set(steps))
    return steps


def choose_timesteps(sd_steps, ddpo_a_steps, ddpo_b_steps, num_timestamps):
    """
    Choose a set of timesteps common to all three trajectories, spread evenly.
    """
    common = sorted(
        set(sd_steps).intersection(ddpo_a_steps).intersection(ddpo_b_steps)
    )
    if not common:
        raise ValueError(
            "No common timesteps found between SD, DDPO_A, and DDPO_B directories."
        )

    num_timestamps = min(num_timestamps, len(common))
    idxs = np.linspace(0, len(common) - 1, num=num_timestamps, dtype=int)
    idxs = sorted(set(idxs.tolist()))  # ensure uniqueness
    return [common[i] for i in idxs]


def load_triplet_images(timestep):
    """
    Load SD, DDPO_A, DDPO_B images for a given timestep index.
    """
    step_str = f"{timestep:03d}"

    sd_path = os.path.join(SD_DIR, f"step_{step_str}.png")
    ddpo_a_path = os.path.join(DDPO_A_DIR, f"step_{step_str}.png")
    ddpo_b_path = os.path.join(DDPO_B_DIR, f"step_{step_str}.png")

    for path in [sd_path, ddpo_a_path, ddpo_b_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing image: {path}")

    sd_img = Image.open(sd_path).convert("RGB")
    ddpo_a_img = Image.open(ddpo_a_path).convert("RGB")
    ddpo_b_img = Image.open(ddpo_b_path).convert("RGB")

    return sd_img, ddpo_a_img, ddpo_b_img


def make_three_row_strip(output_path=OUTPUT_PATH):
    # Discover available steps
    sd_steps = available_steps_from_dir(SD_DIR)
    ddpo_a_steps = available_steps_from_dir(DDPO_A_DIR)
    ddpo_b_steps = available_steps_from_dir(DDPO_B_DIR)

    timesteps = choose_timesteps(sd_steps, ddpo_a_steps, ddpo_b_steps, NUM_TIMESTAMPS)

    # Sanity-check against score arrays
    max_step = min(len(SD_SCORES), len(DDPO_A_SCORES), len(DDPO_B_SCORES)) - 1
    for t in timesteps:
        if t > max_step:
            raise ValueError(
                f"Chosen timestep {t} exceeds available score length {max_step}."
            )

    # Load one triplet to get dimensions
    first_sd, first_ddpo_a, first_ddpo_b = load_triplet_images(timesteps[0])
    w, h = first_sd.size
    assert first_ddpo_a.size == (w, h), "SD and DDPO_A images must be the same size."
    assert first_ddpo_b.size == (w, h), "SD and DDPO_B images must be the same size."

    num_cols = len(timesteps)
    total_width = w * num_cols

    # 3 rows, each: title + image + label
    num_rows = 3
    total_height = (
        PADDING_TOP
        + num_rows * (TITLE_HEIGHT + h + LABEL_HEIGHT)
        + (num_rows - 1) * ROW_GAP
        + PADDING_BOTTOM
    )

    canvas = Image.new("RGB", (total_width, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Fonts
    if FONT_PATH:
        try:
            label_font = ImageFont.truetype(FONT_PATH, LABEL_FONT_SIZE)
            title_font = ImageFont.truetype(FONT_PATH, TITLE_FONT_SIZE)
        except Exception:
            print("Warning: Could not load font. Using default.")
            label_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
    else:
        label_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Per-row y positions
    y = PADDING_TOP
    row_configs = [
        ("baseline", SD_SCORES),
        ("ddpo_a", DDPO_A_SCORES),
        ("ddpo_b", DDPO_B_SCORES),
    ]

    for row_idx, (row_name, score_array) in enumerate(row_configs):
        row_title = ROW_TITLES[row_idx] if row_idx < len(ROW_TITLES) else row_name

        # Title position (centered horizontally over all columns)
        title_text_w, title_text_h = text_size(draw, row_title, title_font)
        title_x = (total_width - title_text_w) // 2
        title_y = y + (TITLE_HEIGHT - title_text_h) // 2
        draw.text((title_x, title_y), row_title, fill=(0, 0, 0), font=title_font)

        img_y = y + TITLE_HEIGHT
        label_y = img_y + h

        for col, t in enumerate(timesteps):
            x = col * w

            # Get appropriate row image
            if row_name == "baseline":
                sd_img, _, _ = load_triplet_images(t)
                img = sd_img
            elif row_name == "ddpo_a":
                _, ddpo_a_img, _ = load_triplet_images(t)
                img = ddpo_a_img
            else:  # "ddpo_b"
                _, _, ddpo_b_img = load_triplet_images(t)
                img = ddpo_b_img

            canvas.paste(img, (x, img_y))

            score = score_array[t]
            label = f"t = {t}   score {score:.2f}"

            text_w, text_h = text_size(draw, label, label_font)
            text_x = x + (w - text_w) // 2
            text_y = label_y + (LABEL_HEIGHT - text_h) // 2
            draw.text((text_x, text_y), label, fill=(0, 0, 0), font=label_font)

        # Move y down for next row
        y = label_y + LABEL_HEIGHT + ROW_GAP

    canvas.save(output_path)
    print("Saved SD vs DDPO 3-row strip to:", output_path)
    print("Timesteps used:", timesteps)


if __name__ == "__main__":
    make_three_row_strip()
