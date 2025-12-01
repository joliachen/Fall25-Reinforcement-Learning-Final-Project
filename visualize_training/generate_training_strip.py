#!/usr/bin/env python3
"""
generate_training_strip.py

Combine frames generated during training into a single horizontal strip
with checkpoint labels and compressibility scores.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
INPUT_DIR = "training_trajectory"
OUTPUT_PATH = "doc/training_strip.png"

FONT_SIZE = 28
LABEL_HEIGHT = 60
PADDING = 20
BACKGROUND_COLOR = (255, 255, 255)

# Optional: set to a TTF for nicer labels
FONT_PATH = None
# ---------------------------------------------------------------------


def text_size(draw, text, font):
    """
    Safe text measurement that works across Pillow versions.
    Tries textbbox → multiline_textbbox → textsize.
    """
    # Try textbbox (Pillow ≥ 8.0)
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font, align="center")
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    # Try multiline_textbbox (newer versions)
    if hasattr(draw, "multiline_textbbox"):
        bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    # Last fallback: split lines manually with textsize
    lines = text.split("\n")
    widths, heights = zip(*(draw.textsize(line, font=font) for line in lines))
    return max(widths), sum(heights)


def load_frames_sorted(input_dir):
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )
    return [os.path.join(input_dir, f) for f in files]


def load_scores_and_ckpts(input_dir):
    scores = np.load(os.path.join(input_dir, "scores.npy"))
    ckpts = np.load(os.path.join(input_dir, "checkpoints.npy"))
    return scores, ckpts


def make_strip(input_dir, output_path):
    frame_paths = load_frames_sorted(input_dir)
    scores, ckpts = load_scores_and_ckpts(input_dir)

    assert len(frame_paths) == len(scores) == len(ckpts), \
        "Mismatch: frames / scores / checkpoints must match."

    frames = [Image.open(fp).convert("RGB") for fp in frame_paths]
    w, h = frames[0].size

    total_width = w * len(frames)
    total_height = h + LABEL_HEIGHT + PADDING

    canvas = Image.new("RGB", (total_width, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Load font or fallback
    if FONT_PATH:
        try:
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except Exception:
            print("Warning: Could not load font. Using default.")
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # Draw frames + labels
    for i, (frame, score, ckpt_id) in enumerate(zip(frames, scores, ckpts)):
        x = i * w
        canvas.paste(frame, (x, 0))

        ckpt_name = "baseline" if ckpt_id < 0 else f"ckpt {ckpt_id}"
        label = f"{ckpt_name}\nscore {score:.2f}"

        text_w, text_h = text_size(draw, label, font)
        text_x = x + (w - text_w) // 2
        text_y = h + (LABEL_HEIGHT - text_h) // 2

        draw.multiline_text((text_x, text_y), label, fill=(0, 0, 0),
                            font=font, align="center")

    canvas.save(output_path)
    print("Saved strip to:", output_path)


if __name__ == "__main__":
    make_strip(INPUT_DIR, OUTPUT_PATH)
