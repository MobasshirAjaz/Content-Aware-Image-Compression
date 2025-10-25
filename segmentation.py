"""Segmentation pipeline using Grounded SAM2 (or fallback dummy masks).

This module:
- Loads an input image (from anywhere, typically `input_images/`).
- Accepts a list of subject names (from `LLM.py`).
- Runs Grounded SAM / Grounding DINO (if available) to get masks for each subject.
- Saves colored masks in `colored_masks/` and black/white masks in `blackwhite_masks/`.
- Creates background masks (union of subjects subtracted from full image) and saves
  them under `.../background/` subfolders.

If the heavy models are not installed in the environment, you can run with
`use_dummy=True` to generate simple placeholder masks (useful for local dev).
"""

import os
import sys
from typing import List, Dict
import numpy as np
from PIL import Image, ImageDraw
import random


BASE_DIR = os.path.dirname(__file__)
COLORED_DIR = os.path.join(BASE_DIR, "colored_masks")
BW_DIR = os.path.join(BASE_DIR, "blackwhite_masks")


def ensure_dirs():
    for d in (COLORED_DIR, BW_DIR):
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "background"), exist_ok=True)


def _random_color(seed: str) -> tuple:
    random.seed(hash(seed) & 0xFFFFFFFF)
    return tuple(random.randint(50, 230) for _ in range(3))


def _save_mask_images(image_path: str, masks: Dict[str, np.ndarray]):
    """Save masks for each subject.

    masks: dict of subject -> boolean numpy mask (H x W)
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    # Save each subject mask
    for name, mask in masks.items():
        # color mask (RGBA overlay on original)
        color = _random_color(name)
        color_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        overlay = Image.new("RGBA", (w, h), color + (150,))
        color_img.paste(overlay, mask=Image.fromarray((mask * 255).astype("uint8")))
        composed = Image.alpha_composite(img, color_img)
        color_out = os.path.join(COLORED_DIR, f"{base}__{name}.png")
        composed.save(color_out)

        # black/white mask
        bw = Image.fromarray((mask * 255).astype("uint8"), mode="L")
        bw_out = os.path.join(BW_DIR, f"{base}__{name}.png")
        bw.save(bw_out)

    # background masks: union subjects and subtract from full
    union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for m in masks.values():
        union |= m

    background_mask = ~union
    # save background colored (solid gray) and bw
    bg_color_img = Image.new("RGBA", (w, h), (120, 120, 120, 255))
    bg_out_color = os.path.join(COLORED_DIR, "background", f"{base}__background.png")
    # color background: show original where background_mask True, else transparent
    bg_visible = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bg_visible.paste(img, mask=Image.fromarray((background_mask * 255).astype("uint8")))
    bg_visible.save(bg_out_color)

    # blackwhite background
    bg_bw_out = os.path.join(BW_DIR, "background", f"{base}__background.png")
    Image.fromarray((background_mask * 255).astype("uint8"), mode="L").save(bg_bw_out)


def run_grounded_sam(image_path: str, subjects: List[str]):
    """Attempt to run Grounded SAM2 to produce masks for each subject.

    This function will attempt to import model code from common repos. If the
    environment does not have Grounded SAM / Grounding DINO installed, it will
    raise ImportError with instructions.
    """
    try:
        # Try to import segment anything and grounding libraries
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        # Grounding DINO imports vary across forks; keep generic attempt
        # from groundingdino.models import build_model
    except Exception as e:
        raise ImportError(
            "Grounded SAM / Segment Anything not available in this environment. "
            "Install the required packages and model weights, or run with use_dummy=True. "
            "See project README or https://github.com/facebookresearch/segment-anything and "
            "https://github.com/IDEA-Research/Grounded-SAM for setup instructions.\n"
            f"(original import error: {e})"
        )

    # TODO: model loading and inference implementation depending on your local repo
    raise NotImplementedError("Grounded SAM inference not implemented in this template. Please implement model loading and mask extraction using your local Grounded SAM / Grounding DINO setup.")


def generate_masks(image_path: str, subjects: List[str], use_dummy: bool = False) -> Dict[str, np.ndarray]:
    """Main entry: return dict subject->mask (np.bool array).

    If use_dummy is True, we create placeholder masks by splitting the image
    into vertical slices (useful for dev without heavy models).
    """
    ensure_dirs()
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if use_dummy:
        masks = {}
        n = max(1, len(subjects))
        slice_w = w // n
        for i, name in enumerate(subjects):
            mask = np.zeros((h, w), dtype=bool)
            x0 = i * slice_w
            x1 = w if i == n - 1 else (i + 1) * slice_w
            mask[:, x0:x1] = True
            masks[name] = mask
        _save_mask_images(image_path, masks)
        return masks

    # Otherwise attempt to run the real grounded SAM pipeline
    masks = run_grounded_sam(image_path, subjects)
    # run_grounded_sam should return dict subject->mask
    _save_mask_images(image_path, masks)
    return masks


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run segmentation to produce color and bw masks for subjects.")
    p.add_argument("image", help="Path to input image (or rely on input_images folder)")
    p.add_argument("--subjects", nargs="+", help="List of subject names (from LLM)")
    p.add_argument("--dummy", action="store_true", help="Use dummy masks for testing")
    args = p.parse_args()

    if not args.subjects:
        print("No subjects provided. Provide subject names (e.g. 'person dog car') or use LLM to get them.")
        sys.exit(1)

    masks = generate_masks(args.image, args.subjects, use_dummy=args.dummy)
    print("Generated masks for:", list(masks.keys()))
