"""Create transparent images per subject using black/white masks.

Given an input image and a folder of black/white masks (one per subject and a
background mask under a `background/` subfolder), this script will create PNGs
with transparency where the white parts of the masks are opaque and the rest
transparent. Subject images go into `transparent_images/`, background goes into
`transparent_images/background/`.
"""

import os
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(__file__)
INPUT_IMAGES = os.path.join(BASE_DIR, "input_images")
BW_DIR = os.path.join(BASE_DIR, "blackwhite_masks")
OUT_DIR = os.path.join(BASE_DIR, "transparent_images")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "background"), exist_ok=True)


def _load_bw_mask(path: str) -> Image.Image:
    m = Image.open(path).convert("L")
    return m


def make_transparent(image_path: str, mask_path: str, out_path: str):
    img = Image.open(image_path).convert("RGBA")
    mask = _load_bw_mask(mask_path)
    if mask.size != img.size:
        mask = mask.resize(img.size, resample=Image.NEAREST)

    # alpha channel comes from mask (255 where white)
    alpha = mask
    r, g, b, _ = img.split()
    rgba = Image.merge("RGBA", (r, g, b, alpha))
    rgba.save(out_path)


def process_all_for_image(image_path: str):
    ensure_dirs()
    base = os.path.splitext(os.path.basename(image_path))[0]

    # find masks matching this base inside BW_DIR
    masks = []
    for f in os.listdir(BW_DIR):
        if f.lower().endswith(".png") and f.startswith(base + "__"):
            masks.append(os.path.join(BW_DIR, f))

    # subject masks
    for mpath in masks:
        name = os.path.splitext(os.path.basename(mpath))[0].split("__", 1)[1]
        outfn = os.path.join(OUT_DIR, f"{base}__{name}.png")
        make_transparent(image_path, mpath, outfn)

    # background mask in background folder
    bgdir = os.path.join(BW_DIR, "background")
    if os.path.isdir(bgdir):
        for f in os.listdir(bgdir):
            if f.lower().endswith(".png") and f.startswith(base + "__"):
                mpath = os.path.join(bgdir, f)
                outfn = os.path.join(OUT_DIR, "background", f)
                make_transparent(image_path, mpath, outfn)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Create transparent images from bw masks")
    p.add_argument("image", help="Path to input image (or put it in input_images and pass it)")
    args = p.parse_args()

    if not os.path.exists(args.image):
        print("Image not found:", args.image)
        raise SystemExit(1)

    process_all_for_image(args.image)
    print("Transparent images written to:", OUT_DIR)
