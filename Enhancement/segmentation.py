"""Segmentation wrapper for the Enhancement pipeline.

Uses importlib to load the Compression segmentation module under a different
internal name, avoiding circular imports (since this file has the same name).
Redirects all output directories to Enhancement-local folders.
"""

import os
import sys
import importlib.util

# ---------------------------------------------------------------------------
# 1. Load Compression/segmentation.py via importlib (avoids circular import)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPRESSION_DIR = os.path.join(os.path.dirname(BASE_DIR), "Compression")

_comp_seg_path = os.path.join(COMPRESSION_DIR, "segmentation.py")

if not os.path.isfile(_comp_seg_path):
    print(
        f"Error: Could not find Compression segmentation module at:\n"
        f"  {_comp_seg_path}\n"
        f"Make sure the Compression folder exists.",
        file=sys.stderr
    )
    sys.exit(1)

# Add Compression dir to sys.path so that segmentation.py's own imports
# (supervision, sam2, transformers, etc.) can resolve correctly
if COMPRESSION_DIR not in sys.path:
    sys.path.insert(0, COMPRESSION_DIR)

try:
    _spec = importlib.util.spec_from_file_location(
        "_compression_segmentation",  # unique internal name — NOT "segmentation"
        _comp_seg_path
    )
    _comp_seg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_comp_seg)
except Exception as e:
    print(
        f"Error: Failed to load segmentation from Compression.\n"
        f"Make sure its dependencies are installed.\n"
        f"Details: {e}",
        file=sys.stderr
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Override output directories to point to Enhancement-local folders
# ---------------------------------------------------------------------------
COLORED_DIR = os.path.join(BASE_DIR, "colored_masks")
BW_DIR = os.path.join(BASE_DIR, "blackwhite_masks")

_comp_seg.COLORED_DIR = COLORED_DIR
_comp_seg.BW_DIR = BW_DIR

# ---------------------------------------------------------------------------
# 3. Re-export the public API (unchanged behaviour, new output location)
# ---------------------------------------------------------------------------
generate_masks = _comp_seg.generate_masks
load_models = _comp_seg.load_models
run_grounded_sam = _comp_seg.run_grounded_sam
ensure_dirs = _comp_seg.ensure_dirs


# ---------------------------------------------------------------------------
# 4. CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="[Enhancement] Run segmentation to produce color and B&W masks."
    )
    p.add_argument("image", help="Path to the input image.")
    p.add_argument(
        "--subjects", nargs="+", required=True,
        help="List of subject names (e.g., 'person dog car')."
    )
    p.add_argument(
        "--dummy", action="store_true",
        help="Use dummy masks for testing instead of the model."
    )
    p.add_argument(
        "--grounding-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face model name for Grounding DINO."
    )

    args = p.parse_args()
    _comp_seg.GROUNDING_MODEL = args.grounding_model

    if not os.path.exists(args.image):
        print(f"Error: Image not found at '{args.image}'")
        sys.exit(1)

    generate_masks(args.image, args.subjects, use_dummy=args.dummy)
    print(
        "\nProcess complete. Masks saved to "
        f"'{COLORED_DIR}' and '{BW_DIR}' directories."
    )
