"""Transparency wrapper for the Enhancement pipeline.

Uses importlib to load the Compression make_transparent module under a
different internal name, avoiding circular imports (since this file has
the same name). Redirects I/O directories to Enhancement-local folders.
"""

import os
import sys
import importlib.util

# ---------------------------------------------------------------------------
# 1. Load Compression/make_transparent.py via importlib
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPRESSION_DIR = os.path.join(os.path.dirname(BASE_DIR), "Compression")

_comp_mt_path = os.path.join(COMPRESSION_DIR, "make_transparent.py")

if not os.path.isfile(_comp_mt_path):
    print(
        f"Error: Could not find Compression make_transparent module at:\n"
        f"  {_comp_mt_path}\n"
        f"Make sure the Compression folder exists.",
        file=sys.stderr
    )
    sys.exit(1)

try:
    _spec = importlib.util.spec_from_file_location(
        "_compression_make_transparent",  # unique internal name
        _comp_mt_path
    )
    _comp_mt = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_comp_mt)
except Exception as e:
    print(
        f"Error: Failed to load make_transparent from Compression.\n"
        f"Make sure its dependencies are installed.\n"
        f"Details: {e}",
        file=sys.stderr
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Override directories to Enhancement-local paths
# ---------------------------------------------------------------------------
INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "input_images")
MASKS_DIR = os.path.join(BASE_DIR, "blackwhite_masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "transparent_images")

_comp_mt.INPUT_IMAGES_DIR = INPUT_IMAGES_DIR
_comp_mt.MASKS_DIR = MASKS_DIR
_comp_mt.OUTPUT_DIR = OUTPUT_DIR

# ---------------------------------------------------------------------------
# 3. Re-export public API
# ---------------------------------------------------------------------------
make_transparent = _comp_mt.make_transparent
process_image = _comp_mt.process_image


# ---------------------------------------------------------------------------
# 4. CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="[Enhancement] Create transparent images from black & white masks."
    )
    parser.add_argument(
        "image",
        help=f"Path to a single input image. Should be located in '{INPUT_IMAGES_DIR}'."
    )
    args = parser.parse_args()

    full_image_path = os.path.join(INPUT_IMAGES_DIR, args.image)

    if not os.path.exists(full_image_path):
        print(f"Error: Image not found at '{full_image_path}'", file=sys.stderr)
        sys.exit(1)

    process_image(full_image_path)
    print("\nScript finished.")
