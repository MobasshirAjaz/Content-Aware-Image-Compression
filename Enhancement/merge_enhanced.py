# merge_enhanced.py
# Enhancement equivalent of Compression/merge_compressed.py
#
# Composites the enhanced background (PNG) with the original foreground
# subject PNGs (untouched) into a single final output image.

import os
import sys
from pathlib import Path

# --- Dependency imports with error handling ---
try:
    from PIL import Image
except ImportError:
    print(
        "Error: Pillow is not installed.\n"
        "Install it with: pip install pillow"
    )
    sys.exit(1)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENHANCED_DIR = os.path.join(BASE_DIR, 'enhanced_output')
FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, 'final_enhanced_output')
BACKGROUND_FILENAME = 'background.png'  # Enhanced background is a PNG (not JPG)


def find_latest_image_folder(directory: str):
    """Finds the most recently modified subfolder in the given directory."""
    try:
        subfolders = [d for d in Path(directory).iterdir() if d.is_dir()]
        if not subfolders:
            return None
        latest_folder = max(subfolders, key=lambda f: f.stat().st_mtime)
        return latest_folder
    except FileNotFoundError:
        return None


def run():
    """
    Finds the enhanced background PNG and foreground PNGs, composites them,
    and saves the final result as both PNG and JPG.

    Raises exceptions on failure (no sys.exit) so it can be called from gui.py.
    """
    print("--- Starting Enhanced Merge Script ---")

    Path(FINAL_OUTPUT_DIR).mkdir(exist_ok=True)

    image_folder = find_latest_image_folder(ENHANCED_DIR)

    if not image_folder:
        raise FileNotFoundError(
            f"No processed image folders found in '{ENHANCED_DIR}'. "
            f"Run enhance_background.py first."
        )

    image_name = image_folder.name
    print(f"Processing latest folder: '{image_name}'")

    background_path = image_folder / BACKGROUND_FILENAME
    foreground_paths = list(image_folder.glob('*.png'))

    # Remove the background from the foreground list if it appears there
    foreground_paths = [
        p for p in foreground_paths
        if p.name.lower() != BACKGROUND_FILENAME.lower()
    ]

    if not background_path.is_file():
        raise FileNotFoundError(
            f"Background file '{BACKGROUND_FILENAME}' not found in '{image_folder}'."
        )

    if not foreground_paths:
        print(
            "Warning: No foreground PNGs found. "
            "The final image will just be the enhanced background."
        )

    # --- Compositing Logic ---
    print(f"  Loading enhanced background: '{background_path.name}'")
    with Image.open(background_path) as background_img:
        if background_img.mode != 'RGBA':
            background_img = background_img.convert('RGBA')

        for fg_path in foreground_paths:
            print(f"  Compositing foreground: '{fg_path.name}'")
            with Image.open(fg_path) as fg_image:
                if fg_image.mode != 'RGBA':
                    fg_image = fg_image.convert('RGBA')

                # Ensure foreground matches background size
                if fg_image.size != background_img.size:
                    fg_image = fg_image.resize(
                        background_img.size, resample=Image.LANCZOS
                    )

                background_img = Image.alpha_composite(background_img, fg_image)

        # --- Final Save Logic ---

        # 1. Save as lossless PNG
        print("\n  Saving as optimized PNG...")
        final_save_name_png = f"{image_name}_enhanced.png"
        final_save_path_png = Path(FINAL_OUTPUT_DIR) / final_save_name_png
        background_img.save(
            final_save_path_png,
            'PNG',
            optimize=True,
            compress_level=9
        )

        # 2. Save as JPG (flattened)
        print("  Saving as optimized JPG...")
        final_rgb_image = Image.new("RGB", background_img.size, (255, 255, 255))
        final_rgb_image.paste(background_img, mask=background_img.split()[3])

        final_save_name_jpg = f"{image_name}_enhanced.jpg"
        final_save_path_jpg = Path(FINAL_OUTPUT_DIR) / final_save_name_jpg

        final_rgb_image.save(
            final_save_path_jpg,
            'JPEG',
            quality=95,      # High quality — we're enhancing, not compressing
            optimize=True
        )

        # --- Final Output Message ---
        print("\n--- Enhanced Merge Finished Successfully ---")
        print(f"Final PNG image saved to: {final_save_path_png}")
        print(f"Final JPG image saved to: {final_save_path_jpg}")


def main():
    """CLI entry-point wrapper around run()."""
    try:
        run()
    except Exception as e:
        print(
            f"An unexpected error occurred during merging: {e}",
            file=sys.stderr
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

