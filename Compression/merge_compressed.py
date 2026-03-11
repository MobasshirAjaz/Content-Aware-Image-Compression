# merge_compressed.py (Modified to save both PNG and JPG)

import os
import sys
from PIL import Image
from pathlib import Path

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPRESSED_DIR = os.path.join(BASE_DIR, 'compressed_output')
FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, 'final_merged_output')
BACKGROUND_FILENAME = 'background.jpg' 

def find_latest_image_folder(directory: str) -> Path | None:
    """Finds the most recently modified subfolder in the given directory."""
    try:
        subfolders = [d for d in Path(directory).iterdir() if d.is_dir()]
        if not subfolders:
            return None
        latest_folder = max(subfolders, key=lambda f: f.stat().st_mtime)
        return latest_folder
    except FileNotFoundError:
        return None

def main():
    """
    Finds a low-quality JPG background and high-quality PNG foregrounds,
    composites them, and saves the final result as both a PNG and a JPG.
    """
    print("--- Starting Merging Script ---")

    Path(FINAL_OUTPUT_DIR).mkdir(exist_ok=True)

    image_folder = find_latest_image_folder(COMPRESSED_DIR)

    if not image_folder:
        print(f"Error: No processed image folders found in '{COMPRESSED_DIR}'.", file=sys.stderr)
        sys.exit(1)

    image_name = image_folder.name
    print(f"Processing latest folder: '{image_name}'")

    try:
        background_path = image_folder / BACKGROUND_FILENAME
        foreground_paths = list(image_folder.glob('*.png'))

        if not background_path.is_file():
            print(f"Error: Background file '{BACKGROUND_FILENAME}' not found in '{image_folder}'.", file=sys.stderr)
            sys.exit(1)

        if not foreground_paths:
            print(f"Warning: No foreground PNGs found. The final image will just be the background.")

        # --- Compositing Logic ---
        print(f"  Loading base background: '{background_path.name}'")
        with Image.open(background_path) as background_img:
            if background_img.mode != 'RGBA':
                background_img = background_img.convert('RGBA')

            for fg_path in foreground_paths:
                print(f"  Compositing foreground: '{fg_path.name}'")
                with Image.open(fg_path) as fg_image:
                    if fg_image.mode != 'RGBA':
                        fg_image = fg_image.convert('RGBA')
                    
                    background_img = Image.alpha_composite(background_img, fg_image)

            # At this point, 'background_img' holds the final composited RGBA image.

            # --- Final Save Logic (Modified) ---

            # 1. Save as a lossless, transparent PNG
            # This preserves transparency and is ideal for further use where quality is key.
            print("\n  Saving as optimized PNG...")
            final_save_name_png = f"{image_name}_merged.png"
            final_save_path_png = Path(FINAL_OUTPUT_DIR) / final_save_name_png
            background_img.save(
                final_save_path_png,
                'PNG',
                optimize=True,
                compress_level=9  # 9 is max compression (slower), 6 is a good default.
            )

            # 2. Save as a lossy, flattened JPG
            # This creates the smallest file size for web use but loses transparency.
            print("  Saving as optimized JPG...")
            final_rgb_image = Image.new("RGB", background_img.size, (255, 255, 255))
            final_rgb_image.paste(background_img, mask=background_img.split()[3])

            final_save_name_jpg = f"{image_name}_merged.jpg"
            final_save_path_jpg = Path(FINAL_OUTPUT_DIR) / final_save_name_jpg
            
            final_rgb_image.save(
                final_save_path_jpg,
                'JPEG',
                quality=60,      # A good balance of quality and size for web.
                optimize=True    # Makes an extra pass to reduce file size.
            )

            # --- Final Output Message ---
            print("\n--- Merge Finished Successfully ---")
            print(f"Final PNG image saved to: {final_save_path_png}")
            print(f"Final JPG image saved to: {final_save_path_jpg}")
            sys.exit(0)

    except Exception as e:
        print(f"An unexpected error occurred during merging: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()