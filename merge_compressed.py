# merge_compressed.py (Final Version)

import os
import sys
from PIL import Image
from pathlib import Path

# --- Configuration ---
COMPRESSED_DIR = 'compressed_output'
FINAL_OUTPUT_DIR = 'final_merged_output'
# The background is now a JPG, as created by the modified compression script
BACKGROUND_FILENAME = 'background.jpg' 

def find_latest_image_folder(directory: str) -> Path | None:
    """Finds the most recently modified subfolder in the given directory."""
    try:
        # Get all subdirectories
        subfolders = [d for d in Path(directory).iterdir() if d.is_dir()]
        if not subfolders:
            return None
        # Return the path of the most recently modified folder
        latest_folder = max(subfolders, key=lambda f: f.stat().st_mtime)
        return latest_folder
    except FileNotFoundError:
        return None

def main():
    """
    Finds a low-quality JPG background and high-quality PNG foregrounds,
    composites them, and saves the final result as a JPG.
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
        # Define the path for the low-quality JPG background
        background_path = image_folder / BACKGROUND_FILENAME
        
        # Find all high-quality PNG files to use as foregrounds
        foreground_paths = list(image_folder.glob('*.png'))

        if not background_path.is_file():
            print(f"Error: Background file '{BACKGROUND_FILENAME}' not found in '{image_folder}'.", file=sys.stderr)
            sys.exit(1)

        if not foreground_paths:
            print(f"Warning: No foreground PNGs found. The final image will just be the background.")

        # --- Compositing Logic ---
        print(f"  Loading base background: '{background_path.name}'")
        with Image.open(background_path) as background_img:
            # CRITICAL STEP: Convert the RGB background JPG to RGBA mode in memory.
            # This is required for alpha_composite to work with the transparent PNGs.
            if background_img.mode != 'RGBA':
                background_img = background_img.convert('RGBA')

            # Layer each high-quality PNG foreground on top of the background
            for fg_path in foreground_paths:
                print(f"  Compositing foreground: '{fg_path.name}'")
                with Image.open(fg_path) as fg_image:
                    # Ensure foreground is RGBA (it should be, but this is safe)
                    if fg_image.mode != 'RGBA':
                        fg_image = fg_image.convert('RGBA')
                    
                    # Perform the composite operation
                    background_img = Image.alpha_composite(background_img, fg_image)

            # --- Final Save Logic ---
            # Create a solid white background to flatten the final image onto
            final_rgb_image = Image.new("RGB", background_img.size, (255, 255, 255))
            
            # Paste the final composited RGBA image onto the white background, using
            # its own alpha channel as the mask to handle transparency.
            final_rgb_image.paste(background_img, mask=background_img.split()[3])

            # Save the final merged image as a JPG
            final_save_name = f"{image_name}_merged.jpg"
            final_save_path = Path(FINAL_OUTPUT_DIR) / final_save_name

            # You can use a higher quality here (like 75-85) because the background
            # is already heavily compressed, ensuring a small final file size.
            final_rgb_image.save(final_save_path, 'JPEG', quality=85)

            print("\n--- Merge Finished Successfully ---")
            print(f"Final JPG image saved to: {final_save_path}")
            sys.exit(0)

    except Exception as e:
        print(f"An unexpected error occurred during merging: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()