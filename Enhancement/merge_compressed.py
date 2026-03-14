import os
import sys
import argparse
from PIL import Image
from pathlib import Path

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, 'final_merged_output')

def main():
    print("--- Starting Merging Script ---")

    # Set up argument parsing to accept directories from gui.py
    parser = argparse.ArgumentParser(description="Merge enhanced background with original subjects.")
    parser.add_argument('--subjects_dir', type=str, required=True, help="Directory containing the subject PNGs")
    parser.add_argument('--bg_dir', type=str, required=True, help="Directory containing the enhanced background")
    parser.add_argument('--base_name', type=str, required=True, help="Base name of the image for the final filename")
    args = parser.parse_args()

    Path(FINAL_OUTPUT_DIR).mkdir(exist_ok=True)

    try:
        # Real-ESRGAN will output this exact name based on our GUI flags
        background_path = Path(args.bg_dir) / 'background_enhanced.png'
        subjects_dir = Path(args.subjects_dir)
        
        # Grab all PNG files in the base subjects directory (ignoring the 'background' subfolder)
        foreground_paths = [p for p in subjects_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png']

        if not background_path.is_file():
            print(f"Error: Enhanced background not found at '{background_path}'.", file=sys.stderr)
            sys.exit(1)

        if not foreground_paths:
            print(f"Warning: No foreground PNGs found in '{args.subjects_dir}'. Final image will just be the background.")

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
                    
                    # Dimensions should match due to -s 1, but we include a failsafe
                    if fg_image.size != background_img.size:
                        print(f"  Warning: Resizing '{fg_path.name}' to match background dimensions.")
                        fg_image = fg_image.resize(background_img.size, Image.Resampling.LANCZOS)

                    background_img = Image.alpha_composite(background_img, fg_image)

            # --- Final Save Logic ---
            print("\n  Saving as optimized PNG...")
            final_save_name_png = f"{args.base_name}_enhanced_merged.png"
            final_save_path_png = Path(FINAL_OUTPUT_DIR) / final_save_name_png
            background_img.save(final_save_path_png, 'PNG', optimize=True, compress_level=9)

            print("  Saving as optimized JPG...")
            final_rgb_image = Image.new("RGB", background_img.size, (255, 255, 255))
            final_rgb_image.paste(background_img, mask=background_img.split()[3])

            final_save_name_jpg = f"{args.base_name}_enhanced_merged.jpg"
            final_save_path_jpg = Path(FINAL_OUTPUT_DIR) / final_save_name_jpg
            final_rgb_image.save(final_save_path_jpg, 'JPEG', quality=95, optimize=True)

            print("\n--- Merge Finished Successfully ---")
            print(f"Final PNG image saved to: {final_save_path_png}")
            print(f"Final JPG image saved to: {final_save_path_jpg}")
            sys.exit(0)

    except Exception as e:
        print(f"An unexpected error occurred during merging: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()