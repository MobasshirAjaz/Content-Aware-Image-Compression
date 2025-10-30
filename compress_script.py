# compress_script.py (Modified for Drastic Difference)

import os
import sys
from PIL import Image
from pathlib import Path

# --- Configuration ---
INPUT_DIR = 'transparent_images'
OUTPUT_DIR = 'compressed_output'

BACKGROUND_SUBFOLDER = 'background'
FOREGROUND_COMPRESSION = 3 # High-quality lossless for PNG foregrounds

# NEW: Quality setting for the intermediate background JPG
BACKGROUND_JPG_QUALITY = 5 # Lower this for more drastic results (e.g., 25)

def main():
    """
    Saves foregrounds as high-quality PNGs and the background as a low-quality JPG
    to achieve a drastic difference in the final merged image.
    """
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    print("\n--- Starting Drastic Compression Script ---")
    print(f"Reading from: '{INPUT_DIR}', Writing to: '{OUTPUT_DIR}'")
    
    try:
        for root, dirs, files in os.walk(INPUT_DIR):
            for filename in files:
                if not filename.lower().endswith(('.png', '.webp', '.tiff')):
                    continue

                input_path = Path(root) / filename
                relative_path = input_path.relative_to(INPUT_DIR)
                
                # --- LOGIC CHANGE IS HERE ---
                if BACKGROUND_SUBFOLDER in relative_path.parts:
                    # --- PROCESS BACKGROUND AS LOW-QUALITY JPG ---
                    output_subfolder = relative_path.parent.parent
                    final_output_dir = Path(OUTPUT_DIR) / output_subfolder
                    final_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # The output name is now a JPG
                    output_path = final_output_dir / 'background.jpg'
                    
                    with Image.open(input_path) as img:
                        # Create a white background to flatten the transparent image onto
                        bg_flatten = Image.new("RGB", img.size, (255, 255, 255))
                        # Paste the transparent background using its own alpha mask
                        bg_flatten.paste(img, mask=img.split()[3])
                        # Save as a low-quality JPG to discard data
                        bg_flatten.save(output_path, 'JPEG', quality=BACKGROUND_JPG_QUALITY)
                        print(f"  Processed BACKGROUND '{relative_path}' -> '{output_path.relative_to(OUTPUT_DIR)}' (JPG Quality {BACKGROUND_JPG_QUALITY})")

                else:
                    # --- PROCESS FOREGROUNDS AS HIGH-QUALITY PNG ---
                    output_subfolder = relative_path.parent
                    final_output_dir = Path(OUTPUT_DIR) / output_subfolder
                    final_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = final_output_dir / filename
                    
                    with Image.open(input_path) as img:
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        
                        img.save(
                            output_path,
                            format='PNG',
                            optimize=True,
                            compress_level=FOREGROUND_COMPRESSION
                        )
                        print(f"  Processed FOREGROUND '{relative_path}' -> '{output_path.relative_to(OUTPUT_DIR)}' (PNG Level {FOREGROUND_COMPRESSION})")

    except Exception as e:
        print(f"An error occurred during compression: {e}", file=sys.stderr)
        sys.exit(1)

    print("--- Compression Script Finished Successfully ---")
    sys.exit(0)

if __name__ == "__main__":
    main()