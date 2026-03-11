# make_transparent_compatible.py

"""Create transparent images per subject using black/white masks.

Given an input image and a folder of black/white masks (one per subject and a
background mask under a `background/` subfolder), this script will create a 
dedicated subfolder in `transparent_images/` and save the resulting transparent
PNGs there.
"""

import os
from PIL import Image
import sys

# --- Configuration ---
# Assuming a standard project structure where this script is in the root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "input_images")
MASKS_DIR = os.path.join(BASE_DIR, "blackwhite_masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "transparent_images")


def make_transparent(image_path: str, mask_path: str, out_path: str):
    """Applies a black/white mask to an image to create transparency."""
    try:
        img = Image.open(image_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L") # "L" is for luminance (grayscale)

        # Ensure mask and image are the same size
        if mask.size != img.size:
            mask = mask.resize(img.size, resample=Image.NEAREST)

        # Get the Red, Green, Blue channels from the original image
        r, g, b, _ = img.split()
        
        # The mask itself becomes the new alpha channel
        # White (255) in the mask becomes fully opaque
        # Black (0) in the mask becomes fully transparent
        rgba = Image.merge("RGBA", (r, g, b, mask))
        
        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        rgba.save(out_path, 'PNG')
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred in make_transparent: {e}", file=sys.stderr)


def process_image(image_path: str):
    """
    Finds all corresponding masks for a given image and creates the transparent
    PNGs in a structured output folder.
    """
    print(f"\n--- Processing image: {os.path.basename(image_path)} ---")
    
    # Get the base name of the image without extension (e.g., "image1")
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # This is the crucial change: create a dedicated output folder for this image job
    job_output_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(job_output_dir, exist_ok=True)
    
    # 1. Process Foreground Masks
    # Masks are expected to be named like "image1__subjectA.png"
    for mask_filename in os.listdir(MASKS_DIR):
        if mask_filename.lower().startswith(base_name + "__") and mask_filename.lower().endswith(".png"):
            
            # Extract the subject name (e.g., "subjectA")
            try:
                subject_name = os.path.splitext(mask_filename)[0].split("__", 1)[1]
            except IndexError:
                print(f"  Skipping malformed mask name: {mask_filename}")
                continue

            mask_path = os.path.join(MASKS_DIR, mask_filename)
            output_filename = f"{subject_name}.png"
            output_path = os.path.join(job_output_dir, output_filename)
            
            print(f"  Applying mask '{mask_filename}' -> '{output_filename}'")
            make_transparent(image_path, mask_path, output_path)

    # 2. Process Background Mask
    # The background mask is expected in a subfolder: "blackwhite_masks/background/"
    background_mask_dir = os.path.join(MASKS_DIR, "background")
    if os.path.isdir(background_mask_dir):
        for mask_filename in os.listdir(background_mask_dir):
            if mask_filename.lower().startswith(base_name + "__") and mask_filename.lower().endswith(".png"):
                
                # Create the corresponding background output folder
                output_bg_dir = os.path.join(job_output_dir, "background")
                os.makedirs(output_bg_dir, exist_ok=True)

                mask_path = os.path.join(background_mask_dir, mask_filename)
                # The output name can be simplified as we know it's the background
                output_filename = "background.png" 
                output_path = os.path.join(output_bg_dir, output_filename)
                
                print(f"  Applying background mask '{mask_filename}' -> 'background/background.png'")
                make_transparent(image_path, mask_path, output_path)
                break # Assume only one background mask per image

    print(f"--- Finished processing. Output in: '{job_output_dir}' ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create transparent images from black & white masks.")
    parser.add_argument("image", help=f"Path to a single input image. Should be located in '{INPUT_IMAGES_DIR}'.")
    args = parser.parse_args()

    # Build the full path to the image
    full_image_path = os.path.join(INPUT_IMAGES_DIR, args.image)

    if not os.path.exists(full_image_path):
        print(f"Error: Image not found at '{full_image_path}'", file=sys.stderr)
        sys.exit(1)

    process_image(full_image_path)
    print("\nScript finished.")