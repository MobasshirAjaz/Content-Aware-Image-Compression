"""Segmentation pipeline using Grounded SAM2 (or fallback dummy masks).

This module:
- Loads an input image (from anywhere, typically `input_images/`).
- Accepts a list of subject names (from `LLM.py`).
- Runs Grounded SAM2 / Grounding DINO to get masks for each subject.
- Saves colored masks in `colored_masks/` and black/white masks in `blackwhite_masks/`.
- Creates background masks (union of subjects subtracted from full image) and saves
  them under `.../background/` subfolders.

This implementation is based on the Hugging Face Grounded-SAM2 demo and requires
the necessary libraries (transformers, supervision, sam2) and model checkpoints.
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List
import random

# --- Model & Pipeline Imports ---
try:
    import supervision as sv
    from supervision.draw.color import ColorPalette
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except ImportError as e:
    print(f"Error: Missing required libraries for Grounded SAM2. Please install them. Details: {e}")
    sys.exit(1)


# --- Directory and Path Configuration ---
BASE_DIR = os.path.dirname(__file__)
COLORED_DIR = os.path.join(BASE_DIR, "colored_masks")
BW_DIR = os.path.join(BASE_DIR, "blackwhite_masks")

# A custom color map for consistent colors in supervision
CUSTOM_COLOR_MAP = [
    '#ff3838', '#ff9d97', '#ff701f', '#ffb21d', '#cfd231', '#48f90a', '#92cc17',
    '#3ddb86', '#1a9334', '#00d4bb', '#2c99a8', '#00c2ff', '#344593', '#6473ff',
    '#0018ec', '#8438ff', '#520085', '#cb38ff', '#ff95c8', '#ff37c7'
]

# --- Global Model Loading & Configuration ---
# Load models only once when the module is imported for efficiency.

print("Loading Grounded SAM2 models...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model checkpoints (adjust paths if necessary)
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Check if model checkpoints exist
if not os.path.exists(SAM2_CHECKPOINT) or not os.path.exists(SAM2_MODEL_CONFIG):
    print(f"Error: SAM2 model files not found.")
    print(f"Please download '{os.path.basename(SAM2_CHECKPOINT)}' and place it in a 'checkpoints/' directory.")
    print(f"And ensure '{os.path.basename(SAM2_MODEL_CONFIG)}' is in a 'configs/sam2.1/' directory.")
    sys.exit(1)

# Use bfloat16 for performance if available
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

# Build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build Grounding DINO from Hugging Face
processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)
print("Models loaded successfully.")


# --- Directory and Helper Functions ---
def ensure_dirs():
    """Create all necessary output directories."""
    for d in (COLORED_DIR, BW_DIR):
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "background"), exist_ok=True)


def run_grounded_sam(image_path: str, subjects: List[str]):
    """
    Runs the full Grounded SAM2 pipeline on a single image to produce and save masks.
    """
    print(f"Running segmentation on '{os.path.basename(image_path)}' for subjects: {subjects}")
    # Format subjects into a text prompt for Grounding DINO
    text_prompt = " . ".join(subjects) + " ."
    text_prompt = text_prompt.lower()

    # --- 1. Load Image and Run Grounding DINO ---
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # Get bounding boxes for SAM2
    input_boxes = results[0]["boxes"].cpu().numpy()
    if len(input_boxes) == 0:
        print("Warning: Grounding DINO found no objects for the given subjects.")
        return

    # --- 2. Run SAM2 with Box Prompts ---
    sam2_predictor.set_image(np.array(image))
    masks, _, _ = sam2_predictor.predict(
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1) # (N, 1, H, W) -> (N, H, W)

    # --- 3. Process and Save Masks using Supervision ---
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=np.array(list(range(len(results[0]["labels"])))) # Assign a unique ID to each detection
    )
    class_names = results[0]["labels"]
    base = os.path.splitext(os.path.basename(image_path))[0]

    # Annotator for saving individual colored masks
    mask_annotator_individual = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    img_cv2 = cv2.imread(image_path) # Load with OpenCV for annotation

    class_counts = {}
    collected_masks = []

    for i in range(len(detections)):
        single_detection = detections[i]
        class_name = class_names[single_detection.class_id[0]]
        current_count = class_counts.get(class_name, 0)
        class_counts[class_name] = current_count + 1

        name_suffix = f"{class_name}" if current_count == 0 else f"{class_name}_{current_count}"
        mask_filename = f"{base}__{name_suffix}.png"

        # Save B&W mask
        current_mask = single_detection.mask[0]
        binary_mask_img = Image.fromarray((current_mask * 255).astype("uint8"), mode="L")
        binary_mask_img.save(os.path.join(BW_DIR, mask_filename))
        collected_masks.append(current_mask.astype(bool))

        # Save individual colored mask
        segmented_image = mask_annotator_individual.annotate(scene=img_cv2.copy(), detections=single_detection)
        cv2.imwrite(os.path.join(COLORED_DIR, mask_filename), segmented_image)

    # --- 4. Create and Save Background Mask ---
    if collected_masks:
        union = np.logical_or.reduce(collected_masks)
        background_mask = ~union

        # Save B&W background mask
        bg_bw_path = os.path.join(BW_DIR, "background", f"{base}__background.png")
        Image.fromarray((background_mask * 255).astype("uint8"), mode="L").save(bg_bw_path)

        # Save colored background (original image where mask is true)
        bg_color_path = os.path.join(COLORED_DIR, "background", f"{base}__background.png")
        bg_visible = Image.new("RGBA", image.size, (0, 0, 0, 0))
        bg_visible.paste(image, mask=Image.fromarray((background_mask * 255).astype("uint8")))
        bg_visible.save(bg_color_path)

    print("Finished saving segmentation masks.")


def generate_masks(image_path: str, subjects: List[str], use_dummy: bool = False):
    """
    Main entry point for generating masks. Can use the real model or fall back to dummy data.
    """
    ensure_dirs()
    if use_dummy:
        print("Using dummy masks for development.")
        _generate_dummy_masks(image_path, subjects)
        return

    # Run the real Grounded SAM pipeline
    run_grounded_sam(image_path, subjects)


def _generate_dummy_masks(image_path: str, subjects: List[str]):
    """Generates placeholder masks by splitting the image into vertical slices."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    n = max(1, len(subjects))
    slice_w = w // n
    masks = {}
    for i, name in enumerate(subjects):
        mask = np.zeros((h, w), dtype=bool)
        x0 = i * slice_w
        x1 = w if i == n - 1 else (i + 1) * slice_w
        mask[:, x0:x1] = True
        masks[name] = mask
    
    # Save the generated dummy masks
    _save_dummy_mask_images(image_path, masks, img)

def _save_dummy_mask_images(image_path: str, masks: dict, img: Image.Image):
    """Saves the images for the dummy mask generator."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    img_rgba = img.convert("RGBA")
    w, h = img.size

    def _random_color(seed: str) -> tuple:
        random.seed(hash(seed) & 0xFFFFFFFF)
        return tuple(random.randint(50, 230) for _ in range(3))

    for name, mask in masks.items():
        # B&W mask
        Image.fromarray((mask * 255).astype("uint8"), mode="L").save(os.path.join(BW_DIR, f"{base}__{name}.png"))
        
        # Color mask
        color = _random_color(name)
        overlay = Image.new("RGBA", (w, h), color + (150,))
        color_img = Image.new("RGBA", (w, h))
        color_img.paste(overlay, mask=Image.fromarray((mask * 255).astype("uint8")))
        composed = Image.alpha_composite(img_rgba, color_img)
        composed.save(os.path.join(COLORED_DIR, f"{base}__{name}.png"))
    
    # Background
    if masks:
        union = np.logical_or.reduce(list(masks.values()))
        background_mask = ~union
        Image.fromarray((background_mask * 255).astype("uint8"), mode="L").save(os.path.join(BW_DIR, "background", f"{base}__background.png"))
        bg_visible = Image.new("RGBA", (w, h))
        bg_visible.paste(img, mask=Image.fromarray((background_mask * 255).astype("uint8")))
        bg_visible.save(os.path.join(COLORED_DIR, "background", f"{base}__background.png"))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run segmentation to produce color and B&W masks.")
    p.add_argument("image", help="Path to the input image.")
    p.add_argument("--subjects", nargs="+", required=True, help="List of subject names (e.g., 'person dog car').")
    p.add_argument("--dummy", action="store_true", help="Use dummy masks for testing instead of the model.")
    
    # --- FIX: Add the new argument here ---
    p.add_argument(
        "--grounding-model", 
        default="IDEA-Research/grounding-dino-tiny", 
        help="Hugging Face model name for Grounding DINO."
    )
    
    args = p.parse_args()

    # --- FIX: Use the argument to set the model name ---
    # Note: argparse converts --grounding-model to args.grounding_model
    GROUNDING_MODEL = args.grounding_model 

    if not os.path.exists(args.image):
        print(f"Error: Image not found at '{args.image}'")
        sys.exit(1)
        
    # Now, you must load the models *after* parsing the arguments
    print(f"Loading Grounding DINO model: {GROUNDING_MODEL}...")
    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)
    print("Models loaded successfully.")


    generate_masks(args.image, args.subjects, use_dummy=args.dummy)
    print("\nProcess complete. Masks saved to 'colored_masks/' and 'blackwhite_masks/' directories.")