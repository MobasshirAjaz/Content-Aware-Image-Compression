import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--text-prompt", default="")
parser.add_argument("--img-path", default="temp_after.jpg")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default="outputs/grounded_sam2_hf_demo")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
TEXT_PROMPT = args.text_prompt
IMG_PATH = args.img_path
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
DUMP_JSON_RESULTS = not args.no_dump_json

### MODIFIED ###
# Define and create all output directories
OUTPUT_DIR = Path(args.output_dir)
ROOT_DIR = Path(__file__).resolve().parents[1]
COLORED_DIR = ROOT_DIR / "colored_masks"
BW_DIR = ROOT_DIR / "blackwhite_masks"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COLORED_DIR.mkdir(parents=True, exist_ok=True)
BW_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(COLORED_DIR / "background", exist_ok=True)
os.makedirs(BW_DIR / "background", exist_ok=True)
### END MODIFIED ###


# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

image = Image.open(img_path)
sam2_predictor.set_image(np.array(image.convert("RGB")))

inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

# get the box prompt for SAM 2
input_boxes = results[0]["boxes"].cpu().numpy()

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# Post-process the output of the model
if masks.ndim == 4:
    masks = masks.squeeze(1)

confidences = results[0]["scores"].cpu().numpy().tolist()
class_names = results[0]["labels"]
class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

# Load original image with OpenCV for annotations
img = cv2.imread(img_path)

# Create a Supervision Detections object
detections = sv.Detections(
    xyxy=input_boxes,
    mask=masks.astype(bool),
    class_id=class_ids
)


### MODIFIED: Save individual masks and colored segmentation in repo-root dirs ###
print("Saving individual segmentation results and masks...")
# Keep track of counts for each class name to create unique filenames
class_counts = {}
mask_annotator_individual = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))

# base name derived from image
base = os.path.splitext(os.path.basename(img_path))[0]

# collect masks to compute background later
collected_masks = []

for i in range(len(detections)):
    single_detection = detections[i]
    class_name = class_names[single_detection.class_id[0]]

    current_count = class_counts.get(class_name, 0)
    class_counts[class_name] = current_count + 1

    # prepare filename (append count if >0)
    name_suffix = f"{class_name}" if current_count == 0 else f"{class_name}_{current_count}"
    mask_filename = f"{base}__{name_suffix}.png"

    # --- 1. Save the black and white mask ---
    current_mask = single_detection.mask[0]
    binary_mask = (current_mask * 255).astype(np.uint8)
    bw_path = BW_DIR / mask_filename
    cv2.imwrite(str(bw_path), binary_mask)
    collected_masks.append(current_mask.astype(bool))

    # --- 2. Save the individual colored segmentation result ---
    segmented_image = mask_annotator_individual.annotate(scene=img.copy(), detections=single_detection)
    colored_path = COLORED_DIR / mask_filename
    cv2.imwrite(str(colored_path), segmented_image)

print("Finished saving individual files.")

# Create background mask by union of all subject masks and subtracting from full image
if len(collected_masks) > 0:
    union = np.zeros_like(collected_masks[0], dtype=bool)
    for m in collected_masks:
        union |= m

    background_mask = ~union
    # Save BW background
    bg_bw_path = BW_DIR / "background" / f"{base}__background.png"
    Image.fromarray((background_mask * 255).astype("uint8"), mode="L").save(bg_bw_path)

    # Save colored background: original where background True, else transparent
    pil_img = Image.open(img_path).convert("RGBA")
    w, h = pil_img.size
    bg_visible = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bg_visible.paste(pil_img, mask=Image.fromarray((background_mask * 255).astype("uint8")))
    bg_color_path = COLORED_DIR / "background" / f"{base}__background.png"
    bg_visible.save(bg_color_path)

### END MODIFIED ###


# Visualize and save the combined annotated image (original functionality)
print("Saving combined annotated image...")
box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

mask_annotator_combined = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame_with_mask = mask_annotator_combined.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame_with_mask)


# Dump the results in standard format and save as json files
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    mask_rles = [single_mask_to_rle(mask) for mask in masks]
    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    results = {
        "image_path": img_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": image.width,
        "img_height": image.height,
    }
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)