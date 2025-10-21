import json
from pycocotools import mask as maskUtils
import numpy as np
import cv2
import os

with open('outputs/grounded_sam2_hf_demo/grounded_sam2_hf_model_demo_results.json', 'r') as f:
    data = json.load(f)

image = cv2.imread(data['image_path'])
if image is None:
    raise FileNotFoundError(f"Could not find {data['path']}")

output_dir = "original_color_masks"
os.makedirs(output_dir, exist_ok=True)

height, width = image.shape[:2]
combined_mask = np.zeros((height, width), dtype=np.uint8)

class_counter = {}

for ann in data['annotations']:
    class_name = ann['class_name']
    seg = ann['segmentation']
    rle = {
        'size': seg['size'],
        'counts': seg['counts'].encode('utf-8')
    }

    # Decode RLE to binary mask (H×W)
    mask = maskUtils.decode(rle)
    combined_mask = np.maximum(combined_mask, mask)

    # Apply mask to original image (keep original colors where mask==1)
    colored_mask = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

    # Handle duplicate names
    class_counter[class_name] = class_counter.get(class_name, 0) + 1
    index = class_counter[class_name]

    # Save output image
    mask_path = os.path.join(output_dir, f"{class_name}_{index}.png")
    cv2.imwrite(mask_path, colored_mask)
    print(f"Saved: {mask_path}")

inverse_mask = cv2.bitwise_not(combined_mask * 255)
background_only = cv2.bitwise_and(image, image, mask=inverse_mask)
cv2.imwrite(os.path.join(output_dir, "background_only.png"), background_only)