from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
import supervision as sv

# -------- CONFIGURATION --------
# Replace with the path to your image
IMAGE_PATH = "input.jpg"

# Define the subjects (prompts)
PROMPTS = {
    "person": "person",
    "cat": "cat",
    "bicycle": "bicycle"
}

# -------- LOAD MODEL --------
ontology = CaptionOntology(PROMPTS)
model = GroundedSAM2(ontology=ontology)

# -------- RUN PREDICTION --------
results = model.predict(IMAGE_PATH)

# -------- VISUALIZE / SAVE OUTPUT --------
image = sv.Image.open(IMAGE_PATH)
annotator = sv.MaskAnnotator()

# Apply the masks and labels on the image
detections = sv.Detections.from_inference(results)
labels = [f"{class_name}" for class_name in detections.class_id]
annotated = annotator.annotate(scene=image, detections=detections, labels=labels)

# Save the visualization
sv.Image.save(annotated, "output_masked.jpg")
print("✅ Masks generated and saved as output_masked.jpg")
