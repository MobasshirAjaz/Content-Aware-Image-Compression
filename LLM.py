"""LLM helper: identify important subjects in an image.

This module reads image files from an `input_images/` directory (by default),
calls an LLM (ollama) to identify important objects, and writes a JSON file
with the ranked list. It returns a list of object names suitable for
consumption by the segmentation step.

Notes:
- Expects Ollama to be installed and available via the `ollama` Python client.
- Saves subject JSON to `outputs/` by default so downstream code can read it.
"""

import os
import base64
import re
import json
from typing import List, Optional

try:
    import ollama
except Exception:
    ollama = None


BASE_INPUT_DIR = os.path.join(os.path.dirname(__file__), "input_images")
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(BASE_INPUT_DIR, exist_ok=True)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


def _latest_image_in_input() -> Optional[str]:
    files = [f for f in os.listdir(BASE_INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(os.path.join(BASE_INPUT_DIR, p)), reverse=True)
    return os.path.join(BASE_INPUT_DIR, files[0])


def identify_objects(image_path: str, model: str = "gemma3", min_confidence: float = 0.0) -> Optional[List[dict]]:
    """Call Ollama (if available) to identify important objects.

    Returns the parsed JSON (list of {"object": name, "confidence": 0..1}).
    On failure returns None.
    """
    if ollama is None:
        raise RuntimeError("`ollama` package not available. Install Ollama client or run on a machine with Ollama.")

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = (
        "Identify the important objects in this image. do not include background images like sky.\n"
        "Return only a ranked JSON list of object names with confidence scores between 0 and 1.\n"
        "Example output: [{\"object\": \"person\", \"confidence\": 0.98}, ...]"
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
    )

    raw_output = response["message"]["content"].strip()
    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if not match:
        print("LLM response did not contain a JSON list. Raw output:\n", raw_output)
        return None

    json_str = match.group(0)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM response:", e)
        print("Raw output was:\n", raw_output)
        return None

    # Optionally filter by confidence
    if min_confidence > 0.0:
        parsed = [p for p in parsed if p.get("confidence", 0.0) >= min_confidence]

    return parsed


def save_subjects_json(image_path: str, subjects: List[dict]) -> str:
    """Save subjects list to outputs/<image_basename>_subjects.json and return path."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(BASE_OUTPUT_DIR, f"{base}_subjects.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subjects, f, indent=2, ensure_ascii=False)
    return out_path


def run_on_latest(min_confidence: float = 0.75, model: str = "gemma3") -> Optional[List[str]]:
    """Convenience: find latest image in `input_images`, ask LLM, save JSON, and return list of subject names.

    Returns list of subject strings (filtered by min_confidence) or None on error.
    """
    img = _latest_image_in_input()
    if img is None:
        print(f"No images found in {BASE_INPUT_DIR}. Place images there and re-run.")
        return None

    parsed = identify_objects(img, model=model, min_confidence=min_confidence)
    if parsed is None:
        return None

    save_subjects_json(img, parsed)
    names = [p["object"] for p in parsed if p.get("confidence", 0.0) >= min_confidence]
    return names


if __name__ == "__main__":
    # Quick CLI: process latest image in input_images and print subjects
    subjects = run_on_latest(min_confidence=0.75)
    if subjects is None:
        print("No subjects found or LLM unavailable.")
    else:
        print("Detected subjects:", subjects)