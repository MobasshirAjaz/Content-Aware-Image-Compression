import ollama
import base64
import re
import json
import io

def identify_objects(image_path, model="gemma3"):

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = """
    Identify the important objects in this image.
    Return only a ranked JSON list of object names with confidence scores between 0 and 1.
    without any other texts
    """

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_b64]
        }]
    )

    raw_output = response["message"]["content"].strip()
    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError as e:
            print("JSON parsing failed:", e)
            return None
    else:
        print("No JSON found in response")
        return None

result = identify_objects("temp_after.jpg")
print(result)
