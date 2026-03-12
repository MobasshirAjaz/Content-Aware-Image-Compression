"""LLM wrapper for the Enhancement pipeline.

Uses importlib to load the Compression LLM module under a different internal
name, avoiding circular imports (since this file has the same name).
Redirects I/O directories to Enhancement-local folders.
"""

import os
import sys
import importlib.util

# ---------------------------------------------------------------------------
# 1. Load Compression/LLM.py via importlib
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPRESSION_DIR = os.path.join(os.path.dirname(BASE_DIR), "Compression")

_comp_llm_path = os.path.join(COMPRESSION_DIR, "LLM.py")

if not os.path.isfile(_comp_llm_path):
    print(
        f"Error: Could not find Compression LLM module at:\n"
        f"  {_comp_llm_path}\n"
        f"Make sure the Compression folder exists.",
        file=sys.stderr
    )
    sys.exit(1)

try:
    _spec = importlib.util.spec_from_file_location(
        "_compression_llm",  # unique internal name
        _comp_llm_path
    )
    _comp_llm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_comp_llm)
except Exception as e:
    print(
        f"Error: Failed to load LLM from Compression.\n"
        f"Make sure its dependencies are installed.\n"
        f"Details: {e}",
        file=sys.stderr
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Override directories to Enhancement-local paths
# ---------------------------------------------------------------------------
BASE_INPUT_DIR = os.path.join(BASE_DIR, "input_images")
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(BASE_INPUT_DIR, exist_ok=True)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

_comp_llm.BASE_INPUT_DIR = BASE_INPUT_DIR
_comp_llm.BASE_OUTPUT_DIR = BASE_OUTPUT_DIR

# ---------------------------------------------------------------------------
# 3. Re-export public API
# ---------------------------------------------------------------------------
identify_objects = _comp_llm.identify_objects
save_subjects_json = _comp_llm.save_subjects_json
run_on_latest = _comp_llm.run_on_latest


# ---------------------------------------------------------------------------
# 4. CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    subjects = run_on_latest(min_confidence=0.75)
    if subjects is None:
        print("No subjects found or LLM unavailable.")
    else:
        print("Detected subjects:", subjects)
