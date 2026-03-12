# enhance_background.py
# Enhancement equivalent of Compression/compress_script.py
#
# Instead of compressing (reducing quality of) the background, this script
# ENHANCES (upscales/improves quality of) the background using Real-ESRGAN
# while leaving foreground subject images untouched.

import os
import sys
import shutil
from pathlib import Path

# --- Dependency imports with error handling ---
try:
    from PIL import Image
except ImportError:
    print(
        "Error: Pillow is not installed.\n"
        "Install it with: pip install pillow"
    )
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(
        f"Error: Missing required library.\n"
        f"Install with: pip install opencv-python numpy\n"
        f"Details: {e}"
    )
    sys.exit(1)

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
except ImportError as e:
    print(
        f"Error: 'basicsr' is not installed.\n"
        f"Install it with: pip install basicsr\n"
        f"Details: {e}"
    )
    sys.exit(1)

try:
    from realesrgan import RealESRGANer
except ImportError as e:
    print(
        f"Error: 'realesrgan' is not installed.\n"
        f"Install it with: pip install realesrgan\n"
        f"Details: {e}"
    )
    sys.exit(1)

try:
    import torch
except ImportError as e:
    print(
        f"Error: 'torch' (PyTorch) is not installed.\n"
        f"Install it following: https://pytorch.org/get-started/locally/\n"
        f"Details: {e}"
    )
    sys.exit(1)


# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'transparent_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'enhanced_output')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')

BACKGROUND_SUBFOLDER = 'background'

# Real-ESRGAN model configuration
MODEL_NAME = 'RealESRGAN_x4plus'
MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
NETSCALE = 4       # The native upscale factor of the model
OUTSCALE = 4       # Final output scale relative to input (we downscale back later)
TILE = 0           # 0 = no tiling. Set to e.g. 512 if you hit CUDA OOM.
TILE_PAD = 10
PRE_PAD = 0
USE_HALF = True    # Use fp16 for speed; set False on CPU or if you get NaN


def _get_upsampler():
    """Load the Real-ESRGAN upsampler, downloading weights if necessary."""
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=NETSCALE
    )

    model_path = os.path.join(WEIGHTS_DIR, f'{MODEL_NAME}.pth')
    if not os.path.isfile(model_path):
        print(f"Model weights not found locally. Downloading to '{WEIGHTS_DIR}'...")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        model_path = load_file_from_url(
            url=MODEL_URL,
            model_dir=WEIGHTS_DIR,
            progress=True,
            file_name=None
        )

    # Determine half precision: only on CUDA
    half = USE_HALF and torch.cuda.is_available()

    gpu_id = 0 if torch.cuda.is_available() else None

    upsampler = RealESRGANer(
        scale=NETSCALE,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=TILE,
        tile_pad=TILE_PAD,
        pre_pad=PRE_PAD,
        half=half,
        gpu_id=gpu_id
    )
    return upsampler


def enhance_image(upsampler, img_cv2, original_size=None):
    """
    Enhance a single image using Real-ESRGAN.

    Parameters
    ----------
    upsampler : RealESRGANer
        The pre-loaded upsampler instance.
    img_cv2 : np.ndarray
        Input image in BGR format (as read by cv2.imread).
    original_size : tuple[int, int] or None
        If provided (width, height), the enhanced image will be resized back
        to this resolution after upscaling.  This keeps it mergeable with
        the foreground subjects which remain at the original resolution.

    Returns
    -------
    np.ndarray
        Enhanced (and optionally resized) image in BGR format.
    """
    try:
        output, _ = upsampler.enhance(img_cv2, outscale=OUTSCALE)
    except RuntimeError as e:
        print(f"Error during enhancement: {e}")
        print("If you encounter CUDA out of memory, try setting TILE to e.g. 512.")
        raise

    # Downscale back to original resolution so it matches the foregrounds
    if original_size is not None:
        w, h = original_size
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)

    return output


def find_latest_job_folder(directory: str):
    """Find the most recently modified subfolder (= latest processed image job)."""
    try:
        subfolders = [d for d in Path(directory).iterdir() if d.is_dir()]
        if not subfolders:
            return None
        return max(subfolders, key=lambda f: f.stat().st_mtime)
    except FileNotFoundError:
        return None


def run():
    """
    Walk the transparent_images folder, enhance the background using Real-ESRGAN,
    and copy foregrounds unchanged into enhanced_output/.

    Raises exceptions on failure (no sys.exit) so it can be called from gui.py.
    """
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(
            f"Input directory '{INPUT_DIR}' not found. "
            f"Run segmentation and make_transparent first."
        )

    print("\n--- Starting Background Enhancement Script ---")
    print(f"Reading from : '{INPUT_DIR}'")
    print(f"Writing to   : '{OUTPUT_DIR}'")

    # Load the Real-ESRGAN model once
    print("\nLoading Real-ESRGAN model...")
    upsampler = _get_upsampler()
    print("Model loaded successfully.\n")

    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if not filename.lower().endswith(('.png', '.webp', '.tiff')):
                continue

            input_path = Path(root) / filename
            relative_path = input_path.relative_to(INPUT_DIR)

            if BACKGROUND_SUBFOLDER in relative_path.parts:
                # ----- ENHANCE THE BACKGROUND -----
                output_subfolder = relative_path.parent.parent
                final_output_dir = Path(OUTPUT_DIR) / output_subfolder
                final_output_dir.mkdir(parents=True, exist_ok=True)

                output_path = final_output_dir / 'background.png'

                with Image.open(input_path) as img_pil:
                    original_size = img_pil.size  # (width, height)

                    # Flatten RGBA → RGB (white background for transparent areas)
                    if img_pil.mode == 'RGBA':
                        bg_flatten = Image.new("RGB", img_pil.size, (255, 255, 255))
                        bg_flatten.paste(img_pil, mask=img_pil.split()[3])
                    else:
                        bg_flatten = img_pil.convert("RGB")

                    # Convert to cv2 array (BGR)
                    img_cv2 = cv2.cvtColor(np.array(bg_flatten), cv2.COLOR_RGB2BGR)

                # Enhance with Real-ESRGAN and resize back to original
                enhanced_cv2 = enhance_image(
                    upsampler, img_cv2, original_size=original_size
                )

                # Save enhanced background as high-quality PNG
                cv2.imwrite(str(output_path), enhanced_cv2)
                print(
                    f"  ENHANCED  background '{relative_path}' -> "
                    f"'{output_path.relative_to(OUTPUT_DIR)}'"
                )

            else:
                # ----- COPY FOREGROUNDS UNCHANGED -----
                output_subfolder = relative_path.parent
                final_output_dir = Path(OUTPUT_DIR) / output_subfolder
                final_output_dir.mkdir(parents=True, exist_ok=True)

                output_path = final_output_dir / filename
                shutil.copy2(str(input_path), str(output_path))
                print(
                    f"  COPIED    foreground '{relative_path}' -> "
                    f"'{output_path.relative_to(OUTPUT_DIR)}'"
                )

    print("--- Background Enhancement Script Finished Successfully ---")


def main():
    """CLI entry-point wrapper around run()."""
    try:
        run()
    except Exception as e:
        print(f"An error occurred during enhancement: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
