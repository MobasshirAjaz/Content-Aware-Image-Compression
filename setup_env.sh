#!/bin/bash
set -e

CONDA_EXE="/c/Users/KIIT/miniconda3/Scripts/conda.exe"
ENV_NAME="image_compression"

echo "Installing PyTorch..."
$CONDA_EXE run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing standard dependencies..."
$CONDA_EXE run -n $ENV_NAME pip install pillow ollama opencv-python numpy supervision transformers

echo "Installing SAM2..."
$CONDA_EXE run -n $ENV_NAME pip install git+https://github.com/facebookresearch/sam2.git

echo "Installing GroundingDINO..."
# We just need transformers for huggingface auto processor which we already installed, but let's see if there are missing deps for IDEAS-Research/grounding-dino-tiny
# Wait, AutoModelForZeroShotObjectDetection is in transformers.

echo "Setup complete!"
