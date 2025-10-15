# Use an official NVIDIA PyTorch image as a base.
# This image includes CUDA 12.1 and cuDNN 8, which are necessary for GPU access.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install the Python dependencies
# --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python-headless
