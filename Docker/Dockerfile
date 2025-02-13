# Use an NVIDIA CUDA runtime image (if GPU acceleration is needed)
# If GPU is not required, you might alternatively use a standard Python image.
FROM nvidia/cuda:12.4.127-runtime-ubuntu20.04

# FROM python:3-slim            # Comment out nvidia/cuda and uncomment this line if you don't have an nvida GPU

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies:
# - python3, pip, and python3-venv to create a virtual environment
# - build tools and libraries needed by some Python packages (e.g., OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        ca-certificates \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to ensure the latest installer is used
RUN pip3 install --upgrade pip

# Create a virtual environment in /opt/venv and activate it by default
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory for your app
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt inside the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Default command (adjust as needed for your application)
CMD ["python"]