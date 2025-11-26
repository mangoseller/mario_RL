# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python build dependencies
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    # For stable-retro
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Headless rendering (Xvfb)
    xvfb \
    x11-utils \
    # Video encoding for RecordVideo
    ffmpeg \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY src/ ./src/

# Copy test and utility scripts
COPY test_setup.py ./
COPY run_training.sh ./
RUN chmod +x run_training.sh

# Copy your custom SuperMarioWorld-Snes data (ROM + data.json + scenario.json)
# This directory should contain: rom.sfc, data.json, scenario.json, metadata.json
COPY SuperMarioWorld-Snes/ /tmp/SuperMarioWorld-Snes/

# Find where retro installed its data and replace with your custom version
RUN RETRO_DATA=$(python -c "import retro; print(retro.data.path())") && \
    rm -rf "${RETRO_DATA}/stable/SuperMarioWorld-Snes" && \
    cp -r /tmp/SuperMarioWorld-Snes "${RETRO_DATA}/stable/SuperMarioWorld-Snes" && \
    rm -rf /tmp/SuperMarioWorld-Snes

# Create directories for outputs
RUN mkdir -p /workspace/evals /workspace/model_checkpoints

# Set up Xvfb wrapper script for headless rendering
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ============================================================================
# WANDB API KEY - Set this when running the container!
# Option 1: docker run -e WANDB_API_KEY="your-key-here" ...
# Option 2: Set in Vast.ai environment variables
# Option 3: Uncomment and hardcode below (not recommended for shared images)
# ============================================================================
# ENV WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"

# Environment variables
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
