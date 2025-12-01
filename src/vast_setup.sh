#!/bin/bash
# vast_setup.sh - Complete setup script for Vast.ai instances
# Installs Python 3.10, dependencies, and configures the RL environment

set -e

echo "=========================================="
echo "  Vast.ai Environment Setup Script"
echo "=========================================="

# Deactivate any active venv
deactivate 2>/dev/null || true

# --- 1. Install Python 3.10 ---
echo ""
echo "[1/6] Installing Python 3.10..."
apt-get update
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

# Install pip for Python 3.10 (ignore existing system pip)
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 - --ignore-installed

echo "Python installed: $(python3.10 --version)"

# --- 2. Install System Dependencies ---
echo ""
echo "[2/6] Installing system dependencies..."
apt-get install -y \
    xvfb \
    ffmpeg \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    x11-utils \
    unzip

# --- 3. Install Python Packages ---
echo ""
echo "[3/6] Installing Python packages..."
python3.10 -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install packages separately to avoid dependency conflicts
python3.10 -m pip install --no-cache-dir gymnasium
python3.10 -m pip install --no-cache-dir stable-retro
python3.10 -m pip install --no-cache-dir torchrl
python3.10 -m pip install --no-cache-dir einops
python3.10 -m pip install --no-cache-dir wandb
python3.10 -m pip install --no-cache-dir tqdm moviepy imageio imageio-ffmpeg

# --- 4. Set Up Display (Xvfb) ---
echo ""
echo "[4/6] Starting Xvfb virtual display..."

# Kill any existing Xvfb on :99
pkill -f "Xvfb :99" 2>/dev/null || true
sleep 1

# Start Xvfb
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 2

# Verify Xvfb started
if kill -0 $XVFB_PID 2>/dev/null; then
    echo "Xvfb started successfully (PID: $XVFB_PID)"
else
    echo "WARNING: Xvfb may not have started correctly"
fi

export DISPLAY=:99

# --- 5. Configure Environment ---
echo ""
echo "[5/6] Configuring environment..."

# Add to bashrc for persistence
grep -q 'export DISPLAY=:99' ~/.bashrc || echo 'export DISPLAY=:99' >> ~/.bashrc
grep -q 'export PYTHONPATH=/workspace' ~/.bashrc || echo 'export PYTHONPATH=/workspace:$PYTHONPATH' >> ~/.bashrc
grep -q 'alias python=python3.10' ~/.bashrc || echo 'alias python=python3.10' >> ~/.bashrc
grep -q 'alias pip=' ~/.bashrc || echo 'alias pip="python3.10 -m pip"' >> ~/.bashrc

# Create workspace directories
mkdir -p /workspace/evals /workspace/model_checkpoints

# --- 6. Set Up ROM (if available) ---
echo ""
echo "[6/6] Checking for ROM setup..."

if [ -d "/workspace/SuperMarioWorld-Snes" ]; then
    echo "Found SuperMarioWorld-Snes folder, importing ROM..."
    
    # Check for required files
    if [ ! -f "/workspace/SuperMarioWorld-Snes/rom.sfc" ]; then
        echo "WARNING: rom.sfc not found in SuperMarioWorld-Snes folder"
    elif [ ! -f "/workspace/SuperMarioWorld-Snes/data.json" ]; then
        echo "WARNING: data.json not found in SuperMarioWorld-Snes folder"
    else
        # Import ROM using retro's import tool
        echo "Importing ROM with stable-retro..."
        python3.10 -m retro.import /workspace/SuperMarioWorld-Snes
        
        # Copy additional config files (data.json, scenario.json, metadata.json)
        RETRO_DATA=$(python3.10 -c "import retro; print(retro.data.path())")
        TARGET_DIR="${RETRO_DATA}/stable/SuperMarioWorld-Snes"
        
        if [ -d "$TARGET_DIR" ]; then
            echo "Copying config files to: $TARGET_DIR"
            cp -f /workspace/SuperMarioWorld-Snes/*.json "$TARGET_DIR/" 2>/dev/null || true
            echo "ROM and config files installed!"
        else
            echo "WARNING: ROM import may have failed - target directory not created"
        fi
        
        # Verify installation
        echo "Verifying environment..."
        python3.10 -c "
import retro
env = retro.make('SuperMarioWorld-Snes', state='YoshiIsland2')
obs = env.reset()
print(f'Environment created! Observation shape: {obs[0].shape}')
env.close()
" && echo "ROM verification passed!" || echo "WARNING: ROM verification failed"
    fi
else
    echo "No SuperMarioWorld-Snes folder found in /workspace"
    echo "Upload your ROM folder and run:"
    echo "  python3.10 -m retro.import /workspace/SuperMarioWorld-Snes"
fi

# --- Summary ---
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo "Python:    $(python3.10 --version)"
echo "CUDA:      $(python3.10 -c 'import torch; print(torch.cuda.is_available())')"
echo "Display:   $DISPLAY"
echo "Workspace: /workspace"
echo ""
echo "Directories created:"
echo "  - /workspace/evals"
echo "  - /workspace/model_checkpoints"
echo ""
echo "IMPORTANT: Use 'python3.10' to run scripts, or 'source ~/.bashrc' for aliases."
echo "=========================================="
