#!/bin/bash
# run_training.sh - Convenience script for running training on Vast.ai
# Usage: ./run_training.sh [train|test|finetune|resume] [model] [checkpoint]

set -e

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

# Your WandB API key (get from https://wandb.ai/authorize)
# Option 1: Hardcode here (not recommended if sharing image)
# WANDB_API_KEY="YOUR_API_KEY_HERE"

# Option 2: Set via environment variable before running this script
# export WANDB_API_KEY="your-key-here"

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  MarioRL Training Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Xvfb is running
if ! pgrep -x "Xvfb" > /dev/null; then
    echo -e "${YELLOW}Starting Xvfb...${NC}"
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
    sleep 2
    export DISPLAY=:99
    echo -e "${GREEN}Xvfb started on display :99${NC}"
else
    echo -e "${GREEN}Xvfb already running${NC}"
    export DISPLAY=:99
fi

# Check WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: WANDB_API_KEY not set${NC}"
    echo "Training will run but won't log to WandB"
    echo "Set with: export WANDB_API_KEY='your-key-here'"
    echo ""
else
    echo -e "${GREEN}WandB API key configured${NC}"
fi

# Parse arguments
MODE=${1:-test}
MODEL=${2:-ImpalaLarge}
CHECKPOINT=${3:-}

echo ""
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Model: $MODEL"
echo "  Checkpoint: ${CHECKPOINT:-None}"
echo ""

# Run verification first
echo -e "${YELLOW}Running setup verification...${NC}"
cd /workspace
python test_setup.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Setup verification failed! Please fix issues before training.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Setup verified! Starting training...${NC}"
echo ""

# Build training command
CMD="python src/train.py --mode $MODE --model $MODEL"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

echo "Running: $CMD"
echo ""

# Run training
cd /workspace
exec $CMD
