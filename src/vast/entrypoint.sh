#!/bin/bash
set -e

# Trap signals for clean shutdown
cleanup() {
    echo "Shutting down Xvfb..."
    if [ -n "$XVFB_PID" ]; then
        kill $XVFB_PID 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start Xvfb (virtual framebuffer) for headless rendering
echo "Starting Xvfb on display :99..."
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to start
sleep 2

# Verify Xvfb is running
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi

echo "Xvfb started successfully (PID: $XVFB_PID)"
export DISPLAY=:99

# Verify display is working
if command -v xdpyinfo &> /dev/null; then
    if xdpyinfo -display :99 &> /dev/null; then
        echo "Display :99 verified working"
    else
        echo "WARNING: Display :99 may not be fully functional"
    fi
fi

# Print environment info
echo ""
echo "=== Environment Info ==="
echo "Python: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Working directory: $(pwd)"
echo "DISPLAY: $DISPLAY"
if [ -n "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY: Set (${WANDB_API_KEY:0:8}...)"
else
    echo "WANDB_API_KEY: NOT SET"
fi
echo "========================"
echo ""

# Execute the command passed to the container
exec "$@"
