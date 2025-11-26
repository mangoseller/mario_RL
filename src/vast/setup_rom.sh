#!/bin/bash
# setup_rom.sh
# Run this after uploading your SuperMarioWorld-Snes folder to /workspace

set -e

# Check if the custom game folder exists
if [ ! -d "/workspace/SuperMarioWorld-Snes" ]; then
    echo "ERROR: /workspace/SuperMarioWorld-Snes not found!"
    echo "Please upload your custom game folder first."
    echo ""
    echo "It should contain:"
    echo "  - rom.sfc (your ROM file)"
    echo "  - data.json (RAM state extraction)"
    echo "  - scenario.json"
    echo "  - metadata.json"
    exit 1
fi

# Check for required files
for file in rom.sfc data.json; do
    if [ ! -f "/workspace/SuperMarioWorld-Snes/$file" ]; then
        echo "ERROR: Missing required file: $file"
        exit 1
    fi
done

# Find retro data path
RETRO_DATA=$(python -c "import retro; print(retro.data.path())")
TARGET_DIR="${RETRO_DATA}/stable/SuperMarioWorld-Snes"

echo "Retro data path: $RETRO_DATA"
echo "Installing custom game data to: $TARGET_DIR"

# Backup existing if present
if [ -d "$TARGET_DIR" ]; then
    echo "Backing up existing game data..."
    mv "$TARGET_DIR" "${TARGET_DIR}.backup"
fi

# Copy custom game data
cp -r /workspace/SuperMarioWorld-Snes "$TARGET_DIR"

echo "Custom game data installed successfully!"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import retro
env = retro.make('SuperMarioWorld-Snes', state='YoshiIsland2')
obs = env.reset()
print('Environment created successfully!')
print(f'Observation shape: {obs[0].shape}')
env.close()
"

echo ""
echo "Setup complete! You can now run training."
