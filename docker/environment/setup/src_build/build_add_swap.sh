#!/bin/bash

# Define the size and location of the new swap file
# For example, to create a 8GB swap file:
SWAP_FILE_SIZE="16G"
SWAP_FILE="/swapfile_temp"

# Check if a swap file already exists and if so, exit
if [ -f "$SWAP_FILE" ]; then
    echo "Swap file already exists at $SWAP_FILE. Enabling swap file."
    sudo swapon "$SWAP_FILE"
    echo "Done. This swap file will be active until the next WSL shutdown."
    exit 0
fi

# Create a new, empty file of the specified size
echo "Creating swap file of size $SWAP_FILE_SIZE..."
sudo fallocate -l $SWAP_FILE_SIZE "$SWAP_FILE"

# Set the correct permissions (only root can read/write)
echo "Setting permissions..."
sudo chmod 600 "$SWAP_FILE"

# Mark the file as a swap space
echo "Marking the file as swap..."
sudo mkswap "$SWAP_FILE"

# Enable the new swap file
echo "Enabling the new swap file..."
sudo swapon "$SWAP_FILE"

# Verify the changes
echo "Swap file enabled. Checking current swap status:"
swapon --show

echo "Done. This swap file will be active until the next WSL shutdown."