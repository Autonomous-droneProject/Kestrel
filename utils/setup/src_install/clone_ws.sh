#!/bin/bash

# This script clones a specific Git repository into a Kestrel directory
# within the user's home directory.

# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Define the repository URL and the target directory
# Replace the URL below with the actual repository you want to clone.
REPO_URL="https://github.com/Autonomous-droneProject/Kestrel.git"
TARGET_DIR="$HOME/Kestrel"

echo "--------------------------------------------------------"
echo "  Cloning predetermined repository to ~/Kestrel"
echo "--------------------------------------------------------"
echo "Repository URL: $REPO_URL"
echo "Target Directory: $TARGET_DIR"

# 2. Check if the target directory already exists
if [ -d "$TARGET_DIR" ]; then
    echo ""
    echo "Warning: The target directory '$TARGET_DIR' already exists."
    echo "Please remove or rename the directory before running this script."
    echo "Continuing"
fi

# 3. Perform the git clone operation
echo ""
echo "Cloning the repository into '$TARGET_DIR'..."
git clone "$REPO_URL" "$TARGET_DIR" --recurse-submodules
