#!/bin/bash

# This script checks for a non-root user and creates one if it doesn't exist.
# It is designed for initial setup of a fresh Ubuntu/WSL installation.

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if the script is being run as root.
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root. Please use 'sudo'."
    exit 1
fi

# Get the number of users with a UID greater than 999 (standard for non-system users).
# It's a robust way to count non-system users.
user_count=$(getent passwd | awk -F: '$3 >= 1000 && $1 != "nobody" { print $1 }' | wc -l)

# Check if a non-root, non-system user exists.
if [ "$user_count" -gt 0 ]; then
    echo "A non-root user already exists. No new user will be created."
    echo "Current users:"
    getent passwd | awk -F: '$3 >= 1000 && $1 != "nobody" { print $1 }'
    exit 0
fi

echo "No non-root user found. A new user will be created."
echo "----------------------------------------------------"

# Prompt for the new username
read -p "Enter the new username: " NEW_USER

# Use the adduser command, which will handle the password prompt securely.
echo "Creating user '$NEW_USER'. You will be prompted to enter a password."
echo "----------------------------------------------------"
adduser --gecos "" --shell /bin/bash "$NEW_USER"

# Add the new user to the 'sudo' group to grant admin privileges.
echo "Adding '$NEW_USER' to the 'sudo' group for administrative access..."
usermod -aG sudo "$NEW_USER"

# Set the new user as the default user for the WSL distribution.
echo "Setting '$NEW_USER' as the default user for this WSL instance..."
echo -e "[user]\ndefault=$NEW_USER" | sudo tee /etc/wsl.conf > /dev/null

echo "----------------------------------------------------"
echo "Setup complete. User '$NEW_USER' has been created."
echo "Please close this terminal and run 'wsl --shutdown' from PowerShell"
echo "to apply the changes. The next time you open WSL, you will be logged in as '$NEW_USER'."