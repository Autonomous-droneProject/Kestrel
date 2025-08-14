#!/bin/bash

# Define the configuration to be added
WSL_CONF_CONTENT="[automount]
options = metadata,umask=22,fmask=111
"

# Flag to track if a restart is required
RESTART_REQUIRED=0

# --- Function to check if we are in a WSL environment ---
is_wsl() {
    grep -q "Microsoft" /proc/version
}

# --- Main script execution ---

echo "--- WSL Git Permissions Configurator ---"

# Check if the script is running inside WSL
if ! is_wsl; then
    echo "ERROR: This script must be run from within a WSL environment."
    echo "Please open a WSL terminal (e.g., Ubuntu) and run this script from there."
    exit 2 # Use a different exit code for this specific error
fi

# Check for root permissions
if [ "$(id -u)" -ne 0 ]; then
    echo "This script requires root permissions to modify /etc/wsl.conf."
    echo "Please run it with 'sudo' or as the root user."
    echo "Example: sudo ./configure_permissions.sh"
    exit 3 # Use a different exit code for this specific error
fi

# Check if the /etc/wsl.conf file exists
if [ -f "/etc/wsl.conf" ]; then
    echo "The /etc/wsl.conf file already exists."
    # Check if the required configuration is already present
    if ! grep -q "metadata,umask=22,fmask=111" /etc/wsl.conf; then
        echo "Updating automount options."
        # Use sed to replace or append the content
        sed -i '/\[automount\]/d' /etc/wsl.conf # Remove old [automount] section if it exists
        echo "$WSL_CONF_CONTENT" | tee -a /etc/wsl.conf > /dev/null
        RESTART_REQUIRED=1
    else
        echo "The required configuration is already in /etc/wsl.conf. No changes made."
        # Do not set RESTART_REQUIRED, as no changes were made
    fi
else
    echo "Creating a new /etc/wsl.conf file and adding the configuration."
    echo "$WSL_CONF_CONTENT" | sudo tee /etc/wsl.conf > /dev/null
    echo "File created successfully."
    RESTART_REQUIRED=1
fi

echo ""

if [ "$RESTART_REQUIRED" -eq 1 ]; then
    echo "Configuration changes were applied."
    echo "For the changes to take effect, you must restart WSL."
    echo "--------------------------------------------------------"
    echo "Please run the following command in Windows PowerShell or Command Prompt:"
    echo "wsl --shutdown"
    echo "--------------------------------------------------------"
    echo "Exiting with code 1 to indicate that a restart is pending."
else
    echo "No changes were needed. WSL restart is not required."
    echo "Exiting with code 0."
fi

# Exit with the appropriate code
exit $RESTART_REQUIRED