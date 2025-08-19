#!/usr/bin/env bash
set -euo pipefail

# install_gz_harmonic_noble.sh
# Non-interactive installer for Gazebo Harmonic on Ubuntu Noble (24.04) only.
# Follows official binary install steps: add OSRF GPG key, add repo, apt update, install gz-harmonic.
# Official docs: https://gazebosim.org/docs/harmonic/install_ubuntu/

REQUIRED_CODENAME="noble"
REPO_FILE="/etc/apt/sources.list.d/gazebo-stable.list"
KEYRING="/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg"

echo
echo "==== Gazebo Harmonic installer (noble/24.04) ===="
echo

# ensure apt-get present
if ! command -v apt-get >/dev/null 2>&1; then
  echo "Error: apt-get not found. This script requires Debian/Ubuntu's apt package manager."
  exit 1
fi

# check codename
DETECTED_CODENAME="$(lsb_release -cs || true)"
if [ "$DETECTED_CODENAME" != "$REQUIRED_CODENAME" ]; then
  echo "Error: This installer is explicitly targeted for Ubuntu 'noble' (24.04)."
  echo "Detected codename: '${DETECTED_CODENAME:-unknown}'. Aborting to avoid accidental changes."
  exit 2
fi

echo "Confirmed distro codename: $DETECTED_CODENAME (noble). Proceeding..."

# run as a safe non-interactive installer
export DEBIAN_FRONTEND=noninteractive

echo
echo "Updating package lists and installing prerequisites..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends curl lsb-release gnupg apt-transport-https ca-certificates

echo
echo "Creating keyrings directory (if missing)..."
sudo mkdir -p /usr/share/keyrings

echo
echo "Downloading OSRF GPG key to $KEYRING (will overwrite if exists)..."
sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o "$KEYRING"
if [ ! -s "$KEYRING" ]; then
  echo "Error: failed to download or write GPG key. Check network and permissions."
  exit 3
fi

echo
echo "Writing APT source list to $REPO_FILE (will overwrite existing file)..."
echo "deb [arch=$(dpkg --print-architecture) signed-by=$KEYRING] http://packages.osrfoundation.org/gazebo/ubuntu-stable $REQUIRED_CODENAME main" | \
  sudo tee "$REPO_FILE" > /dev/null

echo
echo "Updating APT cache..."
sudo apt-get update -y

sudo apt-get install -y gz-harmonic
sudo apt-get install -y ros-jazzy-gz-tools-vendor ros-jazzy-gz-sim-vendor
sudo apt-get install ros-jazzy-ros-gz
set +u
source /opt/ros/jazzy/setup.bash
set -u

echo
echo "Post-install verification..."
if command -v gz >/dev/null 2>&1; then
  echo "'gz' command located at: $(command -v gz)"
  echo "You can run 'gz sim' to launch Gazebo Harmonic."
  echo
  echo "Installation completed successfully."
  exit 0
else
  echo "Warning: 'gz' not found after install. Try: sudo apt-get update && sudo apt-get install -y gz-harmonic"
  exit 4
fi
