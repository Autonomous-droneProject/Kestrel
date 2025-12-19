#!/usr/bin/env bash
set -euo pipefail

# Simplified, robust rosdep install script
# Implements the requested sequence:
# 1) install software-properties-common and enable 'universe'
# 2) install curl
# 3) download and install ros2-apt-source .deb from latest GitHub release
# 4) ensure only 'ros2.sources' remains under /etc/apt/sources.list.d/
# 5) run rosdep install --from-paths src --ignore-src -r -y

WORKSPACE_ROOT="$(pwd)"
SRC_DIR="$WORKSPACE_ROOT/src"

echo "[rosdep_install] workspace root: $WORKSPACE_ROOT"

command_exists() { command -v "$1" >/dev/null 2>&1; }

require_sudo() {
  if ! command_exists sudo; then
    echo "[rosdep_install] ERROR: sudo is required to run this script." >&2
    exit 1
  fi
}

backup_file() {
  local f="$1"
  if [ -e "$f" ]; then
    sudo cp -a "$f" "${f}.bak-rosdep" || true
    echo "[rosdep_install] Backed up $f -> ${f}.bak-rosdep"
  fi
}

echo "[rosdep_install] Ensuring apt helpers and universe repository are available..."
require_sudo
sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common || true
sudo add-apt-repository -y universe || true

echo "[rosdep_install] Updating apt and installing curl..."
sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates gnupg lsb-release dpkg || true

# determine latest ros-apt-source release tag
echo "[rosdep_install] Determining latest ros-apt-source release from GitHub..."
ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}' || true)
if [ -z "${ROS_APT_SOURCE_VERSION:-}" ]; then
  echo "[rosdep_install] ERROR: could not determine ros-apt-source release tag from GitHub." >&2
  echo "[rosdep_install] Aborting. You can add the ROS apt source manually or retry later." >&2
  exit 1
fi

# compute UBUNTU codename
CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")
if [ -z "${CODENAME:-}" ]; then
  # fallback to lsb_release
  if command_exists lsb_release; then
    CODENAME=$(lsb_release -cs)
  else
    echo "[rosdep_install] ERROR: cannot determine Ubuntu codename." >&2
    exit 1
  fi
fi

DEBNAME="ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${CODENAME}_all.deb"
TMP_DEB="/tmp/ros2-apt-source.deb"
DOWNLOAD_URL="https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/${DEBNAME}"

echo "[rosdep_install] Downloading ros-apt-source package: $DOWNLOAD_URL"
if ! curl -L -o "$TMP_DEB" "$DOWNLOAD_URL" --fail -s; then
  echo "[rosdep_install] ERROR: failed to download $DOWNLOAD_URL" >&2
  exit 1
fi

echo "[rosdep_install] Installing $TMP_DEB"
if ! sudo dpkg -i "$TMP_DEB"; then
  echo "[rosdep_install] dpkg reported problems, attempting to fix with apt-get -f install..."
  sudo env DEBIAN_FRONTEND=noninteractive apt-get -y -f install
  sudo dpkg -i "$TMP_DEB" || {
    echo "[rosdep_install] ERROR: dpkg installation failed after fixing dependencies." >&2
    rm -f "$TMP_DEB" || true
    exit 1
  }
fi
rm -f "$TMP_DEB"

echo "[rosdep_install] Ensuring only 'ros2.sources' remains under /etc/apt/sources.list.d/"
require_sudo
shopt -s nullglob
ROSDIR=/etc/apt/sources.list.d
for f in "$ROSDIR"/ros2*; do
  [ -e "$f" ] || continue
  bn=$(basename "$f")
  if [ "$bn" != "ros2.sources" ]; then
    backup_file "$f"
    echo "[rosdep_install] Removing $f"
    sudo rm -f "$f" || true
  else
    echo "[rosdep_install] Keeping $f"
  fi
done
shopt -u nullglob

echo "[rosdep_install] Refreshing apt lists..."
sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq

# Ensure rosdep is installed
if ! command_exists rosdep; then
  echo "[rosdep_install] rosdep not found. Installing python3-rosdep via apt..."
  sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-rosdep || true
  if ! command_exists rosdep; then
    echo "[rosdep_install] Attempting pip3 install for rosdep..."
    if command_exists pip3; then
      pip3 install --user rosdep
      export PATH="$HOME/.local/bin:$PATH"
    else
      echo "[rosdep_install] ERROR: pip3 not found; please install rosdep manually." >&2
      exit 1
    fi
  fi
fi

# Initialize rosdep if needed
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  echo "[rosdep_install] Initializing rosdep database (sudo may be required)..."
  sudo rosdep init || true
fi

echo "[rosdep_install] Updating rosdep database..."
rosdep update || echo "[rosdep_install] rosdep update failed or already up-to-date"

if [ -d "$SRC_DIR" ]; then
  echo "[rosdep_install] Running: rosdep install --from-paths $SRC_DIR --ignore-src -r -y"
  rosdep install --from-paths "$SRC_DIR" --ignore-src -r -y
else
  echo "[rosdep_install] No src directory at $SRC_DIR; skipping rosdep install." >&2
fi

echo "[rosdep_install] Done."
