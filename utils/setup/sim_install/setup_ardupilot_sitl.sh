#!/usr/bin/env bash
# setup_ardupilot_sitl_with_path.sh

# IMPORTANT: Run as a normal user (NOT root). The script will use sudo where required.

set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

export DEBIAN_FRONTEND=noninteractive
ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
GIT_URL="${GIT_URL:-https://github.com/ArduPilot/ardupilot.git}"
VEHICLE="${VEHICLE:-ArduCopter}"  
WAF_BOARD="${WAF_BOARD:-sitl}"
RUN_SITL_FLAG="${RUN_SITL:-0}"

MARKER_START="# >>> ardupilot-sitl PATH additions >>>"
MARKER_END="# <<< ardupilot-sitl PATH additions <<<"

echo "==== ArduPilot SITL automated installer (with .bashrc PATH updates) ===="
echo "Target dir : $ARDUPILOT_DIR"
echo "Vehicle    : $VEHICLE"
echo

# Don't run as root
if [ "$EUID" -eq 0 ]; then
  echo "ERROR: Do not run this script as root. Run it as a normal user." >&2
  exit 1
fi

# Ensure apt-get present
if ! command -v apt-get >/dev/null 2>&1; then
  echo "ERROR: apt-get not found. This script is intended for Debian/Ubuntu." >&2
  exit 2
fi

echo "Updating apt and installing base tools (git, curl, ca-certificates, lsb-release, python3, build-essential)..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends git curl ca-certificates lsb-release python3 python3-venv python3-pip build-essential

# Configure git to use https for submodules if necessary
git config --global url."https://".insteadOf git:// || true

# Clone or update ArduPilot repo
if [ -d "$ARDUPILOT_DIR/.git" ]; then
  echo "ArduPilot repo already exists in $ARDUPILOT_DIR — updating..."
  cd "$ARDUPILOT_DIR"
  git fetch --all --prune
  git checkout --quiet master || true
  git pull --rebase --autostash || true
  git submodule sync --recursive || true
  git submodule update --init --recursive || true
else
  echo "Cloning ArduPilot into $ARDUPILOT_DIR (this may take a while)..."
  git clone --recurse-submodules "$GIT_URL" "$ARDUPILOT_DIR"
  cd "$ARDUPILOT_DIR"
fi

# Find the install-prereqs script
PREREQ_CANDIDATES=(
  "Tools/environment_install/install-prereqs-ubuntu.sh"
  "Tools/scripts/install-prereqs-ubuntu.sh"
  "Tools/environment_install/install-prereqs-ubuntu-20.04.sh"
  "Tools/environment_install/install-prereqs-ubuntu-22.04.sh"
)

PREREQ_SCRIPT=""
for p in "${PREREQ_CANDIDATES[@]}"; do
  if [ -x "$ARDUPILOT_DIR/$p" ] || [ -f "$ARDUPILOT_DIR/$p" ]; then
    PREREQ_SCRIPT="$ARDUPILOT_DIR/$p"
    break
  fi
done

if [ -z "$PREREQ_SCRIPT" ]; then
  found=$(find "$ARDUPILOT_DIR/Tools" -maxdepth 3 -type f -iname "install-prereqs-ubuntu*.sh" 2>/dev/null || true)
  if [ -n "$found" ]; then
    PREREQ_SCRIPT=$(echo "$found" | head -n1)
  fi
fi

if [ -z "$PREREQ_SCRIPT" ]; then
  echo "ERROR: could not find install-prereqs-ubuntu.sh inside the repo. Expected it under Tools/..." >&2
  exit 3
fi

echo "Found prerequisites installer: $PREREQ_SCRIPT"
echo "Running prerequisites installer (this will use sudo internally as required)..."
cd "$ARDUPILOT_DIR"
chmod +x "$PREREQ_SCRIPT"

# Run prereqs script. Many versions support -y; try it, fallback to without.
if "$PREREQ_SCRIPT" -y; then
  echo "Prereqs installer finished."
else
  echo "Prereqs installer returned non-zero with -y; trying without -y..."
  if "$PREREQ_SCRIPT"; then
    echo "Prereqs installer finished (without -y)."
  else
    echo "ERROR: prerequisites installer failed. Aborting." >&2
    exit 4
  fi
fi

# Source typical profile files the installer may have modified (best-effort)
if [ -f "$HOME/.profile" ]; then
  # shellcheck disable=SC1090
  . "$HOME/.profile" || true
fi
if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  . "$HOME/.bashrc" || true
fi

# Activate venv if created by installer
if [ -f "$HOME/venv-ardupilot/bin/activate" ]; then
  echo "Activating Python virtualenv: ~/venv-ardupilot"
  # shellcheck disable=SC1091
  . "$HOME/venv-ardupilot/bin/activate"
fi

# Ensure empy known-good version
echo "Ensuring empy==3.3.4 (compatibility fix)..."
python3 -m pip install --user --upgrade pip || true
python3 -m pip install --user empy==3.3.4 || true

# Install python requirements from repo if present
if [ -f "$ARDUPILOT_DIR/requirements.txt" ]; then
  echo "Installing Python requirements from $ARDUPILOT_DIR/requirements.txt"
  python3 -m pip install --user -r "$ARDUPILOT_DIR/requirements.txt" || true
fi
if [ -f "$ARDUPILOT_DIR/Tools/autotest/requirements.txt" ]; then
  echo "Installing Python requirements for Tools/autotest"
  python3 -m pip install --user -r "$ARDUPILOT_DIR/Tools/autotest/requirements.txt" || true
fi

# Configure & build SITL
echo "Configuring waf for board: $WAF_BOARD"
if [ -x ./waf ]; then
  WAF_CMD="./waf"
elif [ -x ./waf-light ]; then
  WAF_CMD="./waf-light"
else
  WAF_CMD="./waf"
fi

echo "Running: $WAF_CMD configure --board $WAF_BOARD"
$WAF_CMD configure --board "$WAF_BOARD"

BUILD_TARGET=""
case "$VEHICLE" in
  ArduCopter|copter|Copter) BUILD_TARGET="copter" ;;
  ArduPlane|plane|Plane) BUILD_TARGET="plane" ;;
  ArduRover|rover|Rover) BUILD_TARGET="rover" ;;
  ArduSub|sub|Sub) BUILD_TARGET="sub" ;;
  *) BUILD_TARGET="copter" ;;
esac

echo "Starting WAF build for target: $BUILD_TARGET (this will take a while)..."
if $WAF_CMD "$BUILD_TARGET"; then
  echo "WAF build completed successfully."
else
  echo "ERROR: waf build failed. Check output above for errors." >&2
  exit 5
fi

# -----------------------------
# Update ~/.bashrc (idempotent)
# -----------------------------
BASHRC="$HOME/.bashrc"
if [ ! -f "$BASHRC" ]; then
  touch "$BASHRC"
fi

# Remove existing marker block if present (safe idempotent replace)
# Use awk to avoid regex pitfalls
awk -v start="$MARKER_START" -v end="$MARKER_END" '
  BEGIN { in_block = 0 }
  $0 == start { in_block = 1; next }
  $0 == end   { in_block = 0; next }
  !in_block { print }
' "$BASHRC" > "${BASHRC}.tmp" && mv "${BASHRC}.tmp" "$BASHRC"


cat >> "$BASHRC" <<EOF

$MARKER_START
# Added by setup_ardupilot_sitl_with_path.sh - ensures SITL tools and ccache are in PATH
export PATH=\$PATH:\$HOME/ardupilot/Tools/autotest
export PATH=/usr/lib/ccache:\$PATH
$MARKER_END
EOF

# Source the updated .bashrc in this shell
. "$BASHRC" || true

SIM_CMD="$ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py -v $VEHICLE -f quad --console --map"

echo
echo "==== Completed ArduPilot + SITL setup ===="
echo "Repository: $ARDUPILOT_DIR"
echo "Built SITL target for: $VEHICLE"
echo
echo "To run SITL (manually), run from the repo root:"
echo "  cd \"$ARDUPILOT_DIR\""
echo "  $SIM_CMD"

exit 0
