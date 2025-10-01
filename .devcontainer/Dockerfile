# Use the public base image so the devcontainer build is intentionally minimal.
# If you later need to add extra layers (tools, caches, deps), add them below.
FROM kestreldev/l4t-ros2-docker:jazzy

# Keep the devcontainer image minimal; runtime apt/source configuration is performed
# by the repository-level installer script `scripts/rosdep_install.sh` so the
# devcontainer image doesn't force package installation at build-time.

# Example of adding small, local dev-only tooling later:
# RUN apt-get update && apt-get install -y --no-install-recommends vim less && rm -rf /var/lib/apt/lists/*
