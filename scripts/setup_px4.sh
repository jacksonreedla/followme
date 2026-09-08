#!/usr/bin/env bash
# Phase 0 toolchain: Gazebo Harmonic + PX4 SITL build deps on Ubuntu 24.04 / Mint 22.
# Needs sudo. Idempotent. Run:  bash scripts/setup_px4.sh
#
# Why not just PX4's Tools/setup/ubuntu.sh alone: on Linux Mint `lsb_release -rs`
# reports "22.3" (not 24.04) so that script mis-detects the release and skips the
# Gazebo install. We install Gazebo Harmonic ourselves against the real Ubuntu
# codename, then run ubuntu.sh with sim tools disabled.
set -euo pipefail

CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-noble}")
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"

echo "== 1/3 Gazebo Harmonic (codename: $CODENAME)"
sudo apt-get update -qq
sudo apt-get install -y -qq curl lsb-release gnupg
sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
  -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $CODENAME main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
sudo apt-get update -qq
# python3-gz-transport13 / -msgs10 give us camera frames + ground-truth poses in Python.
sudo apt-get install -y -qq gz-harmonic python3-gz-transport13 python3-gz-msgs10 \
  python3-gz-math7 || {
  echo "!! python3-gz-* bindings not installable; fallback is ROS 2 Jazzy + ros_gz_bridge (see plan Phase 0.3)";
  sudo apt-get install -y -qq gz-harmonic; }

echo "== 2/3 PX4 build deps (no NuttX toolchain, no PX4-managed sim tools)"
[ -d "$PX4_DIR" ] || { echo "PX4 not cloned at $PX4_DIR"; exit 1; }
bash "$PX4_DIR/Tools/setup/ubuntu.sh" --no-nuttx --no-sim-tools

echo "== 3/3 versions"
gz sim --version | head -1
python3 -c "import gz.transport13, gz.msgs10; print('gz python bindings OK')" || true
echo "Done. Next: cd $PX4_DIR && make px4_sitl gz_x500_mono_cam"
