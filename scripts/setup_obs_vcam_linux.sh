#!/usr/bin/env bash
# Linux-only helper: load v4l2loopback for OBS / pyvirtualcam.
#
# Windows does not use this script — on Windows, `pip install pyvirtualcam`
# provides the virtual camera for `python live_stream.py --obs`.
#
# Creates /dev/video10 labeled "CorridorKey" with exclusive_caps so OBS lists
# it as a normal Video Capture Device (V4L2). Used by:
#   - default --obs (pyvirtualcam on Linux), and
#   - optional --obs-backend v4l2
#
# Usage:
#   sudo bash scripts/setup_obs_vcam_linux.sh
#   sudo bash scripts/setup_obs_vcam_linux.sh --video-nr 10
set -euo pipefail

VIDEO_NR=10
CARD_LABEL="CorridorKey"
DEVICES=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-nr) VIDEO_NR="$2"; shift 2 ;;
    --label) CARD_LABEL="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: sudo $0 [--video-nr N] [--label NAME]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please re-run with sudo (needs modprobe)." >&2
  exit 1
fi

if ! modprobe -n v4l2loopback 2>/dev/null; then
  echo "v4l2loopback kernel module not found."
  echo "Install it, then re-run this script:"
  echo "  Debian/Ubuntu: sudo apt install v4l2loopback-dkms v4l2loopback-utils"
  echo "  Fedora:        sudo dnf install v4l2loopback"
  echo "  Arch:          sudo pacman -S v4l2loopback-dkms"
  exit 1
fi

# Reload so card_label / video_nr apply cleanly.
modprobe -r v4l2loopback 2>/dev/null || true
modprobe v4l2loopback \
  devices="${DEVICES}" \
  video_nr="${VIDEO_NR}" \
  card_label="${CARD_LABEL}" \
  exclusive_caps=1

DEV="/dev/video${VIDEO_NR}"
# Give the video group (and usually the logged-in user via plugdev/video) access.
if getent group video >/dev/null; then
  chgrp video "${DEV}" 2>/dev/null || true
  chmod 660 "${DEV}" 2>/dev/null || true
fi

echo "OK: ${DEV} ready (label=${CARD_LABEL}, exclusive_caps=1)"
echo
echo "Next:"
echo "  python live_stream.py --obs --source 0 --background plate.png"
echo "  OBS → Sources → + → Video Capture Device (V4L2) → ${CARD_LABEL} / ${DEV}"
echo
echo "Tip: to persist across reboots, add a modprobe.d snippet, e.g."
echo "  echo 'options v4l2loopback devices=1 video_nr=${VIDEO_NR} card_label=${CARD_LABEL} exclusive_caps=1' | sudo tee /etc/modprobe.d/corridorkey-vcam.conf"
echo "  echo v4l2loopback | sudo tee /etc/modules-load.d/corridorkey-vcam.conf"
