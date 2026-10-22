#!/usr/bin/env python3
"""Create (or optionally fetch) webcam-free demo assets under demo_assets/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridorkey.demo_assets import ensure_demo_assets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--fetch", action="store_true",
                    help="Try downloading a short CC sample clip (else generate).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ffmpeg-bin", default="ffmpeg")
    args = ap.parse_args()
    clip, plate = ensure_demo_assets(
        args.package_root, fetch=args.fetch, force=args.force,
        ffmpeg_bin=args.ffmpeg_bin)
    print(f"clip:  {clip}")
    print(f"plate: {plate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
