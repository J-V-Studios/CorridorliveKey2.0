#!/usr/bin/env python3
"""Real-time CorridorKey entry point for live streamers.

Pipeline:
  capture (webcam / file / URL) → per-frame key → composite → preview / virt cam / ffmpeg

Example (webcam 0, image plate, on-screen preview)::

  python live_stream.py --source 0 --background plate.png --preview

Example (file loop → low-latency ffmpeg file, no GUI)::

  python live_stream.py --source clip.mp4 --loop --no-preview \\
    --ffmpeg-out keyed.mp4 --max-frames 300 --model-scale 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

from corridorkey.live import (
    LiveConfig,
    LiveKeySession,
    build_live_meta,
    load_live_hint,
    parse_capture_source,
    run_live_loop,
)
from corridorkey.runtime import RuntimePaths, load_model
from corridorkey.video_io import none_path


def _parse_rgb(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected R,G,B floats in 0..1 (e.g. 0.08,0.08,0.10)")
    rgb = tuple(float(p) for p in parts)
    if any(c < 0.0 or c > 1.0 for c in rgb):
        raise argparse.ArgumentTypeError("RGB components must be in [0, 1]")
    return rgb  # type: ignore[return-value]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CorridorKey v2 real-time / live-stream keying")
    ap.add_argument(
        "--source", default="0",
        help="Capture source: webcam index (0), device path, video file, or URL.")
    ap.add_argument(
        "--background", default=None,
        help="Background plate image for composite. Default: solid color.")
    ap.add_argument(
        "--background-rgb", type=_parse_rgb, default=(0.08, 0.08, 0.10),
        help="Solid sRGB background when --background is omitted (R,G,B in 0..1).")
    ap.add_argument("--hint-first-frame", default=None,
                    help="Optional first-frame matte hint (same as offline).")
    ap.add_argument("--hint-quality", type=float, default=0.95)
    ap.add_argument("--carry-hint-quality", type=float, default=0.95)
    ap.add_argument("--no-carry-hint", action="store_true",
                    help="Do not feed previous-frame aux matte as a hint.")
    ap.add_argument("--despill-strength", type=float, default=0.5)
    ap.add_argument(
        "--model-scale", type=float, default=1.0,
        help="Lanczos scale before the model (e.g. 0.5 for faster 1080→540). "
             "Output composite is still returned at capture resolution.")
    ap.add_argument("--mirror", action="store_true",
                    help="Mirror the capture horizontally (selfie / webcam).")
    ap.add_argument("--low-vram", action="store_true",
                    help="Chunked attention; still keeps stages on GPU (no memmaps).")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True,
                    help="Show an OpenCV preview window (press q to quit).")
    ap.add_argument("--preview-mode", choices=("composite", "alpha", "fg"),
                    default="composite")
    ap.add_argument("--no-fps-overlay", action="store_true")
    ap.add_argument("--virtual-cam", action="store_true",
                    help="Send composite to a virtual webcam (needs pyvirtualcam).")
    ap.add_argument(
        "--ffmpeg-out", default=None,
        help="Pipe composite frames to ffmpeg (file path, RTMP URL, etc.).")
    ap.add_argument("--ffmpeg-bin", default="ffmpeg")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (0 = until EOF / quit).")
    ap.add_argument("--loop", action="store_true",
                    help="Loop file sources (useful for dry-runs without a camera).")
    ap.add_argument("--target-fps", type=float, default=0.0,
                    help="Nominal FPS for sinks/overlay (0 = use capture FPS).")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CorridorKey live streaming requires a CUDA GPU "
            "(same as offline infer.py).")

    package_root = Path(__file__).resolve().parent
    paths = RuntimePaths.from_package(package_root, args.weights)
    live_cfg = LiveConfig(
        model_scale=args.model_scale,
        hint_quality=args.hint_quality,
        carry_hint=not args.no_carry_hint,
        carry_hint_quality=args.carry_hint_quality,
        despill_strength=args.despill_strength,
        low_vram=args.low_vram,
        background_rgb=args.background_rgb,
        mirror=args.mirror,
        target_fps=args.target_fps,
    )

    print("Loading model for live path (foundations + stage1 + stage2 on GPU)...")
    model = load_model(paths, device="cuda")

    # Probe a single frame so an optional first-frame hint can be sized correctly.
    source = parse_capture_source(args.source)
    from corridorkey.live import open_capture
    cap = open_capture(source)
    ok, probe = cap.read()
    cap.release()
    if not ok or probe is None:
        raise RuntimeError(f"Could not read a probe frame from {args.source!r}")
    h, w = probe.shape[:2]
    fps = args.target_fps if args.target_fps > 0 else 30.0
    meta = build_live_meta(
        h, w, model_scale=live_cfg.model_scale, fps=fps,
        despill_strength=live_cfg.despill_strength)
    initial_hint = load_live_hint(none_path(args.hint_first_frame), meta)

    session = LiveKeySession(
        model, live_cfg, device="cuda", initial_hint=initial_hint)
    print(
        f"Live ready: source={args.source!r}  capture={w}x{h}  "
        f"model_grid={meta['shape_hw'][1]}x{meta['shape_hw'][0]}  "
        f"carry_hint={live_cfg.carry_hint}  mirror={live_cfg.mirror}")
    print("Integration: capture → LiveKeySession.process_frame → "
          "composite → preview / --virtual-cam / --ffmpeg-out")

    if not args.preview and not args.virtual_cam and not args.ffmpeg_out:
        print("warning: no sinks enabled; enabling preview so the loop is visible",
              file=sys.stderr)
        args.preview = True

    stats = run_live_loop(
        session,
        source,
        background_path=args.background,
        preview=args.preview,
        preview_mode=args.preview_mode,
        virtual_cam=args.virtual_cam,
        ffmpeg_out=args.ffmpeg_out,
        ffmpeg_bin=args.ffmpeg_bin,
        max_frames=args.max_frames,
        loop_file=args.loop,
        show_fps=not args.no_fps_overlay,
    )
    print(f"Live session done: {stats.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
