#!/usr/bin/env python3
"""Real-time CorridorKey for live streamers — GPU keying into OBS (Linux).

Primary streamer UX (Linux):

  # one-time
  sudo bash scripts/setup_obs_vcam_linux.sh

  # every session — CUDA keying → /dev/video10 for OBS
  python live_stream.py --obs --source 0 --background plate.png

Then in OBS: Sources → + → Video Capture Device (V4L2) → CorridorKey /dev/video10

Keying always runs on an NVIDIA CUDA GPU (no CPU-only primary path).
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
    DEFAULT_OBS_V4L2_DEVICE,
    LiveConfig,
    LiveKeySession,
    build_live_meta,
    find_obs_v4l2_device,
    list_v4l2_device_names,
    load_live_hint,
    obs_setup_instructions,
    open_capture,
    parse_capture_source,
    require_cuda_gpu,
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
        description=(
            "CorridorKey v2 real-time GPU keying for live streamers. "
            "Preferred OBS path on Linux: --obs (ffmpeg→v4l2loopback)."
        ))
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

    # --- OBS / outputs ---
    ap.add_argument(
        "--obs", action="store_true",
        help=(
            "Streamer mode: GPU-key and feed a Linux v4l2loopback device for OBS. "
            "Defaults preview off; uses --v4l2-device (or auto /dev/video10)."
        ))
    ap.add_argument(
        "--v4l2-device", default=None,
        help=(
            f"v4l2loopback node for OBS (default with --obs: "
            f"{DEFAULT_OBS_V4L2_DEVICE} or a device labeled CorridorKey)."
        ))
    ap.add_argument(
        "--obs-backend", choices=("v4l2", "pyvirtualcam"), default="v4l2",
        help=(
            "How to write the OBS virtual camera. "
            "v4l2 (default, recommended on Linux): ffmpeg→v4l2loopback. "
            "pyvirtualcam: optional pip package."
        ))
    ap.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None,
                    help="OpenCV preview window (press q to quit). "
                         "Default: on, unless --obs (then off).")
    ap.add_argument("--preview-mode", choices=("composite", "alpha", "fg"),
                    default="composite")
    ap.add_argument("--no-fps-overlay", action="store_true")
    ap.add_argument("--virtual-cam", action="store_true",
                    help="Also/instead use pyvirtualcam (see --obs-backend).")
    ap.add_argument(
        "--ffmpeg-out", default=None,
        help="Optional extra sink: pipe composite to ffmpeg (file / RTMP / MPEG-TS).")
    ap.add_argument("--ffmpeg-bin", default="ffmpeg")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (0 = until EOF / quit).")
    ap.add_argument("--loop", action="store_true",
                    help="Loop file sources (useful for dry-runs without a camera).")
    ap.add_argument("--target-fps", type=float, default=0.0,
                    help="Nominal FPS for sinks/overlay (0 = use capture FPS).")
    ap.add_argument(
        "--check-obs", action="store_true",
        help="Print CUDA + v4l2loopback status and OBS steps; do not run keying.")
    return ap


def _resolve_preview(args: argparse.Namespace) -> bool:
    if args.preview is not None:
        return bool(args.preview)
    # --obs defaults to headless (OBS is the viewer).
    return not bool(args.obs)


def _resolve_v4l2_device(args: argparse.Namespace) -> str | None:
    if args.v4l2_device:
        return args.v4l2_device
    if not args.obs:
        return None
    found = find_obs_v4l2_device(None)
    if found:
        return found
    return DEFAULT_OBS_V4L2_DEVICE


def run_check_obs() -> int:
    """Doctor: CUDA + v4l2loopback readiness for the OBS path."""
    print("CorridorKey OBS readiness check")
    print("===============================")
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA GPU: {'OK — ' + torch.cuda.get_device_name(0) if cuda_ok else 'MISSING'}")
    if not cuda_ok:
        print("  → Live keying will refuse to start (GPU required, no CPU fallback).")
    devices = list_v4l2_device_names()
    if devices:
        print("  V4L2 devices:")
        for dev, name in devices.items():
            print(f"    {dev}: {name}")
    else:
        print("  V4L2 devices: none visible under /sys/class/video4linux")
    target = find_obs_v4l2_device(None) or DEFAULT_OBS_V4L2_DEVICE
    exists = Path(target).exists()
    print(f"  OBS target: {target} ({'present' if exists else 'MISSING'})")
    if not exists:
        print("  → Run: sudo bash scripts/setup_obs_vcam_linux.sh")
    print()
    print(obs_setup_instructions(target))
    if not cuda_ok or not exists:
        return 1
    print("Ready: python live_stream.py --obs --source 0 --background plate.png")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.check_obs:
        return run_check_obs()

    require_cuda_gpu("live streaming / OBS feed")

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

    preview = _resolve_preview(args)
    v4l2_device = _resolve_v4l2_device(args)
    virtual_cam = bool(args.virtual_cam) and not v4l2_device

    if args.obs or v4l2_device:
        if v4l2_device and not Path(v4l2_device).exists():
            print(obs_setup_instructions(v4l2_device), file=sys.stderr)
            raise RuntimeError(
                f"OBS v4l2 device {v4l2_device!r} missing. "
                "Run: sudo bash scripts/setup_obs_vcam_linux.sh"
            )
        print(obs_setup_instructions(v4l2_device or DEFAULT_OBS_V4L2_DEVICE))

    print("Loading model for live path (CUDA: foundations + stage1 + stage2)...")
    model = load_model(paths, device="cuda")

    source = parse_capture_source(args.source)
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
        f"Live ready (GPU): source={args.source!r}  capture={w}x{h}  "
        f"model_grid={meta['shape_hw'][1]}x{meta['shape_hw'][0]}  "
        f"obs={bool(args.obs)}  v4l2={v4l2_device!r}  "
        f"backend={args.obs_backend}  preview={preview}")

    if not preview and not v4l2_device and not virtual_cam and not args.ffmpeg_out:
        print("warning: no sinks enabled; enabling preview so the loop is visible",
              file=sys.stderr)
        preview = True

    stats = run_live_loop(
        session,
        source,
        background_path=args.background,
        preview=preview,
        preview_mode=args.preview_mode,
        virtual_cam=virtual_cam,
        v4l2_device=v4l2_device,
        obs_backend=args.obs_backend,
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
