#!/usr/bin/env python3
"""Real-time CorridorKey for live streamers — GPU keying into OBS.

Windows + Linux. Capture from webcam, file, **desktop/monitor**, or ``--demo``.

Examples::

  # Webcam → OBS virtual camera
  python live_stream.py --obs --source 0 --background plate.png

  # Desktop greenscreen: green windows become transparent in OBS
  python live_stream.py --desktop --obs --transparent --preview

  # Webcam-free sample loop (CUDA still required)
  python live_stream.py --demo --preview
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

from corridorkey.desktop_capture import parse_desktop_source
from corridorkey.live import (
    DEFAULT_OBS_BACKEND,
    DEFAULT_OBS_V4L2_DEVICE,
    LiveConfig,
    LiveKeySession,
    build_live_meta,
    find_obs_v4l2_device,
    is_linux,
    is_windows,
    list_v4l2_device_names,
    load_live_hint,
    obs_setup_instructions,
    open_capture,
    parse_capture_source,
    pyvirtualcam_available,
    require_cuda_gpu,
    run_live_loop,
    validate_obs_backend,
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
            "CorridorKey v2 real-time GPU keying for live streamers "
            "(Windows + Linux). Webcam, desktop capture, or --demo → OBS."
        ))
    ap.add_argument(
        "--source", default="0",
        help="Capture source: webcam index (0), device path, video file, URL, "
             "or 'desktop' / 'monitor:N'.")
    ap.add_argument(
        "--desktop", action="store_true",
        help="Capture the desktop/monitor via mss (Windows + Linux/X11). "
             "Same as --source desktop. Use --monitor N to pick a display.")
    ap.add_argument(
        "--monitor", type=int, default=None,
        help="Monitor index for desktop capture: 0=all, 1=first display, …")
    ap.add_argument(
        "--demo", action="store_true",
        help="Webcam-free demo: generate/use sample greenscreen clip + plate, "
             "loop with preview (CUDA still required).")
    ap.add_argument(
        "--background", default=None,
        help="Background plate image for composite. Default: solid color "
             "(ignored with --transparent).")
    ap.add_argument(
        "--background-rgb", type=_parse_rgb, default=(0.08, 0.08, 0.10),
        help="Solid sRGB background when --background is omitted (R,G,B in 0..1).")
    ap.add_argument(
        "--transparent", action="store_true",
        help="Output keyed cutout with alpha (green/background see-through). "
             "Preview uses a checker; --obs prefers BGRA virtual cam.")
    ap.add_argument(
        "--obs-alpha", action="store_true",
        help="Force BGRA alpha on the OBS virtual camera (implied by "
             "--transparent). Falls back to cutout-on-black if unsupported.")
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
    ap.add_argument(
        "--no-force-green", action="store_true",
        help="Disable extra chroma-green→transparent pass (on by default with "
             "--desktop / --transparent).")

    # --- OBS / outputs ---
    ap.add_argument(
        "--obs", action="store_true",
        help=(
            "Streamer mode: GPU-key and feed a virtual camera for OBS "
            f"(default backend: {DEFAULT_OBS_BACKEND}, Windows+Linux). "
            "Preview defaults off (unless --demo / --transparent / --preview)."
        ))
    ap.add_argument(
        "--obs-backend", choices=("pyvirtualcam", "v4l2"),
        default=DEFAULT_OBS_BACKEND,
        help=(
            "Virtual-camera backend. "
            f"pyvirtualcam (default): Windows + Linux. "
            "v4l2: Linux-only ffmpeg→v4l2loopback (no pyvirtualcam)."
        ))
    ap.add_argument(
        "--v4l2-device", default=None,
        help=(
            "Linux only: v4l2loopback node for --obs-backend v4l2, or to pin "
            f"pyvirtualcam's device (default {DEFAULT_OBS_V4L2_DEVICE})."
        ))
    ap.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None,
                    help="OpenCV preview window (press q to quit).")
    ap.add_argument("--preview-mode", choices=("composite", "alpha", "fg", "checker"),
                    default="composite")
    ap.add_argument("--no-fps-overlay", action="store_true")
    ap.add_argument("--virtual-cam", action="store_true",
                    help="Enable pyvirtualcam without --obs.")
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
        help="Print CUDA + virtual-camera readiness and OBS steps; do not run keying.")
    return ap


def _resolve_preview(args: argparse.Namespace) -> bool:
    if args.preview is not None:
        return bool(args.preview)
    # Demo / transparent desktop UX wants an on-screen window by default.
    if args.demo or args.transparent:
        return True
    return not bool(args.obs)


def _resolve_v4l2_device(args: argparse.Namespace, backend: str) -> str | None:
    if args.v4l2_device:
        if not is_linux():
            raise RuntimeError(
                "--v4l2-device is Linux-only. On Windows use "
                "--obs-backend pyvirtualcam without --v4l2-device.")
        return args.v4l2_device
    if backend != "v4l2":
        return None
    found = find_obs_v4l2_device(None)
    return found or DEFAULT_OBS_V4L2_DEVICE


def run_check_obs() -> int:
    print("CorridorKey OBS readiness check")
    print("===============================")
    print(f"  Platform: {sys.platform}")
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        print(f"  CUDA GPU: OK — {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA GPU: MISSING")
        print("  → Live keying will refuse to start (GPU required, no CPU fallback).")

    pvc = pyvirtualcam_available()
    print(f"  pyvirtualcam: {'OK' if pvc else 'MISSING (pip install pyvirtualcam)'}")

    try:
        from corridorkey.desktop_capture import list_monitors
        mons = list_monitors()
        print(f"  mss desktop capture: OK ({len(mons)} monitor entries, "
              f"0=all … {len(mons) - 1})")
    except Exception as exc:
        print(f"  mss desktop capture: MISSING ({exc})")

    if is_linux():
        devices = list_v4l2_device_names()
        if devices:
            print("  V4L2 devices:")
            for dev, name in devices.items():
                print(f"    {dev}: {name}")
        else:
            print("  V4L2 devices: none (needed for pyvirtualcam / --obs-backend v4l2)")
        target = find_obs_v4l2_device(None) or DEFAULT_OBS_V4L2_DEVICE
        exists = Path(target).exists()
        print(f"  Linux loopback target: {target} ({'present' if exists else 'MISSING'})")
        if not exists:
            print("  → sudo bash scripts/setup_obs_vcam_linux.sh")
        linux_ok = exists
    else:
        linux_ok = True
        if is_windows():
            print("  Windows virtual cam: provided by pyvirtualcam (BGRA alpha OK)")

    print()
    print(obs_setup_instructions(backend=DEFAULT_OBS_BACKEND))
    print("Desktop greenscreen tip:")
    print("  python live_stream.py --desktop --obs --transparent --preview")

    ready = cuda_ok and pvc and linux_ok
    if not ready:
        return 1
    print("Ready: python live_stream.py --obs --source 0 --background plate.png")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.check_obs:
        return run_check_obs()

    require_cuda_gpu("live streaming / OBS feed")
    backend = validate_obs_backend(args.obs_backend)
    package_root = Path(__file__).resolve().parent

    # --- demo assets (non-blocking optional path) ---
    if args.demo:
        from corridorkey.demo_assets import ensure_demo_assets
        clip, plate = ensure_demo_assets(package_root, ffmpeg_bin=args.ffmpeg_bin)
        args.source = str(clip)
        if args.background is None and not args.transparent:
            args.background = str(plate)
        args.loop = True
        print(f"Demo mode: source={args.source}  background={args.background}")

    use_desktop, monitor = parse_desktop_source(
        args.source, bool(args.desktop), args.monitor)
    if use_desktop:
        args.desktop = True

    transparent = bool(args.transparent)
    # Desktop greenscreen UX: force green→transparent unless opted out.
    force_green = (
        not args.no_force_green
        and (use_desktop or transparent or args.demo)
    )
    if use_desktop and not args.no_force_green:
        transparent = True  # default transparent cutout for desktop

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
        force_green_transparent=force_green,
    )

    preview = _resolve_preview(args)
    use_obs = bool(args.obs or args.virtual_cam)
    v4l2_device = _resolve_v4l2_device(args, backend) if use_obs else (
        args.v4l2_device if args.v4l2_device else None)
    virtual_cam = use_obs and backend == "pyvirtualcam"
    if use_obs and backend == "v4l2":
        virtual_cam = False
        if not v4l2_device:
            v4l2_device = DEFAULT_OBS_V4L2_DEVICE
        if not Path(v4l2_device).exists():
            print(obs_setup_instructions(backend=backend, v4l2_device=v4l2_device),
                  file=sys.stderr)
            raise RuntimeError(
                f"OBS v4l2 device {v4l2_device!r} missing. "
                "Run: sudo bash scripts/setup_obs_vcam_linux.sh"
            )

    if use_obs:
        print(obs_setup_instructions(backend=backend, v4l2_device=v4l2_device))
        if backend == "pyvirtualcam" and not pyvirtualcam_available():
            raise RuntimeError(
                "pip install pyvirtualcam  # required for --obs on Windows and Linux")

    paths = RuntimePaths.from_package(package_root, args.weights)
    print("Loading model for live path (CUDA: foundations + stage1 + stage2)...")
    model = load_model(paths, device="cuda")

    # Probe size for hint / meta
    if use_desktop:
        from corridorkey.desktop_capture import DesktopCapture
        probe_cap = DesktopCapture(monitor=monitor)
        probe = probe_cap.read()
        probe_cap.close()
        source: str | int | None = None
    else:
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
        f"Live ready (GPU): platform={sys.platform}  "
        f"desktop={use_desktop} monitor={monitor if use_desktop else '-'}  "
        f"source={args.source!r}  capture={w}x{h}  "
        f"transparent={transparent}  force_green={force_green}  "
        f"obs={use_obs}  backend={backend}  preview={preview}")

    if not preview and not virtual_cam and not v4l2_device and not args.ffmpeg_out:
        print("warning: no sinks enabled; enabling preview so the loop is visible",
              file=sys.stderr)
        preview = True

    stats = run_live_loop(
        session,
        source,
        desktop=use_desktop,
        monitor=monitor if use_desktop else 0,
        background_path=args.background,
        preview=preview,
        preview_mode=args.preview_mode,
        virtual_cam=virtual_cam,
        v4l2_device=v4l2_device if (backend == "v4l2" or args.v4l2_device) else None,
        obs_backend=backend,
        obs_alpha=bool(args.obs_alpha or transparent),
        transparent=transparent,
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
