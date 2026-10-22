#!/usr/bin/env python3
"""CPU-only checks for the real-time live path helpers (no weights / GPU)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package_root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    root = Path(args.package_root).resolve()
    sys.path.insert(0, str(root))

    from corridorkey.live import (
        DEFAULT_OBS_BACKEND,
        DEFAULT_OBS_V4L2_DEVICE,
        LiveConfig,
        LiveLoopStats,
        _preview_key,
        build_live_meta,
        find_obs_v4l2_device,
        is_linux,
        is_windows,
        list_v4l2_device_names,
        load_background_bgr,
        obs_setup_instructions,
        parse_capture_source,
        require_cuda_gpu,
        validate_obs_backend,
    )
    from live_stream import (
        _parse_rgb,
        _resolve_preview,
        _resolve_v4l2_device,
        build_argparser,
    )

    # --- parse helpers ---
    assert parse_capture_source("0") == 0
    assert parse_capture_source(2) == 2
    assert parse_capture_source("/dev/video0") == "/dev/video0"
    assert parse_capture_source("clip.mp4") == "clip.mp4"
    assert _parse_rgb("0.1,0.2,0.3") == (0.1, 0.2, 0.3)
    assert _preview_key("alpha") == "alpha_u8"
    assert _preview_key("composite") == "composite_u8"

    # --- meta ---
    meta = build_live_meta(720, 1280, shape_quantum=128, model_scale=1.0, fps=30.0)
    assert meta["shape_hw"] == (768, 1280)
    assert meta["num_frames"] == -1

    meta_half = build_live_meta(1080, 1920, model_scale=0.5, shape_quantum=128)
    assert meta_half["model_native_hw"] == (540, 960)
    assert meta_half["output_hw"] == (1080, 1920)

    bg = load_background_bgr(None, (64, 96), (0.0, 1.0, 0.0))
    assert bg.shape == (64, 96, 3) and int(bg[0, 0, 1]) == 255

    stats = LiveLoopStats()
    stats.record(10.0)
    stats.record(20.0)
    assert "frames=2" in stats.summary()

    cfg = LiveConfig()
    assert cfg.carry_hint is True and cfg.low_vram is False

    # --- cross-platform OBS defaults ---
    assert DEFAULT_OBS_BACKEND == "pyvirtualcam"
    assert validate_obs_backend("pyvirtualcam") == "pyvirtualcam"
    if is_linux():
        assert validate_obs_backend("v4l2") == "v4l2"
    else:
        try:
            validate_obs_backend("v4l2")
            raise AssertionError("v4l2 must be rejected off Linux")
        except RuntimeError as exc:
            assert "Linux-only" in str(exc)

    text = obs_setup_instructions(backend="pyvirtualcam")
    assert "Windows OBS steps" in text
    assert "Linux OBS steps" in text
    assert "Video Capture Device" in text
    assert "Chroma Key" in text
    assert "pyvirtualcam" in text

    assert isinstance(list_v4l2_device_names(), dict)
    assert find_obs_v4l2_device("/dev/corridorkey_missing_device_xyz") is None

    try:
        require_cuda_gpu("test")
    except RuntimeError as exc:
        assert "CUDA" in str(exc)

    # --- CLI: --obs defaults to pyvirtualcam on all platforms ---
    ns = build_argparser().parse_args([
        "--source", "1",
        "--obs",
        "--background", "plate.png",
        "--model-scale", "0.5",
        "--mirror",
    ])
    assert ns.obs is True
    assert ns.obs_backend == "pyvirtualcam"
    assert _resolve_preview(ns) is False
    # pyvirtualcam does not force a v4l2 device
    assert _resolve_v4l2_device(ns, "pyvirtualcam") is None

    ns2 = build_argparser().parse_args([
        "--obs", "--preview", "--obs-backend", "v4l2",
        "--v4l2-device", "/dev/video12",
    ])
    assert _resolve_preview(ns2) is True
    if is_linux():
        assert _resolve_v4l2_device(ns2, "v4l2") == "/dev/video12"
    else:
        try:
            _resolve_v4l2_device(ns2, "v4l2")
            raise AssertionError("expected Windows rejection of --v4l2-device")
        except RuntimeError as exc:
            assert "Linux-only" in str(exc)

    ns3 = build_argparser().parse_args([
        "--no-preview", "--ffmpeg-out", "out.mp4", "--max-frames", "10", "--loop",
    ])
    assert ns3.preview is False and ns3.max_frames == 10

    ns4 = build_argparser().parse_args(["--check-obs"])
    assert ns4.check_obs is True

    setup = root / "scripts" / "setup_obs_vcam_linux.sh"
    assert setup.is_file()
    body = setup.read_text(encoding="utf-8")
    assert "Linux-only" in body or "Windows does not use" in body
    assert "v4l2loopback" in body

    # sanity: platform helpers
    assert is_linux() or is_windows() or sys.platform.startswith("darwin")

    print("live helpers ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
