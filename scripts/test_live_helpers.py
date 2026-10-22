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
        DEFAULT_OBS_V4L2_DEVICE,
        LiveConfig,
        LiveLoopStats,
        _preview_key,
        build_live_meta,
        find_obs_v4l2_device,
        list_v4l2_device_names,
        load_background_bgr,
        obs_setup_instructions,
        parse_capture_source,
        require_cuda_gpu,
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

    # --- meta matches offline probe shape contracts ---
    meta = build_live_meta(720, 1280, shape_quantum=128, model_scale=1.0, fps=30.0)
    assert meta["source_hw"] == (720, 1280)
    assert meta["output_hw"] == (720, 1280)
    assert meta["shape_hw"][0] % 128 == 0
    assert meta["shape_hw"][1] % 128 == 0
    assert meta["shape_hw"] == (768, 1280)  # 720→768 pad to quantum 128
    assert meta["num_frames"] == -1
    pad_l, pad_r, pad_t, pad_b = meta["pad"]
    assert pad_l + pad_r + meta["model_native_hw"][1] == meta["shape_hw"][1]
    assert pad_t + pad_b + meta["model_native_hw"][0] == meta["shape_hw"][0]

    meta_half = build_live_meta(1080, 1920, model_scale=0.5, shape_quantum=128)
    assert meta_half["model_native_hw"] == (540, 960)
    assert meta_half["output_hw"] == (1080, 1920)

    # --- solid / image-less background ---
    bg = load_background_bgr(None, (64, 96), (0.0, 1.0, 0.0))
    assert bg.shape == (64, 96, 3)
    assert bg.dtype == np.uint8
    # green sRGB → BGR: B=0, G=255, R=0
    assert int(bg[0, 0, 1]) == 255

    # --- stats + config ---
    stats = LiveLoopStats()
    stats.record(10.0)
    stats.record(20.0)
    summary = stats.summary()
    assert "frames=2" in summary
    assert "mean=15.0" in summary

    cfg = LiveConfig()
    assert cfg.carry_hint is True
    assert cfg.low_vram is False
    assert cfg.model_scale == 1.0

    # --- OBS helpers (safe without a real loopback) ---
    assert DEFAULT_OBS_V4L2_DEVICE == "/dev/video10"
    assert isinstance(list_v4l2_device_names(), dict)
    # preferred missing path → None
    assert find_obs_v4l2_device("/dev/corridorkey_missing_device_xyz") is None
    text = obs_setup_instructions(DEFAULT_OBS_V4L2_DEVICE)
    assert "Video Capture Device" in text
    assert DEFAULT_OBS_V4L2_DEVICE in text
    assert "Chroma Key" in text

    # GPU is required — on this CI/dev box without CUDA we expect refusal.
    try:
        require_cuda_gpu("test")
        # If CUDA is present, that is also fine.
        cuda_required_ok = True
    except RuntimeError as exc:
        assert "CUDA" in str(exc)
        cuda_required_ok = True
    assert cuda_required_ok

    # --- CLI: --obs defaults ---
    ns = build_argparser().parse_args([
        "--source", "1",
        "--obs",
        "--background", "plate.png",
        "--model-scale", "0.5",
        "--mirror",
    ])
    assert ns.obs is True
    assert ns.obs_backend == "v4l2"
    assert _resolve_preview(ns) is False  # --obs ⇒ preview off by default
    assert _resolve_v4l2_device(ns) in (
        DEFAULT_OBS_V4L2_DEVICE,
        find_obs_v4l2_device(None) or DEFAULT_OBS_V4L2_DEVICE,
    )

    ns2 = build_argparser().parse_args([
        "--obs", "--preview", "--v4l2-device", "/dev/video12",
    ])
    assert _resolve_preview(ns2) is True
    assert _resolve_v4l2_device(ns2) == "/dev/video12"

    ns3 = build_argparser().parse_args([
        "--source", "1",
        "--no-preview",
        "--ffmpeg-out", "out.mp4",
        "--max-frames", "10",
        "--loop",
    ])
    assert ns3.preview is False
    assert ns3.ffmpeg_out == "out.mp4"
    assert ns3.max_frames == 10
    assert ns3.loop is True

    ns4 = build_argparser().parse_args(["--check-obs"])
    assert ns4.check_obs is True

    setup = root / "scripts" / "setup_obs_vcam_linux.sh"
    assert setup.is_file()
    body = setup.read_text(encoding="utf-8")
    assert "v4l2loopback" in body
    assert "CorridorKey" in body
    assert "exclusive_caps=1" in body

    print("live helpers ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
