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
        LiveConfig,
        LiveLoopStats,
        _preview_key,
        build_live_meta,
        load_background_bgr,
        parse_capture_source,
    )
    from live_stream import build_argparser, _parse_rgb

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

    # --- stats + CLI wiring ---
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

    ns = build_argparser().parse_args([
        "--source", "1",
        "--no-preview",
        "--ffmpeg-out", "out.mp4",
        "--model-scale", "0.5",
        "--mirror",
        "--loop",
        "--max-frames", "10",
    ])
    assert ns.source == "1"
    assert ns.preview is False
    assert ns.ffmpeg_out == "out.mp4"
    assert ns.model_scale == 0.5
    assert ns.mirror is True
    assert ns.loop is True
    assert ns.max_frames == 10

    print("live helpers ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
