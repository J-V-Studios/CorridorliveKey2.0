#!/usr/bin/env python3
"""CPU-only checks for the real-time live path helpers (no weights / GPU)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package_root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    root = Path(args.package_root).resolve()
    sys.path.insert(0, str(root))

    from corridorkey.desktop_capture import parse_desktop_source
    from corridorkey.demo_assets import ensure_demo_assets, write_background_plate
    from corridorkey.live import (
        DEFAULT_OBS_BACKEND,
        LiveConfig,
        LiveLoopStats,
        _preview_key,
        build_live_meta,
        obs_setup_instructions,
        parse_capture_source,
        require_cuda_gpu,
        validate_obs_backend,
    )
    from live_stream import (
        _parse_rgb,
        _resolve_preview,
        build_argparser,
    )

    assert parse_capture_source("0") == 0
    assert _parse_rgb("0.1,0.2,0.3") == (0.1, 0.2, 0.3)

    assert parse_desktop_source("desktop", False, None) == (True, 0)
    assert parse_desktop_source("monitor:2", False, None) == (True, 2)
    assert parse_desktop_source("0", True, 1) == (True, 1)
    assert parse_desktop_source("0", False, None) == (False, 0)

    meta = build_live_meta(720, 1280, shape_quantum=128, model_scale=1.0)
    assert meta["shape_hw"] == (768, 1280)

    plate = root / "demo_assets" / "_test_plate.png"
    write_background_plate(plate, width=64, height=48)
    assert plate.is_file() and plate.stat().st_size > 0
    plate.unlink(missing_ok=True)

    clip, plate2 = ensure_demo_assets(root, force=True)
    assert clip.is_file() and plate2.is_file()

    assert DEFAULT_OBS_BACKEND == "pyvirtualcam"
    assert validate_obs_backend("pyvirtualcam") == "pyvirtualcam"
    text = obs_setup_instructions(backend="pyvirtualcam")
    assert "Windows OBS steps" in text and "Linux OBS steps" in text

    try:
        require_cuda_gpu("test")
    except RuntimeError as exc:
        assert "CUDA" in str(exc)

    assert _preview_key("composite", transparent=True) == "checker_u8"
    assert _preview_key("cutout") == "cutout_bgra"

    ns = build_argparser().parse_args(["--desktop", "--obs", "--transparent"])
    assert ns.desktop and ns.transparent and ns.obs
    assert _resolve_preview(ns) is True

    ns2 = build_argparser().parse_args(["--demo", "--obs"])
    assert ns2.demo
    assert _resolve_preview(ns2) is True

    ns3 = build_argparser().parse_args(["--obs", "--source", "0"])
    assert _resolve_preview(ns3) is False

    cfg = LiveConfig(force_green_transparent=True)
    assert cfg.force_green_transparent is True

    stats = LiveLoopStats()
    stats.record(12.0)
    assert "frames=1" in stats.summary()

    # Despill must accept RGB and preserve optional alpha (regression for
    # transparent desktop path that previously passed 4ch into the blend).
    import torch
    from corridorkey.runtime import _despill_average_srgb_torch
    rgb = torch.rand(1, 3, 8, 8)
    out = _despill_average_srgb_torch(rgb, strength=0.5)
    assert out.shape == (1, 3, 8, 8)
    rgba = torch.rand(1, 4, 8, 8)
    out_a = _despill_average_srgb_torch(rgba, strength=0.5)
    assert out_a.shape == (1, 4, 8, 8)
    assert torch.allclose(out_a[:, 3:4], rgba[:, 3:4])

    # Channel split for live render: (N,4,H,W) uses dim-1, not dim-0.
    fake = torch.zeros(1, 4, 16, 16)
    fake[:, 0] = 0.25
    fake[:, 1:4] = 0.5
    assert fake[:, 0:1].shape[1] == 1
    assert fake[:, 1:4].shape[1] == 3
    # The old bug: fake[0:1] is still 4-channel
    assert fake[0:1].shape[1] == 4

    print("live helpers ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
