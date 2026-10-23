"""Real-time / live-stream CorridorKey path.

Designed for streamers and low-latency capture loops:

  webcam / capture device  →  per-frame key  →  composite  →  preview / virt cam / ffmpeg

Unlike ``run_inference`` (file-oriented Hann windows + full-clip RVM prepass),
this keeps foundations, Stage 1, and Stage 2 on the GPU and processes one frame
at a time while carrying RVM recurrent state (and optional aux carry-hint)
across frames for temporal continuity.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import color_utils as cu
from .runtime import (
    InferenceConfig,
    _build_foundation_inputs,
    _crop_native_torch,
    _despill_average_srgb_torch,
    _linear_adjust_inverse_torch,
    _resize_output_torch,
    configure_low_vram_inference,
)
from .video_io import (
    ceil_to_multiple,
    load_initial_hint_frame,
    make_checker,
    preprocess_video_frame_views,
    read_image_rgb,
    scaled_hw,
)


# ---------------------------------------------------------------------------
# Meta / config
# ---------------------------------------------------------------------------


@dataclass
class LiveConfig:
    """Streamer-oriented defaults for the real-time path."""

    shape_quantum: int = 128
    model_scale: float = 1.0
    hint_quality: float = 0.95
    carry_hint: bool = True
    carry_hint_quality: float = 0.95
    despill_strength: float = 0.5
    # Keep stages resident on GPU. low_vram staging adds disk I/O latency.
    low_vram: bool = False
    low_vram_foundation_chunk: int = 4
    low_vram_mlp_chunk_tokens: int = 65536
    low_vram_enc_frame_chunk: int = 1
    low_vram_swin_window_batch: int = 16
    low_vram_full_attn_query_chunk_tokens: int = 2048
    linear_brightness: float = 1.0
    linear_contrast: float = 1.0
    linear_contrast_pivot: float = 0.18
    # Solid sRGB background when no plate is loaded (dark gray, streamer-friendly).
    background_rgb: tuple[float, float, float] = (0.08, 0.08, 0.10)
    mirror: bool = False
    target_fps: float = 0.0  # 0 = uncapped
    # Force chroma-green pixels toward alpha=0 (desktop green-window UX).
    force_green_transparent: bool = False
    green_hue_center: float = 60.0   # OpenCV HSV degrees-ish (H in 0..180 → *2)
    green_hue_width: float = 25.0
    green_sat_min: float = 0.35
    green_val_min: float = 0.20


def build_live_meta(
    height: int,
    width: int,
    *,
    shape_quantum: int = 128,
    model_scale: float = 1.0,
    fps: float = 30.0,
    linear_brightness: float = 1.0,
    linear_contrast: float = 1.0,
    linear_contrast_pivot: float = 0.18,
    despill_strength: float = 0.5,
) -> dict:
    """Build the same meta dict shape as ``probe_video_meta`` for a live feed."""
    h_native, w_native = int(height), int(width)
    q = int(shape_quantum)
    h_model, w_model = scaled_hw(h_native, w_native, model_scale)
    h_out = ceil_to_multiple(h_model, q)
    w_out = ceil_to_multiple(w_model, q)
    pad_l = (w_out - w_model) // 2
    pad_r = w_out - w_model - pad_l
    pad_t = (h_out - h_model) // 2
    pad_b = h_out - h_model - pad_t
    return {
        "native_fps": float(fps),
        "native_hw": (h_native, w_native),
        "output_hw": (h_native, w_native),
        "model_native_hw": (h_model, w_model),
        "model_scale": float(model_scale),
        "source_hw": (h_native, w_native),
        "shape_hw": (h_out, w_out),
        "start_frame": 0,
        "num_frames": -1,  # unbounded live stream
        "pad": (pad_l, pad_r, pad_t, pad_b),
        "native_resolution": True,
        "linear_brightness": float(linear_brightness),
        "linear_contrast": float(linear_contrast),
        "linear_contrast_pivot": float(linear_contrast_pivot),
        "despill_strength": float(despill_strength),
        "cutout_linear": False,
    }


def parse_capture_source(source: str | int) -> str | int:
    """Normalize CLI source: int index, device path, file, or URL."""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text


def open_capture(source: str | int) -> cv2.VideoCapture:
    """Open a webcam index, device path, file, or network URL."""
    src = parse_capture_source(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture source: {source!r}")
    return cap


def load_background_bgr(
    path: str | None,
    hw: tuple[int, int],
    solid_rgb: tuple[float, float, float],
) -> np.ndarray:
    """Return uint8 BGR background matching ``hw`` (H, W)."""
    h, w = hw
    if path:
        img = read_image_rgb(path)
        img = np.clip(img, 0.0, 1.0)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    rgb = np.array(solid_rgb, dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    bgr = rgb[0, 0, ::-1]
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[:] = (bgr * 255.0).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class LiveKeySession:
    """Stateful per-frame CorridorKey for live capture loops."""

    def __init__(
        self,
        model,
        cfg: LiveConfig | None = None,
        *,
        device: str = "cuda",
        initial_hint: torch.Tensor | None = None,
    ):
        require_cuda_gpu("LiveKeySession")
        if not str(device).startswith("cuda"):
            raise RuntimeError(
                f"LiveKeySession device must be CUDA (got {device!r}); "
                "CPU is not a supported primary keying path.")
        self.model = model
        self.cfg = cfg or LiveConfig()
        self.device = torch.device(device)
        self.meta: dict | None = None
        self.rvm_state = None
        self._carry_hint: torch.Tensor | None = None
        self._initial_hint = initial_hint
        self._frame_index = 0
        self._checker_t: torch.Tensor | None = None
        self._prepare_model()

    def _prepare_model(self) -> None:
        # Latency path: keep all stages resident (no CPU↔GPU phase swaps).
        self.model.foundations.to(self.device)
        self.model.stage1.to(self.device)
        self.model.stage2.to(self.device)
        self.model.eval()
        if self.cfg.low_vram:
            # Still allow chunked attention if the card is tight, but avoid
            # memmap prepasses that only make sense for finite clips.
            infer = InferenceConfig(
                low_vram=True,
                low_vram_foundation_chunk=self.cfg.low_vram_foundation_chunk,
                low_vram_mlp_chunk_tokens=self.cfg.low_vram_mlp_chunk_tokens,
                low_vram_enc_frame_chunk=self.cfg.low_vram_enc_frame_chunk,
                low_vram_swin_window_batch=self.cfg.low_vram_swin_window_batch,
                low_vram_full_attn_query_chunk_tokens=(
                    self.cfg.low_vram_full_attn_query_chunk_tokens),
            )
            configure_low_vram_inference(self.model, infer)
        torch.cuda.empty_cache()

    def reset(self) -> None:
        """Clear temporal state (RVM + carry hint). Call after a hard cut."""
        self.rvm_state = None
        self._carry_hint = None
        self._frame_index = 0

    def ensure_meta(self, frame_bgr: np.ndarray, fps: float = 30.0) -> dict:
        h, w = frame_bgr.shape[:2]
        if (
            self.meta is None
            or self.meta["source_hw"] != (h, w)
        ):
            self.meta = build_live_meta(
                h, w,
                shape_quantum=self.cfg.shape_quantum,
                model_scale=self.cfg.model_scale,
                fps=fps,
                linear_brightness=self.cfg.linear_brightness,
                linear_contrast=self.cfg.linear_contrast,
                linear_contrast_pivot=self.cfg.linear_contrast_pivot,
                despill_strength=self.cfg.despill_strength,
            )
            H_out, W_out = self.meta["output_hw"]
            self._checker_t = torch.from_numpy(make_checker(H_out, W_out))
            self.reset()
        return self.meta

    @torch.inference_mode()
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        hint: torch.Tensor | None = None,
        background_bgr: np.ndarray | None = None,
        fps: float = 30.0,
    ) -> dict[str, Any]:
        """Key one BGR uint8 frame.

        Returns dict with:
          - ``alpha_u8``: HxWx3 matte preview
          - ``fg_u8``: HxWx3 foreground preview
          - ``composite_u8``: HxWx3 FG over background (despilled)
          - ``alpha``: 1xHxW float tensor on CPU (cropped to native)
          - ``latency_ms``: end-to-end GPU key latency for this frame
        """
        t0 = time.perf_counter()
        if self.cfg.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        meta = self.ensure_meta(frame_bgr, fps=fps)
        model_t, foundation_t, _review = preprocess_video_frame_views(
            frame_bgr, meta, need_model=True)
        assert model_t is not None

        H, W = meta["shape_hw"]
        device = self.device
        rgb_c = model_t.unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
        display = foundation_t.unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
        rgb_radio, rgb_moge = _build_foundation_inputs(display)
        # RVM wants display-coded [0,1] at H/4 (same grid as C-RADIO).
        rgb_rvm = rgb_radio.clamp(0.0, 1.0)

        hint_cpu = torch.zeros(1, 1, 1, H, W, dtype=torch.bfloat16)
        hint_quality = torch.full((1, 1), -1.0, dtype=torch.bfloat16)
        used_hint = False
        if hint is not None:
            hint_cpu[0, 0] = hint.to(dtype=torch.bfloat16)
            hint_quality.fill_(float(self.cfg.hint_quality))
            used_hint = True
        elif self._frame_index == 0 and self._initial_hint is not None:
            hint_cpu[0, 0] = self._initial_hint.to(dtype=torch.bfloat16)
            hint_quality.fill_(float(self.cfg.hint_quality))
            used_hint = True
        elif self.cfg.carry_hint and self._carry_hint is not None:
            hint_cpu[0, 0] = self._carry_hint.to(dtype=torch.bfloat16)
            hint_quality.fill_(float(self.cfg.carry_hint_quality))
            used_hint = True
        if not used_hint:
            hint_quality.fill_(-1.0)

        hint_c = hint_cpu.to(device, non_blocking=True)
        hint_q = hint_quality.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            ff = self.model.foundations(
                rgb_cradio=rgb_radio.reshape(1, 3, *rgb_radio.shape[-2:]),
                rgb_moge=rgb_moge.reshape(1, 3, *rgb_moge.shape[-2:]),
                rgb_rvm=rgb_rvm,
                rvm_state=self.rvm_state,
            )
            cradio_feat = ff.get("cradio")
            moge_feat = ff["moge"]
            rvm_feat = ff.get("rvm")
            self.rvm_state = ff.get("rvm_state", self.rvm_state)
            if cradio_feat is not None:
                cradio_feat = cradio_feat.reshape(1, 1, -1, H // 64, W // 64)
            else:
                cradio_feat = rgb_c.new_zeros(1, 1, 0, H // 64, W // 64)
            moge_feat = moge_feat.reshape(1, 1, -1, H // 64, W // 64)
            if rvm_feat is not None:
                rvm_feat = rvm_feat.reshape(
                    1, 1, rvm_feat.shape[-3], rvm_feat.shape[-2], rvm_feat.shape[-1])
            s1 = self.model.stage1(
                rgb_c, hint_c, moge_feat,
                cradio_feat=cradio_feat,
                rvm_feat=rvm_feat,
                t_stride=1.0,
                hint_quality=hint_q,
            )
            baton = s1["baton"]
            if self.cfg.carry_hint:
                aux_alpha = s1["aux"][0, :, 0:1].float()
                aux_full = F.interpolate(
                    aux_alpha, size=(H, W), mode="bilinear",
                    align_corners=False).clamp(0.0, 1.0)
                self._carry_hint = aux_full[0].to(
                    "cpu", dtype=torch.bfloat16).clone()
            logits = self.model.stage2(rgb_c, baton)
            if isinstance(logits, tuple):
                logits = logits[0]

        rendered = self._render_live(
            meta, logits[0], background_bgr, source_bgr=frame_bgr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._frame_index += 1
        rendered["latency_ms"] = latency_ms
        rendered["frame_index"] = self._frame_index - 1
        return rendered

    def _green_keep_mask(self, source_bgr: np.ndarray, hw: tuple[int, int]) -> torch.Tensor:
        """Return (1,1,H,W) keep-mask in [0,1] — 0 on chroma green, 1 elsewhere.

        Uses OpenCV HSV (H in 0..179). Default center 60 ≈ pure green.
        """
        h, w = hw
        if source_bgr.shape[:2] != (h, w):
            source_bgr = cv2.resize(source_bgr, (w, h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue = hsv[..., 0]
        sat = hsv[..., 1] / 255.0
        val = hsv[..., 2] / 255.0
        center = float(self.cfg.green_hue_center)  # OpenCV H units
        width = float(self.cfg.green_hue_width)
        dh = np.abs(hue - center)
        dh = np.minimum(dh, 180.0 - dh)
        core = (dh <= width * 0.55) & (sat >= self.cfg.green_sat_min) & (
            val >= self.cfg.green_val_min)
        # Soft falloff outside the core
        soft = np.clip((dh - width * 0.35) / max(width * 0.65, 1e-3), 0.0, 1.0)
        keep = np.where(core, 0.0, soft).astype(np.float32)
        return torch.from_numpy(keep)[None, None].to(self.device)

    def _render_live(
        self,
        meta: dict,
        logits: torch.Tensor,
        background_bgr: np.ndarray | None,
        *,
        source_bgr: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Render alpha / fg / composite / BGRA cutout at native resolution.

        ``logits`` must be ``(N, 4, H, W)`` or ``(4, H, W)`` — channel 0 is
        alpha, channels 1:4 are RGB (same layout as the offline render path).
        """
        if logits.ndim == 3:
            logits = logits.unsqueeze(0)
        if logits.ndim != 4 or logits.shape[1] < 4:
            raise ValueError(
                f"expected logits (N, 4, H, W) or (4, H, W); got {tuple(logits.shape)}")
        # Channel axis is dim 1 (NOT dim 0). Dim-0 slicing was treating the
        # whole 4-channel frame as alpha and broke despill on desktop/transparent.
        alpha_model = logits[:, 0:1].float()
        fg_model = logits[:, 1:4].float()
        pa = _resize_output_torch(
            _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0), meta)
        if self.cfg.force_green_transparent and source_bgr is not None:
            keep = self._green_keep_mask(source_bgr, meta["output_hw"])
            pa = (pa * keep.to(device=pa.device, dtype=pa.dtype)).clamp(0.0, 1.0)
        pf_model = _crop_native_torch(fg_model, meta)
        pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
        pf_linear_pos = pf_linear.clamp(min=0.0)
        pf = _resize_output_torch(
            cu.linear_to_srgb(pf_linear_pos), meta).clamp(0.0, 1.0)
        if pf.shape[1] != 3:
            raise RuntimeError(
                f"foreground must be RGB (3ch) before despill; got {tuple(pf.shape)}")
        strength = float(meta.get("despill_strength", 0.5))
        pf_despill = _despill_average_srgb_torch(pf, strength=strength)

        alpha_u8 = (pa[0, 0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        alpha_bgr = cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2BGR)
        fg_u8 = (pf_despill[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        fg_bgr = cv2.cvtColor(fg_u8, cv2.COLOR_RGB2BGR)

        # Straight BGRA cutout (transparent where alpha=0) for OBS virt-cam alpha.
        cutout_bgra = np.dstack([fg_bgr, alpha_u8])

        h, w = meta["output_hw"]
        if background_bgr is None:
            bg = load_background_bgr(None, (h, w), self.cfg.background_rgb)
        else:
            if background_bgr.shape[:2] != (h, w):
                bg = cv2.resize(background_bgr, (w, h), interpolation=cv2.INTER_AREA)
            else:
                bg = background_bgr
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bg_t = torch.from_numpy(bg_rgb.transpose(2, 0, 1)).to(
            device=pf.device, dtype=pf.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(bg_t)
        fg_lin = cu.srgb_to_linear(pf_despill)
        comp_lin = fg_lin * pa + bg_lin * (1.0 - pa)
        comp = cu.linear_to_srgb(comp_lin.clamp(min=0.0)).clamp(0.0, 1.0)
        comp_u8 = (comp[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        composite_bgr = cv2.cvtColor(comp_u8, cv2.COLOR_RGB2BGR)

        # Checker preview for transparent mode (see-through green → checker).
        checker = make_checker(h, w)
        checker_t = torch.from_numpy(checker).to(
            device=pf.device, dtype=pf.dtype).unsqueeze(0)
        chk_lin = cu.srgb_to_linear(checker_t)
        chk_comp = cu.linear_to_srgb(
            (fg_lin * pa + chk_lin * (1.0 - pa)).clamp(min=0.0)).clamp(0.0, 1.0)
        checker_u8 = (chk_comp[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        checker_bgr = cv2.cvtColor(checker_u8, cv2.COLOR_RGB2BGR)

        return {
            "alpha_u8": alpha_bgr,
            "fg_u8": fg_bgr,
            "composite_u8": composite_bgr,
            "checker_u8": checker_bgr,
            "cutout_bgra": cutout_bgra,
            "alpha": pa.detach().cpu(),
        }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


DEFAULT_OBS_V4L2_DEVICE = "/dev/video10"
OBS_V4L2_CARD_LABEL = "CorridorKey"
# Cross-platform default: pyvirtualcam works on Windows + Linux (+ macOS).
DEFAULT_OBS_BACKEND = "pyvirtualcam"


def require_cuda_gpu(purpose: str = "live keying") -> None:
    """Live / OBS path is GPU-only — refuse CPU as a primary path."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CorridorKey {purpose} requires an NVIDIA CUDA GPU "
            "(no CPU-only keying path). Install CUDA-enabled PyTorch and "
            "use an RTX 30-series or newer GPU, same as offline infer.py."
        )


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_windows() -> bool:
    return sys.platform.startswith("win")


def list_v4l2_device_names() -> dict[str, str]:
    """Map ``/dev/videoN`` → card name from sysfs. Empty on non-Linux."""
    out: dict[str, str] = {}
    if not is_linux():
        return out
    sys_root = Path("/sys/class/video4linux")
    if not sys_root.is_dir():
        return out
    for entry in sorted(sys_root.iterdir()):
        name_file = entry / "name"
        if not name_file.is_file():
            continue
        card = name_file.read_text(encoding="utf-8", errors="replace").strip()
        dev = f"/dev/{entry.name}"
        out[dev] = card
    return out


def find_obs_v4l2_device(
    preferred: str | None = None,
    *,
    card_label: str = OBS_V4L2_CARD_LABEL,
) -> str | None:
    """Pick a v4l2loopback device (Linux only). Returns None on Windows/macOS."""
    if not is_linux() and preferred is None:
        return None
    devices = list_v4l2_device_names()
    if preferred:
        if preferred in devices or Path(preferred).exists():
            return preferred
        return None
    for dev, name in devices.items():
        if card_label.lower() in name.lower() or "loopback" in name.lower():
            return dev
    if Path(DEFAULT_OBS_V4L2_DEVICE).exists():
        return DEFAULT_OBS_V4L2_DEVICE
    return None


def pyvirtualcam_available() -> bool:
    try:
        import pyvirtualcam  # noqa: F401
        return True
    except ImportError:
        return False


def obs_setup_instructions(
    *,
    backend: str = DEFAULT_OBS_BACKEND,
    v4l2_device: str | None = None,
    device_hint: str | None = None,
) -> str:
    """OBS Studio steps for Windows and Linux (virtual camera ingest)."""
    device = device_hint or "(the virtual camera printed when live_stream starts)"
    v4l2 = v4l2_device or DEFAULT_OBS_V4L2_DEVICE
    common = """\
Shared notes (Windows + Linux)
------------------------------
- Keying runs on an NVIDIA CUDA GPU only (no CPU-only primary path).
- Leave `live_stream.py --obs` running while OBS is open.
- In OBS: Sources → + → Video Capture Device
  (Linux builds may say "Video Capture Device (V4L2)").
- Do NOT add an OBS Chroma Key / green-screen filter — CorridorKey already
  keyed and composited on the GPU.
- Fit the source to the canvas; set OBS canvas to your stream size
  (e.g. 1920x1080).
"""
    windows = f"""\
Windows OBS steps
-----------------
1. One-time: pip install pyvirtualcam
   (first run may install a virtual-camera driver; approve UAC if prompted.)
2. Start the keyer:
     python live_stream.py --obs --source 0 --background plate.png
3. In OBS: Sources → + → Video Capture Device
4. Device: select the virtual camera (often "OBS Virtual Camera", or the
   name printed by live_stream — currently: {device})
5. Resolution/FPS: match your webcam, or Automatic.
"""
    linux = f"""\
Linux OBS steps (default backend: pyvirtualcam)
-----------------------------------------------
1. One-time:
     pip install pyvirtualcam
     # pyvirtualcam on Linux needs a v4l2loopback device:
     sudo apt install v4l2loopback-dkms v4l2loopback-utils   # Debian/Ubuntu
     sudo bash scripts/setup_obs_vcam_linux.sh
2. Start the keyer:
     python live_stream.py --obs --source 0 --background plate.png
3. In OBS: Sources → + → Video Capture Device (V4L2)
4. Device: select the virtual camera / CorridorKey / {v4l2}
5. Resolution/FPS: match your webcam, or Automatic.

Linux-only alternative (no pyvirtualcam): ffmpeg → v4l2loopback
  python live_stream.py --obs --obs-backend v4l2 --v4l2-device {v4l2}
"""
    backend_note = (
        f"Active --obs-backend: {backend}\n"
        f"(default on all OSes: {DEFAULT_OBS_BACKEND}; "
        f"`v4l2` is Linux-only)\n"
    )
    return backend_note + "\n" + common + "\n" + windows + "\n" + linux


def validate_obs_backend(backend: str) -> str:
    """Normalize backend; gate Linux-only v4l2."""
    backend = (backend or DEFAULT_OBS_BACKEND).strip().lower()
    if backend not in ("pyvirtualcam", "v4l2"):
        raise ValueError(
            f"unsupported --obs-backend {backend!r}; "
            "use 'pyvirtualcam' (Windows+Linux) or 'v4l2' (Linux only)")
    if backend == "v4l2" and not is_linux():
        raise RuntimeError(
            "--obs-backend v4l2 is Linux-only. On Windows use the default "
            "--obs-backend pyvirtualcam (pip install pyvirtualcam).")
    return backend


class FFmpegFrameSink:
    """Pipe BGR uint8 frames to ffmpeg (file, RTMP, MPEG-TS, v4l2, etc.)."""

    def __init__(
        self,
        output: str,
        width: int,
        height: int,
        fps: float,
        *,
        ffmpeg_bin: str = "ffmpeg",
        extra_args: list[str] | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.output = output
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(fps),
            "-i", "-",
        ]
        if extra_args:
            cmd.extend(extra_args)
        else:
            cmd.extend([
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
            ])
        cmd.append(output)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = cv2.resize(
                frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame_bgr))

    def close(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        self.proc.wait(timeout=30)


class V4L2LoopbackSink:
    """Linux-only optional OBS feed: BGR frames → ffmpeg → v4l2loopback.

    Prefer ``VirtualCamSink`` / ``--obs-backend pyvirtualcam`` on both Windows
    and Linux. This backend is a Linux fallback that does not need pyvirtualcam.
    """

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: float,
        *,
        ffmpeg_bin: str = "ffmpeg",
        pix_fmt: str = "yuv420p",
    ):
        if not is_linux():
            raise RuntimeError(
                "V4L2LoopbackSink is Linux-only. On Windows use "
                "--obs-backend pyvirtualcam.")
        if not Path(device).exists():
            raise RuntimeError(
                f"v4l2 device {device!r} not found. "
                "Run: sudo bash scripts/setup_obs_vcam_linux.sh"
            )
        # Uncompressed V4L2 output — low latency for OBS ingest.
        extra = ["-f", "v4l2", "-pix_fmt", pix_fmt]
        self.device = device
        self._sink = FFmpegFrameSink(
            device, width, height, fps,
            ffmpeg_bin=ffmpeg_bin, extra_args=extra)

    def write(self, frame_bgr: np.ndarray) -> None:
        self._sink.write(frame_bgr)

    def close(self) -> None:
        self._sink.close()


class VirtualCamSink:
    """Cross-platform virtual camera via pyvirtualcam (Windows + Linux).

    Windows: DirectShow virtual camera (driver may install on first use).
    Linux: uses a v4l2loopback device under the hood (see setup script).

    When ``with_alpha=True``, prefers BGRA so OBS can composite transparency
    (supported on Windows; Linux v4l2 often falls back to opaque BGR — see docs).
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        *,
        device: str | None = None,
        with_alpha: bool = False,
    ):
        try:
            import pyvirtualcam  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OBS virtual-camera backend requires pyvirtualcam.\n"
                "  pip install pyvirtualcam\n"
                "Linux also needs a v4l2loopback device "
                "(sudo bash scripts/setup_obs_vcam_linux.sh).\n"
                "Linux-only alternative without pyvirtualcam: "
                "--obs-backend v4l2"
            ) from exc
        # Many Linux pyvirtualcam builds only expose BGR (no BGRA/RGBA enum).
        fmt_bgra = getattr(pyvirtualcam.PixelFormat, "BGRA", None)
        fmt_rgba = getattr(pyvirtualcam.PixelFormat, "RGBA", None)
        fmt_bgr = getattr(pyvirtualcam.PixelFormat, "BGR", None)
        want_alpha = bool(with_alpha) and (fmt_bgra is not None or fmt_rgba is not None)
        if with_alpha and not want_alpha:
            print(
                "  [note] This pyvirtualcam build has no BGRA/RGBA pixel format; "
                "sending opaque BGR cutout-on-black (preview checker still shows "
                "transparency). Upgrade pyvirtualcam or use Windows for BGRA alpha.",
                flush=True,
            )
        self.with_alpha = want_alpha
        if self.with_alpha:
            fmt = fmt_bgra if fmt_bgra is not None else fmt_rgba
        else:
            fmt = fmt_bgr
        kwargs: dict[str, Any] = {
            "width": int(width),
            "height": int(height),
            "fps": float(fps),
        }
        if fmt is not None:
            kwargs["fmt"] = fmt
        if device and is_linux():
            kwargs["device"] = device
        try:
            self._cam = pyvirtualcam.Camera(**kwargs)
        except (TypeError, RuntimeError, ValueError, OSError):
            # Retry without device pin / without alpha fmt.
            kwargs.pop("device", None)
            if self.with_alpha and fmt_bgr is not None:
                kwargs["fmt"] = fmt_bgr
                self.with_alpha = False
                print(
                    "  [note] Virtual cam rejected BGRA; falling back to BGR "
                    "cutout-on-black.",
                    flush=True,
                )
            try:
                self._cam = pyvirtualcam.Camera(**kwargs)
            except TypeError:
                kwargs.pop("fmt", None)
                self._cam = pyvirtualcam.Camera(**kwargs)
                self.with_alpha = False
        self.device = getattr(self._cam, "device", device or "(pyvirtualcam)")

    def write(self, frame: np.ndarray) -> None:
        """Accept BGR (HxWx3) or BGRA (HxWx4)."""
        if frame.shape[1] != self._cam.width or frame.shape[0] != self._cam.height:
            frame = cv2.resize(
                frame, (self._cam.width, self._cam.height),
                interpolation=cv2.INTER_AREA)
        if self.with_alpha:
            if frame.shape[2] == 3:
                frame = np.dstack([
                    frame,
                    np.full(frame.shape[:2], 255, dtype=np.uint8),
                ])
            self._cam.send(np.ascontiguousarray(frame))
        else:
            if frame.shape[2] == 4:
                # Composite straight BGRA over black for opaque virt cams.
                bgr = frame[:, :, :3].astype(np.float32)
                a = frame[:, :, 3:4].astype(np.float32) / 255.0
                frame = np.clip(bgr * a, 0, 255).astype(np.uint8)
            self._cam.send(np.ascontiguousarray(frame))
        self._cam.sleep_until_next_frame()

    def close(self) -> None:
        self._cam.close()


@dataclass
class LiveLoopStats:
    frames: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self.frames += 1
        self.latency_ms.append(float(latency_ms))

    def summary(self) -> str:
        if not self.latency_ms:
            return "no frames"
        arr = np.asarray(self.latency_ms, dtype=np.float64)
        return (
            f"frames={self.frames}  "
            f"latency_ms mean={arr.mean():.1f} p50={np.median(arr):.1f} "
            f"p95={np.percentile(arr, 95):.1f}"
        )


def iter_capture_frames(
    cap: cv2.VideoCapture,
    *,
    max_frames: int = 0,
    loop_file: bool = False,
) -> Iterator[np.ndarray]:
    """Yield BGR frames from an open capture, optionally looping file sources."""
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if loop_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
            else:
                break
        yield frame
        n += 1
        if max_frames > 0 and n >= max_frames:
            break


def _preview_key(preview_mode: str, *, transparent: bool = False) -> str:
    if transparent and preview_mode == "composite":
        return "checker_u8"
    return {
        "composite": "composite_u8",
        "alpha": "alpha_u8",
        "fg": "fg_u8",
        "checker": "checker_u8",
        "cutout": "cutout_bgra",
    }.get(preview_mode, "composite_u8")


def run_live_loop(
    session: LiveKeySession,
    source: str | int | None = None,
    *,
    desktop: bool = False,
    monitor: int = 0,
    background_path: str | None = None,
    preview: bool = True,
    preview_mode: str = "composite",
    virtual_cam: bool = False,
    v4l2_device: str | None = None,
    obs_backend: str = DEFAULT_OBS_BACKEND,
    obs_alpha: bool = False,
    transparent: bool = False,
    ffmpeg_out: str | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffmpeg_extra: list[str] | None = None,
    max_frames: int = 0,
    loop_file: bool = False,
    show_fps: bool = True,
) -> LiveLoopStats:
    """Capture → GPU key → composite/cutout → sinks.

    ``desktop=True`` uses mss screen capture (Windows + Linux/X11).
    ``transparent`` / ``obs_alpha`` send BGRA cutout so green/keyed areas are
    see-through in OBS when the virtual camera supports alpha.
    """
    require_cuda_gpu("live loop")
    backend = validate_obs_backend(obs_backend) if (virtual_cam or v4l2_device) else obs_backend
    desktop_cap = None
    cap = None
    if desktop:
        from .desktop_capture import DesktopCapture
        desktop_cap = DesktopCapture(monitor=monitor)
        print(f"  Desktop capture: {desktop_cap.description}", flush=True)
        frame_iter = desktop_cap.frames(max_frames=max_frames)
        fps = float(session.cfg.target_fps) if session.cfg.target_fps > 0 else 30.0
    else:
        if source is None:
            raise ValueError("source is required unless desktop=True")
        cap = open_capture(source)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        if session.cfg.target_fps > 0:
            fps = float(session.cfg.target_fps)
        frame_iter = iter_capture_frames(
            cap, max_frames=max_frames, loop_file=loop_file)

    stats = LiveLoopStats()
    sink_ffmpeg: FFmpegFrameSink | None = None
    sink_v4l2: V4L2LoopbackSink | None = None
    sink_vcam: VirtualCamSink | None = None
    background: np.ndarray | None = None
    window = "CorridorKey Live"
    use_alpha = bool(obs_alpha or transparent)
    try:
        out_key = _preview_key(preview_mode, transparent=transparent)
        for frame in frame_iter:
            meta = session.ensure_meta(frame, fps=fps)
            h, w = meta["output_hw"]
            if background is None and not transparent:
                background = load_background_bgr(
                    background_path, (h, w), session.cfg.background_rgb)
            result = session.process_frame(
                frame,
                background_bgr=None if transparent else background,
                fps=fps,
            )
            stats.record(result["latency_ms"])
            preview_frame = result.get(out_key, result["composite_u8"])
            if preview_frame.ndim == 3 and preview_frame.shape[2] == 4:
                # Preview window: show BGRA over checker
                preview_frame = result["checker_u8"]
            obs_frame = (
                result["cutout_bgra"] if use_alpha else result["composite_u8"]
            )
            want_obs = virtual_cam or (v4l2_device is not None)
            if want_obs:
                if sink_v4l2 is None and sink_vcam is None:
                    if backend == "v4l2":
                        if not v4l2_device:
                            raise RuntimeError(
                                "--obs-backend v4l2 requires --v4l2-device "
                                f"(e.g. {DEFAULT_OBS_V4L2_DEVICE})")
                        sink_v4l2 = V4L2LoopbackSink(
                            v4l2_device, w, h, fps, ffmpeg_bin=ffmpeg_bin)
                        print(f"  OBS feed (Linux v4l2, opaque): {v4l2_device}",
                              flush=True)
                        if use_alpha:
                            print(
                                "  [note] v4l2 backend is RGB-only; cutout is "
                                "composited on black. Prefer --obs-backend "
                                "pyvirtualcam on Windows for BGRA alpha.",
                                flush=True)
                    else:
                        sink_vcam = VirtualCamSink(
                            w, h, fps,
                            device=v4l2_device if is_linux() else None,
                            with_alpha=use_alpha,
                        )
                        print(
                            f"  OBS feed (pyvirtualcam"
                            f"{', BGRA alpha' if sink_vcam.with_alpha else ''}): "
                            f"{sink_vcam.device}",
                            flush=True)
                if sink_v4l2 is not None:
                    # v4l2: always BGR
                    bgr = obs_frame
                    if bgr.shape[2] == 4:
                        a = bgr[:, :, 3:4].astype(np.float32) / 255.0
                        bgr = np.clip(
                            bgr[:, :, :3].astype(np.float32) * a, 0, 255
                        ).astype(np.uint8)
                    sink_v4l2.write(bgr)
                else:
                    assert sink_vcam is not None
                    sink_vcam.write(obs_frame)
            if ffmpeg_out:
                if sink_ffmpeg is None:
                    sink_ffmpeg = FFmpegFrameSink(
                        ffmpeg_out, w, h, fps,
                        ffmpeg_bin=ffmpeg_bin, extra_args=ffmpeg_extra)
                bgr = obs_frame if obs_frame.shape[2] == 3 else result["composite_u8"]
                sink_ffmpeg.write(bgr)
            if preview:
                view = preview_frame
                if show_fps:
                    view = preview_frame.copy()
                    ms = result["latency_ms"]
                    inst = 1000.0 / ms if ms > 1e-3 else 0.0
                    cv2.putText(
                        view, f"{inst:.1f} fps  {ms:.0f} ms",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (40, 220, 40), 2, cv2.LINE_AA)
                cv2.imshow(window, view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if cap is not None:
            cap.release()
        if desktop_cap is not None:
            desktop_cap.close()
        if sink_ffmpeg is not None:
            sink_ffmpeg.close()
        if sink_v4l2 is not None:
            sink_v4l2.close()
        if sink_vcam is not None:
            sink_vcam.close()
        if preview:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    return stats


def live_config_from_inference(cfg: InferenceConfig) -> LiveConfig:
    """Map offline InferenceConfig knobs onto LiveConfig where they overlap."""
    return LiveConfig(
        shape_quantum=cfg.shape_quantum,
        model_scale=cfg.model_scale,
        hint_quality=cfg.hint_quality,
        carry_hint=cfg.carry_hint,
        carry_hint_quality=cfg.carry_hint_quality,
        despill_strength=cfg.despill_strength,
        low_vram=cfg.low_vram,
        low_vram_foundation_chunk=cfg.low_vram_foundation_chunk,
        low_vram_mlp_chunk_tokens=cfg.low_vram_mlp_chunk_tokens,
        low_vram_enc_frame_chunk=cfg.low_vram_enc_frame_chunk,
        low_vram_swin_window_batch=cfg.low_vram_swin_window_batch,
        low_vram_full_attn_query_chunk_tokens=(
            cfg.low_vram_full_attn_query_chunk_tokens),
        linear_brightness=cfg.linear_brightness,
        linear_contrast=cfg.linear_contrast,
        linear_contrast_pivot=cfg.linear_contrast_pivot,
    )


def load_live_hint(path: str | None, meta: dict) -> torch.Tensor | None:
    if not path:
        return None
    return load_initial_hint_frame(path, meta)
