"""Sample green-screen clip + background plate for webcam-free live demos.

Assets live under ``demo_assets/`` (gitignored media). They are created locally
with OpenCV/ffmpeg so ``--demo`` works offline after first generation. An
optional network fetch can replace the clip with a short CC sample.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

DEMO_DIRNAME = "demo_assets"
CLIP_NAME = "greenscreen_demo.mp4"
PLATE_NAME = "background_plate.png"

# Optional short CC green-screen sample (used only with --fetch).
# Free Stock Footage Archive — CC BY 4.0 (credit freestockfootagearchive.com).
OPTIONAL_CLIP_URL = (
    "https://freestockfootagearchive.com/wp-content/uploads/2019/07/"
    "Single_Circle_Transition_Green_Screen.mp4"
)


def demo_dir(package_root: str | Path) -> Path:
    return Path(package_root).resolve() / DEMO_DIRNAME


def demo_paths(package_root: str | Path) -> tuple[Path, Path]:
    root = demo_dir(package_root)
    return root / CLIP_NAME, root / PLATE_NAME


def write_background_plate(path: Path, width: int = 1280, height: int = 720) -> None:
    """Write a simple scenic plate (sky → horizon → ground) as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    sky = np.zeros((height, width, 3), dtype=np.float32)
    sky[..., 0] = 0.15 + 0.35 * (1.0 - y)          # B
    sky[..., 1] = 0.35 + 0.40 * (1.0 - y)          # G
    sky[..., 2] = 0.55 + 0.40 * (1.0 - y)          # R
    ground = np.zeros_like(sky)
    ground[..., 0] = 0.12
    ground[..., 1] = 0.28 + 0.15 * y
    ground[..., 2] = 0.18
    horizon = 0.55
    blend = np.clip((y - horizon) / 0.08, 0.0, 1.0)[..., None]
    img = sky * (1.0 - blend) + ground * blend
    # Soft sun glow
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = int(width * 0.78), int(height * 0.22)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glow = np.clip(1.0 - dist / (height * 0.35), 0.0, 1.0)[..., None]
    img = np.clip(img + glow * np.array([0.15, 0.35, 0.55], dtype=np.float32), 0, 1)
    bgr = (img[..., :3] * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"failed to write background plate: {path}")


def _draw_subject(frame: np.ndarray, t: float, cx: int, cy: int, scale: float) -> None:
    """Draw a simple standing figure (non-green) onto a BGR frame."""
    s = scale
    # subtle bob + sway
    bob = int(6 * math.sin(t * 2.0 * math.pi))
    sway = int(10 * math.sin(t * 2.0 * math.pi * 0.5))
    x = cx + sway
    y = cy + bob
    skin = (160, 185, 220)       # BGR
    shirt = (80, 60, 180)
    pants = (90, 70, 40)
    hair = (30, 40, 50)
    # legs
    cv2.rectangle(frame, (x - int(18 * s), y + int(40 * s)),
                  (x - int(2 * s), y + int(110 * s)), pants, -1)
    cv2.rectangle(frame, (x + int(2 * s), y + int(40 * s)),
                  (x + int(18 * s), y + int(110 * s)), pants, -1)
    # torso
    cv2.rectangle(frame, (x - int(28 * s), y - int(10 * s)),
                  (x + int(28 * s), y + int(45 * s)), shirt, -1)
    # head
    cv2.circle(frame, (x, y - int(40 * s)), int(22 * s), skin, -1)
    cv2.ellipse(frame, (x, y - int(48 * s)), (int(24 * s), int(16 * s)),
                0, 180, 360, hair, -1)
    # arms
    arm_y = y + int(5 * s)
    cv2.line(frame, (x - int(28 * s), arm_y),
             (x - int(55 * s), y + int(35 * s) + bob // 2), skin, int(10 * s))
    cv2.line(frame, (x + int(28 * s), arm_y),
             (x + int(55 * s), y + int(35 * s) - bob // 2), skin, int(10 * s))


def generate_greenscreen_clip(
    path: Path,
    *,
    width: int = 960,
    height: int = 540,
    fps: float = 24.0,
    seconds: float = 4.0,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """Synthesize a short chroma-green clip with a moving subject."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(round(seconds * fps)))
    green = (0, 177, 64)  # chroma-ish BGR
    frames: list[np.ndarray] = []
    for i in range(n):
        t = i / max(n - 1, 1)
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:] = green
        # walk left→right
        cx = int(width * (0.25 + 0.50 * t))
        cy = int(height * 0.52)
        _draw_subject(frame, t, cx, cy, scale=height / 540.0)
        frames.append(frame)

    # Prefer ffmpeg H.264 for broad OpenCV/OBS compatibility.
    if shutil.which(ffmpeg_bin):
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame))
        proc.stdin.close()
        if proc.wait() != 0 or not path.is_file():
            raise RuntimeError(f"ffmpeg failed to write demo clip: {path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def fetch_optional_clip(path: Path, url: str = OPTIONAL_CLIP_URL, timeout: float = 60.0) -> None:
    """Download an optional external sample (may be larger / CC BY)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"[demo] downloading {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(path)


def ensure_demo_assets(
    package_root: str | Path,
    *,
    fetch: bool = False,
    force: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[Path, Path]:
    """Ensure demo clip + plate exist; return ``(clip_path, plate_path)``."""
    clip, plate = demo_paths(package_root)
    demo_dir(package_root).mkdir(parents=True, exist_ok=True)

    if force or not plate.is_file():
        print(f"[demo] writing {plate}")
        write_background_plate(plate)
    else:
        print(f"[demo] plate ok: {plate}")

    if force or not clip.is_file():
        if fetch:
            try:
                fetch_optional_clip(clip)
                print(f"[demo] clip ok (fetched): {clip}")
            except Exception as exc:
                print(f"[demo] fetch failed ({exc}); generating local clip")
                generate_greenscreen_clip(clip, ffmpeg_bin=ffmpeg_bin)
                print(f"[demo] clip ok (generated): {clip}")
        else:
            print(f"[demo] generating {clip}")
            generate_greenscreen_clip(clip, ffmpeg_bin=ffmpeg_bin)
            print(f"[demo] clip ok (generated): {clip}")
    else:
        print(f"[demo] clip ok: {clip}")

    if not clip.is_file() or not plate.is_file():
        raise FileNotFoundError(
            f"demo assets missing under {demo_dir(package_root)}. "
            "Run: python scripts/download_demo_assets.py")
    return clip, plate
