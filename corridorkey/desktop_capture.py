"""Cross-platform desktop / monitor capture for the live path.

Uses ``mss`` (Windows + Linux/X11; Wayland often needs XWayland or a
PipeWire-based tool). Returns BGR uint8 frames suitable for LiveKeySession.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def require_mss():
    try:
        import mss  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Desktop capture requires the 'mss' package.\n"
            "  pip install mss\n"
            "Linux: works best on X11 (or XWayland). Pure Wayland may need "
            "extra portal/PipeWire tooling outside this helper."
        ) from exc
    import mss
    return mss


class DesktopCapture:
    """Grab the full virtual desktop or a single monitor via mss."""

    def __init__(self, monitor: int = 0):
        """
        Parameters
        ----------
        monitor:
            ``0`` = all monitors combined (mss ``monitors[0]``).
            ``1`` = first display, ``2`` = second, …
        """
        mss = require_mss()
        self._sct = mss.mss()
        monitors = self._sct.monitors
        if monitor < 0 or monitor >= len(monitors):
            raise ValueError(
                f"monitor index {monitor} out of range; "
                f"available 0..{len(monitors) - 1} "
                f"(0=all, 1..=per-display). Seen: {monitors}")
        self.monitor_index = int(monitor)
        self._region = dict(monitors[self.monitor_index])
        self.width = int(self._region["width"])
        self.height = int(self._region["height"])

    @property
    def description(self) -> str:
        r = self._region
        return (
            f"mss monitor={self.monitor_index} "
            f"{self.width}x{self.height} @ ({r['left']},{r['top']})"
        )

    def read(self) -> np.ndarray:
        """Return one BGR uint8 frame (H, W, 3)."""
        import cv2
        shot = self._sct.grab(self._region)
        # mss gives BGRA
        bgra = np.asarray(shot, dtype=np.uint8)
        if bgra.ndim != 3 or bgra.shape[2] < 3:
            raise RuntimeError(f"unexpected mss frame shape: {bgra.shape}")
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def frames(self, *, max_frames: int = 0) -> Iterator[np.ndarray]:
        n = 0
        while True:
            yield self.read()
            n += 1
            if max_frames > 0 and n >= max_frames:
                break

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


def list_monitors() -> list[dict]:
    """Return mss monitor dicts (index 0 = virtual desktop bounding box)."""
    mss = require_mss()
    with mss.mss() as sct:
        return [dict(m) for m in sct.monitors]


def parse_desktop_source(source: str | int | None, desktop_flag: bool,
                         monitor: int | None) -> tuple[bool, int]:
    """Resolve CLI into ``(use_desktop, monitor_index)``."""
    if desktop_flag:
        return True, 0 if monitor is None else int(monitor)
    if monitor is not None:
        return True, int(monitor)
    if isinstance(source, str):
        text = source.strip().lower()
        if text in ("desktop", "screen", "display"):
            return True, 0 if monitor is None else int(monitor)
        if text.startswith("monitor:") or text.startswith("monitor="):
            return True, int(text.split(":", 1)[-1].split("=", 1)[-1])
        if text.startswith("desktop:") or text.startswith("desktop="):
            part = text.split(":", 1)[-1].split("=", 1)[-1]
            return True, int(part) if part not in ("", "all") else 0
    return False, 0
