from .model import CorridorKeyV3


class CorridorKeyV2(CorridorKeyV3):
    """Public CorridorKey v2 runtime alias for the exported model."""


__all__ = ["CorridorKeyV2", "LiveConfig", "LiveKeySession", "build_live_meta"]


def __getattr__(name: str):
    # Lazy export so importing CorridorKeyV2 does not pull OpenCV live deps
    # until the live path is actually used.
    if name in ("LiveConfig", "LiveKeySession", "build_live_meta"):
        from . import live as _live
        return getattr(_live, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
