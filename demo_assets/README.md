# Demo assets (webcam-free live preview)

Generated locally (not committed) by:

```bash
python scripts/download_demo_assets.py
# optional network sample:
python scripts/download_demo_assets.py --fetch
```

Or automatically on first `python live_stream.py --demo …`.

Produces:

- `greenscreen_demo.mp4` — short synthetic green-screen clip
- `background_plate.png` — simple scenic plate for composite

CUDA GPU + model weights are still required to run the demo keyer.
