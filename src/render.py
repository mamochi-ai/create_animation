"""Render scenes to PNG frames (parallel) or a quick contact sheet.

  python3 src/render.py s03                 # all frames of s03 -> build/frames/s03/
  python3 src/render.py s03 --sheet 12      # 12 evenly spaced stills -> build/preview/s03_sheet.png
  python3 src/render.py s03 --at 4.5        # single still (local time) -> build/preview/s03_4.50.png
  python3 src/render.py s03 --video         # frames + build/preview/s03.mp4 with scene audio slice
"""
import argparse, importlib, math, os, pathlib, subprocess, sys, time
from multiprocessing import Pool

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(os.environ.get("FILM_DIR", REPO)).resolve()
sys.path.insert(0, str(REPO / "src"))
if ROOT != REPO:
    sys.path.insert(0, str(ROOT))  # film-local `scenes` and `lib` packages win
from common.core import FPS, scene_bounds  # noqa: E402

_mod = None


def _init(scene):
    global _mod
    sys.path.insert(0, str(REPO / "src"))
    if ROOT != REPO:
        sys.path.insert(0, str(ROOT))
    _mod = importlib.import_module(f"scenes.{scene}")


def _frame(args):
    scene, i, out = args
    p = out / f"{i:05d}.png"
    if p.exists():
        return
    img = _mod.render(i / FPS)
    assert img.shape == (1080, 1920, 3) and img.dtype == np.uint8, img.shape
    Image.fromarray(img).save(p, compress_level=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--sheet", type=int)
    ap.add_argument("--at", type=float)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0s, t1s = scene_bounds(a.scene)
    n = int(round((t1s - t0s) * FPS))
    prev = ROOT / "build" / "preview"; prev.mkdir(parents=True, exist_ok=True)
    if a.at is not None or a.sheet:
        _init(a.scene)
        if a.at is not None:
            p = prev / f"{a.scene}_{a.at:.2f}.png"
            Image.fromarray(_mod.render(a.at)).save(p); print(p); return
        ts = [(k + 0.5) * (t1s - t0s) / a.sheet for k in range(a.sheet)]
        cols = 4; rows = math.ceil(len(ts) / cols)
        sheet = Image.new("RGB", (cols * 480, rows * 270 + rows * 0))
        for k, t in enumerate(ts):
            im = Image.fromarray(_mod.render(t)).resize((480, 270), Image.LANCZOS)
            sheet.paste(im, ((k % cols) * 480, (k // cols) * 270))
        p = prev / f"{a.scene}_sheet.png"; sheet.save(p); print(p, [round(t, 2) for t in ts]); return
    out = ROOT / "build" / "frames" / a.scene; out.mkdir(parents=True, exist_ok=True)
    if a.force:
        for f in out.glob("*.png"): f.unlink()
    st = time.time()
    with Pool(a.jobs, initializer=_init, initargs=(a.scene,)) as pool:
        for k, _ in enumerate(pool.imap_unordered(_frame, [(a.scene, i, out) for i in range(n)], chunksize=2)):
            if k % 24 == 0:
                print(f"\r{a.scene}: {k}/{n}  {time.time()-st:.0f}s", end="", flush=True)
    print(f"\r{a.scene}: {n}/{n} frames in {time.time()-st:.0f}s")
    if a.video:
        mix = ROOT / "build" / "audio" / "mix.wav"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(out / "%05d.png")]
        if mix.exists():
            cmd += ["-ss", str(t0s), "-t", str(t1s - t0s), "-i", str(mix)]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-shortest", str(prev / f"{a.scene}.mp4")]
        subprocess.run(cmd, check=True); print(prev / f"{a.scene}.mp4")


if __name__ == "__main__":
    main()
