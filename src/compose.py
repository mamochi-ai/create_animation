"""Final assembly: scene frames -> transitions -> subtitles -> H.264 + AAC.

  python3 src/compose.py            # -> output/hoshi_no_otoshimono.mp4
  python3 src/compose.py --nosubs   # clean version without burned-in subtitles
"""
import argparse
import os
import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(os.environ.get("FILM_DIR", REPO)).resolve()
sys.path.insert(0, str(REPO / "src"))
if ROOT != REPO:
    sys.path.insert(0, str(ROOT))  # film-local `scenes` and `lib` packages win
from common.core import FPS, H, TIMELINE, W, over, text_layer, smoothstep  # noqa: E402

# frames of cross-dissolve at a cut; a film can override via timeline.json "dissolves": [["s01","s02",12], ...]
DISSOLVES = ({(a, b): n for a, b, n in TIMELINE["dissolves"]} if "dissolves" in TIMELINE
             else {("s04", "s05"): 12, ("s06", "s07"): 18})
FADE_IN, FADE_OUT = 14, 20

SPEAKER_COLOR = {"haru": "#FFD27A", "kira": "#FFF3B0"}


def load_frames():
    seq = []
    for sc in TIMELINE["scenes"]:
        n = int(round((sc["end"] - sc["start"]) * FPS))
        d = ROOT / "build" / "frames" / sc["id"]
        files = [d / f"{i:05d}.png" for i in range(n)]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"{sc['id']}: {len(missing)} frames missing (first {missing[0]})")
        seq.append((sc["id"], files))
    return seq


def subtitles():
    meta = {m["id"]: m for m in json.loads((ROOT / "assets/voice/voices.json").read_text())}
    subs = []
    for ln in TIMELINE["lines"]:
        m = meta[ln["id"]]
        t0 = ln["start"] - 0.08
        t1 = max(ln["start"] + m["dur"] + 0.45, t0 + 1.3)
        subs.append((t0, t1, m["sub"], m["who"]))
    # never overlap: clip to next start
    for i in range(len(subs) - 1):
        if subs[i][1] > subs[i + 1][0] - 0.05:
            subs[i] = (subs[i][0], subs[i + 1][0] - 0.05, subs[i][2], subs[i][3])
    return subs


_sub_cache = {}


def sub_layer(text, who):
    key = (text, who)
    if key not in _sub_cache:
        lay = text_layer(text, (W / 2, H - 92), 50, "maru", "#FFFFFF", stroke=5, stroke_color="#0E1026")
        # soft shadow
        sh = text_layer(text, (W / 2 + 3, H - 88), 50, "maru", "#000000", stroke=7, stroke_color="#000000", alpha=0.35)
        _sub_cache[key] = (lay, sh)
    return _sub_cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nosubs", action="store_true")
    ap.add_argument("--out", default=str(REPO / "output" / (TIMELINE.get("output") or "hoshi_no_otoshimono.mp4")))
    a = ap.parse_args()

    seq = load_frames()
    subs = [] if a.nosubs else subtitles()
    audio = ROOT / "build" / "audio" / "mix.wav"
    total = sum(len(f) for _, f in seq)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio.exists():
        cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "256k"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-tune", "animation",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    gi = 0
    for si, (sid, files) in enumerate(seq):
        nxt = seq[si + 1] if si + 1 < len(seq) else None
        prv = seq[si - 1] if si > 0 else None
        nd_in = DISSOLVES.get((prv[0], sid), 0) if prv else 0
        nd_out = DISSOLVES.get((sid, nxt[0]), 0) if nxt else 0
        for i, f in enumerate(files):
            img = np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255
            # timing-preserving dissolve: hold the neighbour's edge frame across the cut
            if nd_out and i >= len(files) - nd_out // 2:
                b = np.asarray(Image.open(nxt[1][0]).convert("RGB")).astype(np.float32) / 255
                w = smoothstep(0, 1, (i - (len(files) - nd_out / 2) + 0.5) / nd_out)
                img = img * (1 - w) + b * w
            elif nd_in and i < nd_in // 2:
                a_ = np.asarray(Image.open(prv[1][-1]).convert("RGB")).astype(np.float32) / 255
                w = smoothstep(0, 1, (i + nd_in / 2 + 0.5) / nd_in)
                img = a_ * (1 - w) + img * w
            t = gi / FPS
            if gi < FADE_IN:
                img *= smoothstep(0, 1, gi / FADE_IN)
            if gi >= total - FADE_OUT - 2:
                img *= smoothstep(0, 1, (total - 2 - gi) / FADE_OUT)
            for (t0, t1, text, who) in subs:
                if t0 <= t < t1:
                    al = min(smoothstep(0, 0.15, t - t0), smoothstep(0, 0.15, t1 - t))
                    lay, sh = sub_layer(text, who)
                    rgba = np.concatenate([img, np.ones((H, W, 1), np.float32)], 2)
                    over(rgba, sh, al)
                    over(rgba, lay, al)
                    img = rgba[..., :3]
            ff.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
            gi += 1
            if gi % 48 == 0:
                print(f"\rcompose {gi}/{total}", end="", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"\n{out}  ({gi} frames = {gi / FPS:.2f}s)")


if __name__ == "__main__":
    main()
