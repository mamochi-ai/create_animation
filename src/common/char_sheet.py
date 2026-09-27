"""Model sheets for Haru / Kira / lantern.

    python3 src/common/char_sheet.py            # -> build/preview/model_sheet_haru.png, model_sheet_kira.png
    python3 src/common/char_sheet.py --quick    # also a quick single-row test sheet
"""
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cairo  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common.core import (add_glow, linear_gradient, new_surface, surface_rgba, finish,  # noqa: E402
                         draw_layer)
from common.characters import (draw_haru, draw_kira, draw_kira_glow, draw_lantern,  # noqa: E402
                               draw_lantern_glow, draw_sparkle)

OUT = ROOT / "build" / "preview"
OUT.mkdir(parents=True, exist_ok=True)


def night_bg(ctx, w, h):
    ctx.set_source(linear_gradient(0, 0, 0, h, [(0, "#0B1030"), (0.6, "#1B2350"), (1, "#2E3A78")]))
    ctx.paint()
    import random
    rnd = random.Random(3)
    for _ in range(int(w * h / 9000)):
        x, y = rnd.uniform(0, w), rnd.uniform(0, h)
        ctx.arc(x, y, rnd.uniform(0.4, 1.3), 0, math.tau)
        ctx.set_source_rgba(1, 1, 1, rnd.uniform(0.15, 0.6))
        ctx.fill()


def label(ctx, x, y, text, size=18):
    ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    ctx.set_source_rgba(0.85, 0.88, 1, 0.8)
    ext = ctx.text_extents(text)
    ctx.move_to(x - ext.width / 2, y)
    ctx.show_text(text)


def ground(ctx, x, y, w=160):
    g = cairo.RadialGradient(x, y, 0, x, y, w / 2)
    g.add_color_stop_rgba(0, 0, 0, 0.05, 0.45)
    g.add_color_stop_rgba(1, 0, 0, 0.05, 0)
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(1, 0.18)
    ctx.translate(-x, -y)
    ctx.set_source(g)
    ctx.arc(x, y, w / 2, 0, math.tau)
    ctx.fill()
    ctx.restore()


def sheet_haru(path):
    W, H = 3000, 3900
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(s)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    night_bg(ctx, W, H)
    label(ctx, W / 2, 50, "HARU — model sheet", 34)
    # row 1: views (scale 0.9)
    views = ["front", "three_quarter_right", "side_right", "back", "side_left", "three_quarter_left"]
    sc = 0.85
    y0 = 600
    for i, v in enumerate(views):
        x = 260 + i * 490
        ground(ctx, x, y0)
        draw_haru(ctx, x, y0, sc, view=v, t=0.3 + i)
        label(ctx, x, y0 + 40, v)
    # row 2: expressions (bust, bigger)
    exprs = ["neutral", "happy", "surprised", "determined", "sad", "gentle", "sleepy", "worried", "laugh"]
    y1 = 1330
    for i, e in enumerate(exprs):
        x = 170 + i * 330
        ctx.save()
        ctx.rectangle(x - 160, y1 - 580, 320, 640)
        ctx.clip()
        draw_haru(ctx, x, y1 + 520, 1.5, view="front" if i % 3 != 2 else "three_quarter_right", expr=e,
                  t=i, crop="bust")
        ctx.restore()
        label(ctx, x, y1 + 90, e)
    # row 3: arm presets
    presets = ["rest", "chin_on_hands", "cup_hands", "hold_lantern", "raise_lantern", "wave", "reach_up",
               "hands_to_mouth", "hips"]
    y2 = 2150
    for i, a in enumerate(presets):
        x = 170 + i * 330
        v = "three_quarter_right" if a in ("cup_hands", "hold_lantern") else "front"
        ground(ctx, x, y2)
        draw_haru(ctx, x, y2, 0.72, view=v, arms=a, arm_phase=0.2, lantern=True, lantern_lit=0.0,
                  expr="happy" if a == "wave" else ("surprised" if a == "hands_to_mouth" else "neutral"),
                  t=i)
        label(ctx, x, y2 + 40, a)
    # row 4: run cycle
    y3 = 2850
    for i in range(8):
        x = 190 + i * 370
        ground(ctx, x, y3)
        draw_haru(ctx, x, y3, 0.72, view="side_right", run_phase=i / 8, arms="run", expr="determined",
                  t=i * 0.05, lantern="back")
        label(ctx, x, y3 + 40, f"run {i}/8")
    # row 5: mouth strip + crouch + cup run + back wave
    y4 = 3700
    for i, vw in enumerate("aiueonx"):
        x = 140 + i * 250
        ctx.save()
        ctx.rectangle(x - 120, y4 - 480, 240, 520)
        ctx.clip()
        draw_haru(ctx, x, y4 + 700, 2.0, view="front", expr="happy" if vw == "x" else "neutral",
                  mouth_open=0.0 if vw in "nx" else 1.0, vowel=vw, crop="bust")
        ctx.restore()
        label(ctx, x, y4 + 30, f"'{vw}'")
    x = 1950
    ground(ctx, x, y4)
    draw_haru(ctx, x, y4, 0.72, view="three_quarter_right", crouch=1.0, expr="surprised", arms="rest")
    label(ctx, x, y4 + 40, "crouch")
    x = 2300
    ground(ctx, x, y4)
    draw_haru(ctx, x, y4, 0.72, view="side_right", run_phase=0.3, arms="cup_hands", lantern="back",
              held=lambda c, hx, hy, s: draw_kira(c, hx, hy - 20 * s, 0.4 * s, glow=0.6))
    label(ctx, x, y4 + 40, "run + cup")
    x = 2700
    ground(ctx, x, y4)
    draw_haru(ctx, x, y4, 0.72, view="back", arms="wave", head_pitch=0.6, arm_phase=0.3)
    label(ctx, x, y4 + 40, "back wave")
    s.write_to_png(str(path))


def sheet_kira(path):
    W, H = 2400, 1500
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(s)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    night_bg(ctx, W, H)
    label(ctx, W / 2, 50, "KIRA & LANTERN — model sheet", 34)
    exprs = ["neutral", "happy", "sad", "hurt", "surprised", "sleepy"]
    for i, e in enumerate(exprs):
        x, y = 200 + i * 330, 260
        draw_kira_glow(ctx, x, y, 1.6, 0.6)
        draw_kira(ctx, x, y, 1.6, expr=e, glow=0.9, t=i)
        label(ctx, x, y + 150, e)
    # glow levels with numpy bloom
    base = surface_rgba(s)
    for i, g in enumerate((0.1, 0.5, 1.0)):
        x, y = 260 + i * 420, 700
        ctx2 = ctx
        draw_kira_glow(ctx2, x, y, 1.5, g, 0.0)
        draw_kira(ctx2, x, y, 1.5, glow=g, expr="sad" if g < 0.3 else "neutral", t=1.0)
        label(ctx, x, y + 170, f"glow {g}")
    poses = [("rub_head", "hurt"), ("wave", "happy"), ("cheer", "happy"), ("droop", "sleepy")]
    for i, (p, e) in enumerate(poses):
        x, y = 1480 + i * 250, 700
        draw_kira_glow(ctx, x, y, 1.2, 0.6)
        draw_kira(ctx, x, y, 1.2, pose=p, expr=e, glow=0.8, t=0.1)
        label(ctx, x, y + 130, p)
    # mouth strip
    for i, v in enumerate("aiueonx"):
        x, y = 200 + i * 160, 1150
        draw_kira(ctx, x, y, 1.1, mouth_open=1.0 if v not in "nx" else 0, vowel=v, glow=0.9)
        label(ctx, x, y + 100, f"'{v}'")
    # lanterns
    for i, (lit, k) in enumerate(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))):
        x, y = 1450 + i * 300, 980
        draw_lantern_glow(ctx, x, y, 1.6, lit)
        draw_lantern(ctx, x, y, 1.6, lit=lit, kira=k, sway=0.08 * (i - 1))
        label(ctx, x, y + 360, ["unlit", "lit", "lit + kira"][i])
    s.flush()
    # bloom pass on the whole sheet for bright pixels
    img = surface_rgba(s)
    from common.core import bloom
    img = bloom(img, threshold=0.8, radius=24, strength=0.5)
    Image.fromarray(finish(img)).save(path)


def timing():
    s, ctx = new_surface()
    t0 = time.perf_counter()
    n = 30
    for i in range(n):
        draw_haru(ctx, 960, 900, 1.0, view="three_quarter_right", arms="cup_hands", t=i * 0.04,
                  mouth_open=0.5, vowel="a")
    t1 = time.perf_counter()
    for i in range(n):
        draw_haru(ctx, 960, 900, 1.0, view="side_right", run_phase=i / n, arms="run", lantern="back", t=i * 0.04)
    t2 = time.perf_counter()
    print(f"haru 3/4 cup: {(t1 - t0) / n * 1000:.1f} ms   run: {(t2 - t1) / n * 1000:.1f} ms")


if __name__ == "__main__":
    sheet_haru(OUT / "model_sheet_haru.png")
    sheet_kira(OUT / "model_sheet_kira.png")
    timing()
    print("wrote", OUT / "model_sheet_haru.png", OUT / "model_sheet_kira.png")
