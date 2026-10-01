"""Model sheets for 「あめふりロボ」 characters.

    FILM_DIR=films/ame_robo python3 films/ame_robo/lib/char_sheet.py
-> films/ame_robo/build/preview/model_sheet_mina.png, model_sheet_robo.png, model_sheet_kitten.png
"""
from __future__ import annotations

import math
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
FILM = HERE.parent
REPO = FILM.parents[1]
os.environ.setdefault("FILM_DIR", str(FILM))
for p in (str(FILM), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cairo  # noqa: E402

from lib.characters import (ROBO_FACES, draw_drops_on, draw_kitten, draw_mina, draw_rim_drips,  # noqa: E402
                            draw_robo)

OUT = FILM / "build" / "preview"


def _bg(ctx, w, h, seed=3):
    g = cairo.LinearGradient(0, 0, 0, h)
    g.add_color_stop_rgb(0, *(c / 255 for c in (0x8C, 0x9B, 0xAA)))
    g.add_color_stop_rgb(1, *(c / 255 for c in (0xB7, 0xC2, 0xCC)))
    ctx.set_source(g)
    ctx.paint()
    # rain streaks
    import random
    rnd = random.Random(seed)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for _ in range(int(w * h / 9000)):
        x, y = rnd.uniform(0, w), rnd.uniform(0, h)
        L = rnd.uniform(18, 46)
        ctx.move_to(x, y)
        ctx.line_to(x - L * 0.12, y + L)
        ctx.set_source_rgba(0.85, 0.89, 0.93, rnd.uniform(0.18, 0.45))
        ctx.set_line_width(rnd.uniform(1.0, 2.0))
        ctx.stroke()


def _label(ctx, x, y, text, size=17, col=(0.16, 0.2, 0.27)):
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    ext = ctx.text_extents(text)
    ctx.move_to(x - ext.width / 2, y)
    ctx.set_source_rgba(*col, 0.9)
    ctx.show_text(text)


def _shadow(ctx, x, y, rx, ry=None):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(rx, ry or rx * 0.16)
    ctx.arc(0, 0, 1, 0, math.tau)
    ctx.restore()
    ctx.set_source_rgba(0.15, 0.2, 0.28, 0.25)
    ctx.fill()


def _title(ctx, w, text):
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(34)
    ext = ctx.text_extents(text)
    ctx.move_to(w / 2 - ext.width / 2, 48)
    ctx.set_source_rgba(0.12, 0.16, 0.24, 0.9)
    ctx.show_text(text)


def sheet_mina():
    W, H = 3000, 3150
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c = cairo.Context(s)
    _bg(c, W, H)
    _title(c, W, "MINA — model sheet  (scale 1 ≈ 480 px)")
    views = ["front", "three_quarter_right", "side_right", "back", "side_left", "three_quarter_left"]
    for i, v in enumerate(views):
        x = 260 + i * 496
        _shadow(c, x, 640, 70)
        draw_mina(c, x, 640, 1.0, view=v, t=0.4)
        _label(c, x, 680, v)
    # expressions (bust)
    exprs = ["neutral", "happy", "laugh", "sad", "surprised", "worried", "pout", "amazed", "gentle"]
    for i, e in enumerate(exprs):
        x = 170 + i * 330
        c.save()
        c.rectangle(x - 160, 760, 320, 470)
        c.clip()
        draw_mina(c, x, 1960, 2.6, expr=e, crop="bust", t=0.4, blink=0.0)
        c.restore()
        _label(c, x, 1260, e)
    # lip sync
    for i, v in enumerate("aiueon"):
        x = 200 + i * 290
        c.save()
        c.rectangle(x - 140, 1320, 280, 400)
        c.clip()
        draw_mina(c, x, 2380, 2.4, view="three_quarter_right" if i % 2 else "front", expr="happy",
                  mouth_open=0.9, vowel=v, crop="bust", t=0.4)
        c.restore()
        _label(c, x, 1750, "vowel " + v)
    # arms
    arms = ["rest", "hands_clasped", "hands_to_mouth", "point", "wave", "arms_up", "shrug"]
    for i, a in enumerate(arms):
        x = 230 + i * 300
        ex = {"hands_to_mouth": "laugh", "arms_up": "happy", "shrug": "worried", "point": "surprised"}.get(a, "neutral")
        _shadow(c, x, 2490, 60)
        draw_mina(c, x, 2490, 0.82, arms=a, expr=ex, t=0.6, view="front" if a != "point" else "three_quarter_right")
        _label(c, x, 2525, a)
    # s05 key pose: randoseru held over Robo's head
    from lib.characters import draw_robo
    sc = 0.85
    rx, mx = 2700, 2700 - 175 * sc
    ra = draw_robo(c, rx, 2490, sc, view="three_quarter_left", face="warm", wet=0.9, t=0.5, screen_glow=1.0)
    ht = ra["head_top"]
    _shadow(c, mx, 2490, 60)
    draw_mina(c, mx, 2490, sc, view="three_quarter_right", arms="hold_randoseru_overhead", expr="gentle",
              wet=0.6, t=0.5, hold_target=(ht[0], ht[1] - 60 * sc))
    _label(c, 2560, 2525, "s05: hold_randoseru_overhead, hold_target over Robo")
    row = [({"view": "front", "arms": "hold_randoseru_overhead", "expr": "pout"}, "hold (default, pout)"),
           ({"view": "three_quarter_right", "arms": "hold_randoseru_overhead", "hold_flap": 1.0, "expr": "happy"},
            "hold, hold_flap=1"),
           ({"crouch": 0.8, "expr": "gentle", "arms": "hands_clasped"}, "crouch .8"),
           ({"wet": 1.0, "expr": "sad", "t": 1.3}, "wet 1.0"),
           ({"view": "back", "arms": "wave", "wind": 1.5, "t": 0.9}, "back wave, wind 1.5"),
           ({"view": "three_quarter_left", "arms": "arms_up", "expr": "laugh", "bounce": 0.15}, "cheer, bounce"),
           ({"view": "side_right", "arms": "point", "expr": "happy", "randoseru": None}, "point, no randoseru")]
    for i, (kw, lab) in enumerate(row):
        x = 230 + i * 400
        _shadow(c, x, 3030, 60)
        kw.setdefault("t", 0.5)
        draw_mina(c, x, 3030, 0.82, **kw)
        _label(c, x, 3065, lab, 16)
    # timing
    s2 = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1920, 1080)
    c2 = cairo.Context(s2)
    t0 = time.time()
    for k in range(10):
        draw_mina(c2, 960, 1000, 1.8, view="three_quarter_right", expr="happy", wet=0.5, t=k * 0.1)
    dt = (time.time() - t0) * 100
    _label(c, 200, H - 20, f"draw_mina avg {dt:.1f} ms", 15)
    OUT.mkdir(parents=True, exist_ok=True)
    s.write_to_png(str(OUT / "model_sheet_mina.png"))
    return dt


def sheet_robo():
    W, H = 3000, 4300
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c = cairo.Context(s)
    _bg(c, W, H, 5)
    _title(c, W, "ROBO No.7 — model sheet  (scale 1 ≈ 410 px)")
    views = ["front", "three_quarter_right", "side_right", "side_left", "three_quarter_left"]
    arms = ["rest", "wave", "point", "hold_handle", "salute"]
    for i, v in enumerate(views):
        x = 300 + i * 600
        _shadow(c, x, 560, 90)
        draw_robo(c, x, 560, 1.0, view=v, arms=arms[i], t=0.4, tread_phase=i * 0.3)
        _label(c, x, 600, f"{v} / {arms[i]}")
    for i, f in enumerate(ROBO_FACES):
        x = 160 + i * 298
        draw_robo(c, x, 1150, 0.7, face=f, t=0.9, crop=None) if False else None
        draw_robo(c, x, 1150, 0.7, face=f, t=0.9, speak=0.0)
        _label(c, x, 1185, f)
    for i, sp in enumerate((0.2, 0.5, 0.8, 1.0)):
        x = 260 + i * 330
        draw_robo(c, x, 1600, 0.7, face="dots" if i < 3 else "warm", speak=sp, t=0.3 + i * 0.13)
        _label(c, x, 1635, f"speak {sp}")
    for i, (kw, lab) in enumerate((({"wet": 1.0, "face": "dots"}, "wet 1"), ({"sparks": 1.0, "face": "glitch", "wet": 0.7},
                                   "sparks + glitch"), ({"arms": "flail", "face": "surprised", "t": 0.31}, "flail"))):
        x = 1680 + i * 450
        draw_robo(c, x, 1600, 0.7, t=kw.pop("t", 1.1), **kw)
        _label(c, x, 1635, lab)
    # mini umbrella stages
    for i, p in enumerate((0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 1.0)):
        x = 200 + i * 370
        draw_robo(c, x, 2150, 0.75, mini_umbrella=p, face="question" if p == 1.0 else "dots", t=1.0)
        _label(c, x, 2185, f"mini_umbrella {p}")
    # canopy stages
    stages = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)
    for i, p in enumerate(stages[:4]):
        x = 260 + i * 720
        draw_robo(c, x, 2950, 0.62, canopy=p, view="three_quarter_right", t=0.5, face="scan" if p < 1 else "happy")
        _label(c, x, 2990, f"canopy {p}")
    for i, p in enumerate(stages[4:]):
        x = 230 + i * 960
        y = 4180
        sc = 0.62
        mx = x + 430 * sc
        if p == 1.0:
            draw_mina(c, mx, y, sc * 0.98 * 1.0, view="three_quarter_left", expr="amazed", t=0.5)
        a = draw_robo(c, x, y, sc, canopy=p, view="three_quarter_right", t=0.5, face="happy" if p == 1 else "scan",
                      wet=1.0 if p == 1 else 0.0, sparks=0.6 if p == 1.0 else 0.0)
        if p < 1.0:
            draw_mina(c, mx, y, sc * 0.98, view="three_quarter_left", expr="surprised", t=0.5)
        if p == 1.0 and "canopy_rim_points" in a:
            draw_rim_drips(c, a["canopy_rim_points"], 0.7, sc, amount=1.0)
            draw_drops_on(c, a["drop_points"], 0.7, sc)
        _label(c, x + 200, y + 40, f"canopy {p}" + ("  + Mina under it, Robo outside (wet 1, sparks .6)" if p == 1 else ""))
    s2 = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1920, 1080)
    c = cairo.Context(s2)
    t0 = time.time()
    for k in range(10):
        draw_robo(c, 500, 1050, 1.0, view="three_quarter_right", face="glitch", wet=0.5, sparks=0.5, t=k * 0.1,
                  canopy=1.0)
    dt = (time.time() - t0) * 100
    c = cairo.Context(s)
    _label(c, 200, H - 20, f"draw_robo (with canopy) avg {dt:.1f} ms", 15)
    s.write_to_png(str(OUT / "model_sheet_robo.png"))
    return dt


def sheet_kitten():
    W, H = 1800, 700
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c = cairo.Context(s)
    _bg(c, W, H, 9)
    _title(c, W, "KONEKO — calico kitten  (scale 1 ≈ 70 px)")
    items = [("sit", "neutral", {}), ("sit", "happy", {}), ("sit", "mew", {}), ("stand", "neutral", {}),
             ("curl", "neutral", {}), ("nuzzle", "happy", {}), ("sit", "neutral", {"wet": 1.0}),
             ("sit", "neutral", {"facing": -1})]
    for i, (pose, e, kw) in enumerate(items):
        x = 130 + i * 215
        _shadow(c, x, 320, 50)
        draw_kitten(c, x, 320, 2.6, pose=pose, expr=e, t=0.4, **kw)
        _label(c, x, 355, f"{pose}/{e}" + (" wet" if kw.get("wet") else "") + (" left" if kw.get("facing") else ""))
    # scale reference with Mina and Robo
    draw_mina(c, 700, 680, 0.55, view="three_quarter_right", expr="gentle", crouch=0.0)
    draw_kitten(c, 790, 680, 0.55, pose="sit", t=0.4)
    draw_robo(c, 1000, 680, 0.55, view="three_quarter_left", face="warm")
    draw_kitten(c, 930, 680, 0.55, pose="nuzzle", t=0.4, facing=1)
    _label(c, 850, 400 + 290, "scale check (all 0.55)", 14)
    s2 = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1920, 1080)
    c2 = cairo.Context(s2)
    t0 = time.time()
    for k in range(20):
        draw_kitten(c2, 500, 500, 3.0, pose="sit", t=k * 0.1)
    dt = (time.time() - t0) * 50
    _label(c, 1500, H - 20, f"draw_kitten avg {dt:.1f} ms", 15)
    s.write_to_png(str(OUT / "model_sheet_kitten.png"))
    return dt


if __name__ == "__main__":
    which = sys.argv[1:] or ["mina", "robo", "kitten"]
    for w in which:
        print(w, "%.1f ms" % {"mina": sheet_mina, "robo": sheet_robo, "kitten": sheet_kitten}[w]())
