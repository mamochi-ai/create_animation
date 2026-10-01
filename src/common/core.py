"""Shared foundation for every scene: canvas, easing, colour, glow, lip sync.

All drawing is done with pycairo on 1920x1080 ARGB surfaces; post effects
(glow / blur / grading) are done in numpy + Pillow.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
from functools import lru_cache

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = pathlib.Path(__file__).resolve().parents[2]
# FILM_DIR selects which film (story/, assets/voice/, build/) we are working on; default = first film at repo root
ROOT = pathlib.Path(os.environ.get("FILM_DIR", REPO)).resolve()
TIMELINE = json.loads((ROOT / "story" / "timeline.json").read_text())
W, H = TIMELINE["width"], TIMELINE["height"]
FPS = TIMELINE["fps"]
TAU = math.tau


# ---------------------------------------------------------------- math / easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def invlerp(a, b, x):
    return clamp((x - a) / (b - a)) if b != a else 0.0


def remap(x, a, b, c, d):
    """Map x from [a,b] to [c,d] (clamped)."""
    return lerp(c, d, invlerp(a, b, x))


def smoothstep(a, b, x):
    t = invlerp(a, b, x)
    return t * t * (3 - 2 * t)


def ease_in_out(t):
    t = clamp(t)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ease_out(t, p=3):
    t = clamp(t)
    return 1 - (1 - t) ** p


def ease_in(t, p=3):
    return clamp(t) ** p


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_elastic(t):
    t = clamp(t)
    if t in (0.0, 1.0):
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * TAU / 3) + 1


def spring(t, freq=3.0, damp=4.0):
    """Damped spring settling from 0 to 1 (overshoots)."""
    if t <= 0:
        return 0.0
    return 1 - math.exp(-damp * t) * math.cos(freq * TAU * t)


def noise1(x, seed=0):
    """Smooth deterministic 1D value noise in [-1, 1]."""
    i = math.floor(x)
    f = x - i
    def h(n):
        n = (n * 374761393 + seed * 668265263) & 0xFFFFFFFF
        n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
        return ((n ^ (n >> 16)) & 0xFFFF) / 32767.5 - 1.0
    u = f * f * (3 - 2 * f)
    return lerp(h(i), h(i + 1), u)


def fbm(x, seed=0, octaves=3):
    return sum(noise1(x * 2 ** o, seed + o * 17) / 2 ** o for o in range(octaves)) / 1.75


# ---------------------------------------------------------------- colour
@lru_cache(maxsize=None)
def rgb(hexstr: str):
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def mix(c1, c2, t):
    c1 = rgb(c1) if isinstance(c1, str) else c1
    c2 = rgb(c2) if isinstance(c2, str) else c2
    return tuple(lerp(a, b, t) for a, b in zip(c1, c2))


def set_color(ctx, c, a=1.0):
    c = rgb(c) if isinstance(c, str) else c
    ctx.set_source_rgba(c[0], c[1], c[2], a)


def linear_gradient(x0, y0, x1, y1, stops):
    """stops: [(offset, color, alpha?), ...]"""
    g = cairo.LinearGradient(x0, y0, x1, y1)
    for s in stops:
        c = rgb(s[1]) if isinstance(s[1], str) else s[1]
        g.add_color_stop_rgba(s[0], *c, s[2] if len(s) > 2 else 1.0)
    return g


def radial_gradient(cx, cy, r0, r1, stops, fx=None, fy=None):
    g = cairo.RadialGradient(cx if fx is None else fx, cy if fy is None else fy, r0, cx, cy, r1)
    for s in stops:
        c = rgb(s[1]) if isinstance(s[1], str) else s[1]
        g.add_color_stop_rgba(s[0], *c, s[2] if len(s) > 2 else 1.0)
    return g


# ---------------------------------------------------------------- surfaces
def new_surface(w=W, h=H):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(s)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    return s, ctx


def surface_rgba(s: cairo.ImageSurface) -> np.ndarray:
    """Premultiplied float32 RGBA array (H, W, 4) in 0..1."""
    s.flush()
    a = np.ndarray((s.get_height(), s.get_width(), 4), np.uint8, s.get_data(), strides=(s.get_stride(), 4, 1))
    return a[..., [2, 1, 0, 3]].astype(np.float32) / 255.0


def rgba_surface(a: np.ndarray) -> cairo.ImageSurface:
    """Premultiplied float RGBA array -> cairo surface."""
    h, w = a.shape[:2]
    b = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)[..., [2, 1, 0, 3]].copy()
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    stride = s.get_stride()
    buf = np.ndarray((h, w, 4), np.uint8, s.get_data(), strides=(stride, 4, 1))
    buf[:] = b
    s.mark_dirty()
    return s


def blur(a: np.ndarray, radius: float, downscale: int | None = None) -> np.ndarray:
    """Fast gaussian blur of an (H,W,C) float array; large radii are blurred at low res."""
    if radius <= 0.3:
        return a
    h, w, c = a.shape
    ds = downscale or max(1, min(8, int(radius / 6)))
    small = (w // ds, h // ds)
    out = np.empty_like(a)
    for ch in range(c):
        im = Image.fromarray(a[..., ch].astype(np.float32), mode="F")
        if ds > 1:
            im = im.resize(small, Image.BILINEAR)
        b = Image.fromarray(gaussian_filter(np.asarray(im), radius / ds, mode="nearest"), mode="F")
        if ds > 1:
            b = b.resize((w, h), Image.BILINEAR)
        out[..., ch] = np.asarray(b)
    return out


def over(dst: np.ndarray, src: np.ndarray, opacity=1.0):
    """Premultiplied 'over' compositing of RGBA arrays (in place on dst)."""
    sa = src[..., 3:4] * opacity
    dst[...] = src * opacity + dst * (1 - sa)
    return dst


def add_glow(dst: np.ndarray, src: np.ndarray, radius=30.0, strength=1.0, tint=None):
    """Additive bloom of src's RGB onto dst (screen-like, clamped)."""
    g = blur(src[..., :3], radius)
    if tint is not None:
        g = g * np.array(rgb(tint) if isinstance(tint, str) else tint, np.float32)
    dst[..., :3] = 1 - (1 - dst[..., :3]) * (1 - np.clip(g * strength, 0, 1))
    return dst


def bloom(img: np.ndarray, threshold=0.75, radius=40.0, strength=0.6):
    """Global bloom from bright pixels of an RGB(A) frame."""
    rgbp = img[..., :3]
    lum = rgbp.max(axis=2, keepdims=True)
    bright = rgbp * np.clip((lum - threshold) / (1 - threshold + 1e-6), 0, 1)
    g = blur(bright, radius) * strength + blur(bright, radius * 0.3) * strength * 0.6
    img[..., :3] = 1 - (1 - rgbp) * (1 - np.clip(g, 0, 1))
    return img


def vignette(img: np.ndarray, strength=0.35, color=(0.02, 0.02, 0.08)):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    v = np.clip(d - 0.55, 0, 1)[..., None] ** 1.5 * strength
    img[..., :3] = img[..., :3] * (1 - v) + np.array(color, np.float32) * v
    return img


_GRAIN = None


def grain(img: np.ndarray, frame: int, amount=0.018):
    global _GRAIN
    if _GRAIN is None:
        rng = np.random.default_rng(7)
        _GRAIN = rng.normal(0, 1, (8, H, W)).astype(np.float32)
    img[..., :3] += _GRAIN[frame % 8][..., None] * amount
    return img


def finish(img: np.ndarray) -> np.ndarray:
    """RGBA/RGB float -> uint8 RGB (H, W, 3)."""
    return (np.clip(img[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)


def draw_layer(fn, *args, **kw) -> np.ndarray:
    """Run fn(ctx, *args) on a fresh transparent surface and return its RGBA array."""
    s, ctx = new_surface()
    fn(ctx, *args, **kw)
    return surface_rgba(s)


# ---------------------------------------------------------------- dialogue / lip sync
@lru_cache(maxsize=1)
def _voices():
    meta = json.loads((ROOT / "assets" / "voice" / "voices.json").read_text())
    starts = {l["id"]: l["start"] for l in TIMELINE["lines"]}
    for m in meta:
        m["start"] = starts[m["id"]]
    return meta


def line(line_id):
    return next(m for m in _voices() if m["id"] == line_id)


_OPEN = {"a": 1.0, "o": 0.8, "e": 0.65, "u": 0.45, "i": 0.4, "n": 0.12, "x": 0.0, None: 0.1}


def mouth(who: str, t_global: float):
    """Lip-sync state of a character at global time t.

    Returns (openness 0..1, vowel in 'aiueonx', talking: bool).
    Openness is smoothed across phoneme boundaries so mouths ease, not snap.
    """
    for m in _voices():
        if m["who"] != who:
            continue
        lt = t_global - m["start"]
        if -0.05 <= lt <= m["dur"] + 0.05:
            best, acc, wsum = "x", 0.0, 0.0
            for p in m["phones"]:
                c = 0.5 * (p["t0"] + p["t1"])
                half = max(0.035, 0.5 * (p["t1"] - p["t0"]))
                wgt = max(0.0, 1 - abs(lt - c) / (half + 0.04))
                if wgt > 0:
                    acc += _OPEN.get(p["v"], 0.1) * wgt
                    wsum += wgt
                if p["t0"] <= lt < p["t1"] and p["v"]:
                    best = p["v"]
            o = acc / wsum if wsum else 0.0
            return clamp(o), best, True
    return 0.0, "x", False


def talking(who: str, t_global: float) -> bool:
    return mouth(who, t_global)[2]


def blink(t: float, seed=0, period=3.4) -> float:
    """Eyelid closure 0..1 at time t (natural irregular blinking)."""
    k = math.floor(t / period)
    off = (noise1(k * 1.7, seed) * 0.5 + 0.5) * (period - 0.4)
    lt = t - k * period - off
    if 0 <= lt < 0.16:
        return math.sin(lt / 0.16 * math.pi)
    return 0.0


def scene_bounds(scene_id):
    s = next(s for s in TIMELINE["scenes"] if s["id"] == scene_id)
    return s["start"], s["end"]


# ---------------------------------------------------------------- text
FONTS = {
    "maru": REPO / "assets" / "fonts" / "ZenMaruGothic-Bold.ttf",
    "maru_medium": REPO / "assets" / "fonts" / "ZenMaruGothic-Medium.ttf",
    "klee": REPO / "assets" / "fonts" / "KleeOne-SemiBold.ttf",  # hand-written feel, good for titles
}


@lru_cache(maxsize=64)
def _font(name, size):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONTS[name]), int(size))


def text_layer(text, xy, size=64, font="maru", color="#FFFFFF", anchor="mm",
               stroke=0, stroke_color="#101428", spacing=0, alpha=1.0) -> np.ndarray:
    """Render Japanese text to a premultiplied RGBA float layer (H,W,4).

    Composite with core.over(img, layer) (and optionally add_glow for a luminous title).
    `spacing` adds extra pixels between characters (letter-spacing)."""
    from PIL import ImageDraw
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = _font(font, size)
    c = tuple(int(v * 255) for v in rgb(color))
    sc = tuple(int(v * 255) for v in rgb(stroke_color))
    if spacing:
        widths = [d.textlength(ch, font=f) + spacing for ch in text]
        total = sum(widths) - spacing
        x = xy[0] - total * {"l": 0, "m": 0.5, "r": 1}[anchor[0]]
        for ch, wdt in zip(text, widths):
            d.text((x, xy[1]), ch, font=f, fill=c, anchor="l" + anchor[1],
                   stroke_width=stroke, stroke_fill=sc)
            x += wdt
    else:
        d.text(xy, text, font=f, fill=c, anchor=anchor, stroke_width=stroke, stroke_fill=sc)
    a = np.asarray(im).astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a * alpha
