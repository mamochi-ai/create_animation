"""world.py — procedural environment art for 「ほしのおとしもの」.

Everything is drawn with pycairo in 1920x1080 pixel space (y down) and is fully
deterministic (seeded).  Functions draw onto a cairo context you pass in, so a
scene composes the frame as a stack of calls (sky -> stars -> town -> chars ->
foreground).  Heavy static shapes are baked once into *coded masks* (A8 cairo
surfaces, see `_Coded`) and re-coloured every frame from `sky_colors(phase)`,
so the whole environment can animate continuously from night (phase 0) through
pre-dawn (0.5) to dawn (1) at no extra cost.

Glow recipe (pattern used by the previews / recommended for scenes):

    s, ctx = core.new_surface()
    world.draw_sky(ctx, phase, t) ; world.draw_stars(ctx, phase, t) ; ...
    img = core.surface_rgba(s)
    glow = core.draw_layer(world.draw_fireflies, t, 30, (0, 500, 1920, 1080))
    core.over(img, glow); core.add_glow(img, glow, radius=18, strength=1.4)
    core.bloom(img, threshold=0.7, radius=36, strength=0.55)

See the bottom of each section for the public functions; `world_sheet.py`
renders reference stills of every environment.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cairo
import numpy as np
from scipy.ndimage import gaussian_filter, zoom as nd_zoom

from .core import (W, H, TAU, clamp, lerp, smoothstep, noise1, rgb, mix,
                   radial_gradient, linear_gradient)

HORIZON_Y = 660.0      # screen y of the sky horizon at cam_y = 0
SKY_PAR = 0.55         # parallax factor of the sky/stars for camera tilts
WINDOW = rgb("#FFB85C")
WINDOW_HOT = rgb("#FFD9A0")
WINDOW_GLOW = rgb("#FF9A3D")
TORII = rgb("#D2452F")


# =========================================================================== utils
def _c(c):
    return rgb(c) if isinstance(c, str) else tuple(c)


def _src(ctx, s, alpha=1.0):
    """Set a cairo source from a colour / (colour, alpha) / Pattern."""
    if isinstance(s, cairo.Pattern):
        ctx.set_source(s)
        return
    if isinstance(s, tuple) and len(s) == 2 and not isinstance(s[0], (int, float)):
        c, a = s
        c = _c(c)
        ctx.set_source_rgba(c[0], c[1], c[2], a * alpha)
        return
    c = _c(s)
    ctx.set_source_rgba(c[0], c[1], c[2], (c[3] if len(c) > 3 else 1.0) * alpha)


def _a8(arr):
    """float (h,w) 0..1 -> cairo A8 surface."""
    arr = np.asarray(arr)
    h, w = arr.shape
    s = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    st = s.get_stride()
    buf = np.ndarray((h, st), np.uint8, s.get_data())
    if arr.dtype == np.uint8:
        buf[:, :w] = arr
    else:
        buf[:, :w] = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    s.mark_dirty()
    return s


def _argb(rgba):
    """premultiplied float (h,w,4) RGBA -> cairo ARGB32 surface."""
    h, w = rgba.shape[:2]
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    st = s.get_stride()
    buf = np.ndarray((h, st // 4, 4), np.uint8, s.get_data())
    buf[:, :w] = (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)[..., [2, 1, 0, 3]]
    s.mark_dirty()
    return s


def _shift(a, dx, dy):
    """Shift a 2D array by integer (dx, dy) with zero fill: out[y, x] = a[y-dy, x-dx]."""
    out = np.zeros_like(a)
    h, w = a.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def _noise2(w, h, cell, rng, periodic=False, order=3):
    """Smooth value noise (h, w) in 0..1. `cell` may be (cx, cy) for anisotropy."""
    cx, cy = (cell, cell) if np.isscalar(cell) else cell
    gw, gh = max(2, int(math.ceil(w / cx)) + 1), max(2, int(math.ceil(h / cy)) + 1)
    if periodic:
        gw = max(2, int(round(w / cx)))
    g = rng.random((gh, gw)).astype(np.float32)
    mode = "grid-wrap" if periodic else "nearest"
    z = nd_zoom(g, (cy, (w / gw) if periodic else cx), order=order, mode=mode, grid_mode=True)
    return np.clip(z[:h, :w], 0, 1)


def _fbm2(w, h, cell, rng, octaves=4, periodic=False):
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    cx, cy = (cell, cell) if np.isscalar(cell) else cell
    for o in range(octaves):
        acc += amp * _noise2(w, h, (max(2, cx / 2 ** o), max(2, cy / 2 ** o)), rng, periodic)
        tot += amp
        amp *= 0.5
    return acc / tot


class _Coded:
    """A baked sprite made of alpha masks.

    Draw opaque shapes with `fill(r, g, b)`: `a` (coverage) becomes the body
    mask and the r/g/b values become three extra masks (occlusion-correct,
    because later shapes overwrite earlier ones).  After `bake()` the sprite
    is re-coloured at paint time: body colour through `a`, then `r`, `g`, `b`
    and an automatic top-edge `rim` mask.
    """

    def __init__(self, w, h, x0=0.0, y0=0.0):
        self.w, self.h, self.x0, self.y0 = int(w), int(h), x0, y0
        self.surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surf)
        self.ctx.translate(-x0, -y0)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.m = {}

    def set(self, r=0.0, g=0.0, b=0.0, a=1.0):
        self.ctx.set_source_rgba(r, g, b, a)

    def fill(self, r=0.0, g=0.0, b=0.0, a=1.0, preserve=False):
        self.ctx.set_source_rgba(r, g, b, a)
        self.ctx.fill_preserve() if preserve else self.ctx.fill()

    def stroke(self, width, r=0.0, g=0.0, b=0.0, a=1.0):
        self.ctx.set_line_width(width)
        self.ctx.set_source_rgba(r, g, b, a)
        self.ctx.stroke()

    def bake(self, rim=(0, 3, 1.0), rim2=None, soft=0.0):
        self.surf.flush()
        st = self.surf.get_stride()
        arr = np.ndarray((self.h, st // 4, 4), np.uint8, self.surf.get_data())[:, :self.w].copy()
        A = arr[..., 3]
        if soft > 0:
            arr = np.stack([gaussian_filter(arr[..., i].astype(np.float32), soft) for i in range(4)], -1)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            A = arr[..., 3]
        self.m["a"] = _a8(A)
        for key, idx in (("r", 2), ("g", 1), ("b", 0)):
            ch = arr[..., idx]
            self.m[key] = _a8(ch) if ch.max() > 2 else None
        af = A.astype(np.float32) / 255.0
        for name, spec in (("rim", rim), ("rim2", rim2)):
            if spec is None:
                self.m[name] = None
                continue
            dx, dy, sft = spec
            r = af * (1 - _shift(af, dx, dy))
            if sft > 0:
                r = gaussian_filter(r, sft) * af
            self.m[name] = _a8(np.clip(r * 1.4, 0, 1))
        self.surf = self.ctx = None
        return self

    def paint(self, ctx, cols, ox=0.0, oy=0.0, alpha=1.0):
        """cols: {'a'|'r'|'g'|'b'|'rim'|'rim2': colour | (colour, alpha) | Pattern}."""
        x, y = self.x0 + ox, self.y0 + oy
        for key in ("a", "r", "g", "b", "rim", "rim2"):
            s = cols.get(key)
            m = self.m.get(key)
            if s is None or m is None:
                continue
            _src(ctx, s, alpha)
            if alpha < 1.0 and isinstance(s, cairo.Pattern):
                ctx.push_group()
                ctx.set_source(s)
                ctx.mask_surface(m, x, y)
                ctx.pop_group_to_source()
                ctx.paint_with_alpha(alpha)
            else:
                ctx.mask_surface(m, x, y)


def _vgrad(y0, y1, c0, c1, a0=1.0, a1=1.0):
    return linear_gradient(0, y0, 0, y1, [(0, _c(c0), a0), (1, _c(c1), a1)])


def _lighten(c, k):
    return mix(_c(c), (1, 1, 1), k)


def _darken(c, k):
    return mix(_c(c), (0, 0, 0), k)


# =========================================================================== palette
_PAL = [
    (0.0, dict(
        top="#050920", upper="#0B1030", mid="#1B2350", low="#29336F", horizon="#34407E",
        hglow="#50559C", hglow_a=0.38,
        far="#2A356B", hill="#192050", mids="#121840", near="#0A0E28", fog="#3A4A8E",
        rim="#7486DA", roof="#1F2852", roof_near="#1A2148",
        cloud="#1B2454", cloud_lit="#8E9CD8", cloud_under="#27316C",
        grass="#133049", grass_tip="#4C7FA2", leaf="#10213D", leaf_mid="#1B3358", leaf_lit="#4666A6",
        ground="#0B152E", stone="#3C4776", stone_lit="#8E9ED2", wood="#2B2236",
        torii="#B03A3A", bark="#171830", sun_col="#8090D0",
        stars=1.0, milky=1.0, windows=1.0, moon=1.0, sun=0.0, mist=0.28)),
    (0.5, dict(
        top="#0E1446", upper="#1D2465", mid="#3D4C8F", low="#7C6FA8", horizon="#E7A6A0",
        hglow="#F2A89E", hglow_a=0.42,
        far="#5E5B95", hill="#2C2A5E", mids="#22214E", near="#141434", fog="#8F7CB2",
        rim="#DDA8CC", roof="#30316A", roof_near="#27285C",
        cloud="#4A4580", cloud_lit="#F4B5B0", cloud_under="#D092A8",
        grass="#1D3A55", grass_tip="#7088AA", leaf="#1A2847", leaf_mid="#2A3862", leaf_lit="#8278B4",
        ground="#141A3A", stone="#595A88", stone_lit="#BBA9D2", wood="#3B2C44",
        torii="#BE4240", bark="#221F38", sun_col="#FFB7A0",
        stars=0.72, milky=0.22, windows=0.88, moon=0.8, sun=0.35, mist=0.5)),
    (1.0, dict(
        top="#6F9BD4", upper="#93B6E0", mid="#C9D3E5", low="#F7E3C4", horizon="#F6C58F",
        hglow="#FFE4B4", hglow_a=0.6,
        far="#A99ABA", hill="#5C4F74", mids="#483F63", near="#2C2842", fog="#EDC8BD",
        rim="#FFCC8C", roof="#5C5682", roof_near="#453F68",
        cloud="#E6BBC0", cloud_lit="#FFF4DE", cloud_under="#FFC894",
        grass="#2D584E", grass_tip="#AFC48C", leaf="#243D38", leaf_mid="#3C5A4C", leaf_lit="#D2BE8A",
        ground="#2D3A40", stone="#8A8496", stone_lit="#F4D4B0", wood="#5A4038",
        torii="#D2452F", bark="#3A2E36", sun_col="#FFF0C8",
        stars=0.0, milky=0.0, windows=0.28, moon=0.45, sun=1.0, mist=0.8)),
]


@lru_cache(maxsize=512)
def _sky_colors_q(q):
    phase = q / 2000.0
    for i in range(len(_PAL) - 1):
        p0, d0 = _PAL[i]
        p1, d1 = _PAL[i + 1]
        if phase <= p1:
            break
    u = (phase - p0) / (p1 - p0)
    out = {}
    for k, v0 in d0.items():
        v1 = d1[k]
        out[k] = mix(rgb(v0), rgb(v1), u) if isinstance(v0, str) else lerp(v0, v1, u)
    return out


def sky_colors(phase):
    """Palette for a time-of-day phase (0 night, 0.5 pre-dawn, 1 dawn).

    Returns a dict of RGB tuples (sky stops top/upper/mid/low/horizon, hglow,
    silhouettes far/hill/mids/near, fog, rim, roof, clouds, foliage, stone,
    torii ...) and scalars stars/milky/windows/moon/sun/mist (0..1).
    """
    return dict(_sky_colors_q(int(round(clamp(float(phase)) * 2000))))


# =========================================================================== sky
@lru_cache(maxsize=4)
def _milky_way(seed=3):
    """Low-res premultiplied RGBA of the Milky Way band in sky space.

    Sky space: x -200..2120, y -1150..850 (screen = sky - cam_y*SKY_PAR).
    """
    ds = 4
    X0, Y0, WW, HH = -200, -1150, 2320, 2000
    w, h = WW // ds, HH // ds
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    X = X0 + xx * ds
    Y = Y0 + yy * ds
    ang = 0.40
    cx, cy = 900.0, -300.0
    ux, uy = math.cos(ang), math.sin(ang)
    s = (X - cx) * ux + (Y - cy) * uy
    d = -(X - cx) * uy + (Y - cy) * ux
    n1 = _fbm2(w, h, 60, rng, 4)
    n2 = _fbm2(w, h, 22, rng, 3)
    n3 = _fbm2(w, h, (40, 14), rng, 3)
    wob = (n1 - 0.5) * 160
    sig = 170 + 70 * np.sin(s / 520.0 + 1.3)
    dd = d + wob
    band = np.exp(-(dd / sig) ** 2)
    core = np.exp(-(dd / (sig * 0.42)) ** 2)
    clump = np.clip((n2 - 0.25) * 1.6, 0, 1)
    inten = band * (0.35 + 0.65 * n1) * 0.55 + core * clump * 0.75
    lane = np.exp(-((dd - 30 * (n3 - 0.5) * 4) / (sig * 0.16)) ** 2) * np.clip((n3 - 0.35) * 2.2, 0, 1)
    inten = inten * (1 - 0.75 * lane * core)
    along = np.clip(1 - np.abs(s) / 1700.0, 0, 1) ** 0.6
    inten = np.clip(inten * along, 0, 1)
    inten = gaussian_filter(inten, 1.2)
    cA = np.array(rgb("#5A63B8"), np.float32)
    cB = np.array(rgb("#A08CDC"), np.float32)
    cC = np.array(rgb("#F4DDBE"), np.float32)
    k1 = np.clip(inten * 2.2, 0, 1)[..., None]
    k2 = np.clip((inten - 0.45) * 2.5, 0, 1)[..., None]
    col = cA * (1 - k1) + cB * k1
    col = col * (1 - k2) + cC * k2
    a = np.clip(inten * 0.62, 0, 0.8)[..., None]
    rgba = np.concatenate([col * a, a], -1)
    return _argb(rgba), X0, Y0, ds


def draw_sky(ctx, phase, t=0.0, cam_y=0.0, cam_x=0.0, horizon_y=None, sun_x=1500.0,
             milky_way=True, milky_alpha=1.0):
    """Full-screen sky gradient (+ horizon glow, + Milky Way at night).

    cam_y: camera tilt in px (negative = looking up; the s01 opening uses
    cam_y≈-1400 → 0). The sky moves with parallax SKY_PAR, so the gradient's
    horizon sits at HORIZON_Y - cam_y*SKY_PAR unless `horizon_y` is given.
    sun_x: x of the dawn glow centre.
    """
    C = sky_colors(phase)
    hy = HORIZON_Y - cam_y * SKY_PAR if horizon_y is None else horizon_y
    # everything in the sky is smooth -> render at 1/4 res and upscale (≈7 ms)
    q = 4
    small = cairo.ImageSurface(cairo.FORMAT_RGB24, W // q + 2, H // q + 2)
    c2 = cairo.Context(small)
    c2.scale(1.0 / q, 1.0 / q)
    y0, y1 = hy - 1300, hy + 40
    span = y1 - y0
    g = cairo.LinearGradient(0, y0, 0, y1)
    for yy, k in ((hy - 1300, "top"), (hy - 860, "upper"), (hy - 470, "mid"),
                  (hy - 190, "low"), (hy + 40, "horizon")):
        g.add_color_stop_rgb((yy - y0) / span, *C[k])
    c2.set_source(g)
    c2.paint()
    # broad horizon glow (town light pollution at night / violet pre-dawn / sunrise)
    c2.save()
    c2.translate(sun_x, hy + 30)
    c2.scale(1.0, 0.28)
    ga = C["hglow_a"]
    rg = radial_gradient(0, 0, 0, 1500, [(0, C["hglow"], ga), (0.3, C["hglow"], ga * 0.5),
                                         (0.65, C["hglow"], ga * 0.14), (1, C["hglow"], 0)])
    c2.set_source(rg)
    c2.paint()
    c2.restore()
    sun = C["sun"]
    if sun > 0.01:
        c2.save()
        c2.translate(sun_x, hy + 10)
        c2.scale(1.0, 0.42)
        rg = radial_gradient(0, 0, 0, 560, [(0, (1, 0.98, 0.9), 0.85 * sun ** 1.5),
                                            (0.15, (1.0, 0.92, 0.72), 0.5 * sun ** 1.5),
                                            (0.45, C["hglow"], 0.18 * sun), (1, C["hglow"], 0)])
        c2.set_source(rg)
        c2.paint()
        c2.restore()
    mw = C["milky"] * milky_alpha
    if milky_way and mw > 0.01:
        surf, X0, Y0, ds = _milky_way()
        c2.save()
        c2.translate(X0 - cam_x * SKY_PAR, Y0 - cam_y * SKY_PAR + (hy - (HORIZON_Y - cam_y * SKY_PAR)))
        c2.scale(ds, ds)
        c2.set_source_surface(surf, 0, 0)
        c2.get_source().set_filter(cairo.FILTER_BILINEAR)
        c2.paint_with_alpha(mw)
        c2.restore()
    small.flush()
    ctx.save()
    ctx.scale(q, q)
    ctx.set_source_surface(small, 0, 0)
    pat = ctx.get_source()
    pat.set_filter(cairo.FILTER_BILINEAR)
    pat.set_extend(cairo.EXTEND_PAD)
    ctx.paint()
    ctx.restore()
    return hy


@lru_cache(maxsize=4)
def _star_field(seed=7):
    rng = np.random.default_rng(seed)
    n1, n2 = 1500, 1100
    x = np.concatenate([rng.uniform(-60, 1980, n1), np.zeros(n2)])
    y = np.concatenate([rng.uniform(-1150, 820, n1), np.zeros(n2)])
    # denser faint stars along the Milky Way band
    ang = 0.40
    s = rng.uniform(-1500, 1500, n2)
    d = rng.normal(0, 150, n2)
    x[n1:] = 900 + s * math.cos(ang) - d * math.sin(ang)
    y[n1:] = -300 + s * math.sin(ang) + d * math.cos(ang)
    n = n1 + n2
    u = rng.random(n)
    r = 0.45 + 1.25 * u ** 4
    r[n1:] *= 0.75
    base = 0.35 + 0.65 * rng.random(n) ** 0.7
    base[n1:] *= 0.7
    bright = np.zeros(n, bool)
    bi = rng.choice(n1, 22, replace=False)
    bright[bi] = True
    r[bi] = rng.uniform(1.6, 2.6, len(bi))
    base[bi] = 1.0
    pal = np.array([rgb(c) for c in ("#FFFFFF", "#E2EBFF", "#C4D6FF", "#FFF2D8", "#FFE2BC", "#FFD6E6")])
    ci = rng.choice(len(pal), n, p=[0.36, 0.26, 0.14, 0.12, 0.07, 0.05])
    freq = rng.uniform(0.4, 2.4, n)
    ph = rng.uniform(0, TAU, n)
    amp = rng.uniform(0.15, 0.6, n)
    order = np.argsort(r)
    return dict(x=x[order], y=y[order], r=r[order], base=base[order], bright=bright[order],
                col=pal[ci][order], freq=freq[order], ph=ph[order], amp=amp[order])


def draw_glint(ctx, x, y, size, alpha=1.0, color="#FFFFFF", rot=0.0, glow=True, diag=0.45):
    """A 4-point star flare (cross spikes + soft glow + hot core)."""
    if alpha <= 0.003 or size <= 0.2:
        return
    c = _c(color)
    if glow:
        rg = radial_gradient(x, y, 0, size * 0.8, [(0, c, 0.5 * alpha), (0.3, c, 0.16 * alpha), (1, c, 0)])
        ctx.set_source(rg)
        ctx.arc(x, y, size * 0.8, 0, TAU)
        ctx.fill()
    for L, r0 in ((size, rot), (size * diag, rot + math.pi / 4)):
        if L < 0.8:
            continue
        w = max(0.5, L * 0.07)
        ctx.new_path()
        for k in range(4):
            a = r0 + k * math.pi / 2
            ctx.line_to(x + math.cos(a) * L, y + math.sin(a) * L)
            ai = a + math.pi / 4
            ctx.line_to(x + math.cos(ai) * w, y + math.sin(ai) * w)
        ctx.close_path()
        rg = radial_gradient(x, y, 0, L, [(0, (1, 1, 1), alpha), (0.2, c, alpha * 0.85), (1, c, 0)])
        ctx.set_source(rg)
        ctx.fill()
    ctx.set_source_rgba(1, 1, 1, min(1, alpha))
    ctx.arc(x, y, max(0.6, size * 0.06), 0, TAU)
    ctx.fill()


def draw_stars(ctx, phase, t=0.0, cam_y=0.0, cam_x=0.0, seed=7, horizon_y=None, alpha=1.0,
               flares=True):
    """Twinkling star field in sky space (same parallax as draw_sky).

    ~2600 stars (≈1500 on screen) of varied size/colour; 22 bright ones get
    cross flares. Stars fade out with phase (all gone at dawn) and near the
    horizon.
    """
    C = sky_colors(phase)
    vis = C["stars"] * alpha
    if vis <= 0.004:
        return
    S = _star_field(seed)
    hy = HORIZON_Y - cam_y * SKY_PAR if horizon_y is None else horizon_y
    dy = hy - (HORIZON_Y - cam_y * SKY_PAR)
    sx = (S["x"] - cam_x * SKY_PAR + 60) % 2040 - 60
    sy = S["y"] - cam_y * SKY_PAR + dy
    on = (sx > -20) & (sx < W + 20) & (sy > -20) & (sy < min(H + 20, hy + 5))
    idx = np.nonzero(on)[0]
    if len(idx) == 0:
        return
    tw = 1 - S["amp"][idx] * (0.5 + 0.5 * np.sin(TAU * S["freq"][idx] * t + S["ph"][idx])) ** 2
    ext = 0.15 + 0.85 * np.clip((hy - sy[idx]) / 420.0, 0, 1) ** 0.8
    ext *= np.clip((hy - sy[idx]) / 40.0, 0, 1)
    br = S["bright"][idx]
    vb = min(1.0, vis * 1.6)
    a = S["base"][idx] * tw * ext * np.where(br, vb, vis)
    xs, ys, rs, cols = sx[idx], sy[idx], S["r"][idx], S["col"][idx]
    for i in range(len(idx)):
        ai = a[i]
        if ai < 0.02:
            continue
        c = cols[i]
        r = rs[i]
        if br[i] and flares:
            draw_glint(ctx, xs[i], ys[i], r * (9 + 5 * tw[i]), ai * 0.95, c, glow=True)
            ctx.set_source_rgba(c[0], c[1], c[2], ai)
            ctx.arc(xs[i], ys[i], r * 0.9, 0, TAU)
            ctx.fill()
            continue
        ctx.set_source_rgba(c[0], c[1], c[2], ai)
        if r < 0.85:
            ctx.rectangle(xs[i] - r, ys[i] - r, 2 * r, 2 * r)
        else:
            ctx.arc(xs[i], ys[i], r, 0, TAU)
            if r > 1.25:
                ctx.fill()
                ctx.set_source_rgba(c[0], c[1], c[2], ai * 0.18)
                ctx.arc(xs[i], ys[i], r * 3.2, 0, TAU)
        ctx.fill()


def draw_moon(ctx, x, y, r=36.0, phase=0.0, t=0.0, illum=0.3, tilt=-2.4, alpha=1.0):
    """Crescent moon with halo. `tilt` = direction (rad) of the dark side;
    illum 0..0.5 controls crescent thickness."""
    C = sky_colors(phase)
    a = alpha * C["moon"]
    if a <= 0.01:
        return
    ctx.save()
    rg = radial_gradient(x, y, r * 0.9, r * 9, [(0, "#D4DEFF", 0.32 * a), (0.18, "#9FB0EE", 0.14 * a),
                                                (0.5, "#6A7AC0", 0.05 * a), (1, "#5060A8", 0)])
    ctx.set_source(rg)
    ctx.arc(x, y, r * 9, 0, TAU)
    ctx.fill()
    # earthshine disc (also hides stars behind the moon)
    sky = mix(C["upper"], C["mid"], 0.3)
    ctx.set_source_rgba(*mix(sky, (0.55, 0.6, 0.85), 0.18), a * (1 - C["sun"]))
    ctx.arc(x, y, r, 0, TAU)
    ctx.fill()
    ctx.push_group()
    lit = radial_gradient(x - math.cos(tilt) * r * 0.5, y - math.sin(tilt) * r * 0.5, 0, r * 1.6,
                          [(0, "#FFFBEA", 1), (0.6, "#FFF0CC", 1), (1, "#E9D9B8", 1)])
    ctx.set_source(lit)
    ctx.arc(x, y, r, 0, TAU)
    ctx.fill()
    ctx.set_operator(cairo.OPERATOR_DEST_OUT)
    off = r * 2 * clamp(illum, 0.02, 1.0)
    ctx.set_source_rgba(0, 0, 0, 1)
    ctx.arc(x + math.cos(tilt) * off, y + math.sin(tilt) * off, r * 1.03, 0, TAU)
    ctx.fill()
    ctx.pop_group_to_source()
    ctx.paint_with_alpha(a)
    ctx.restore()


@lru_cache(maxsize=4)
def _cloud_masks(seed=3):
    """Periodic stratus/wisp cloud masks in sky space (drawn at 3x scale)."""
    ds = 3
    TW, Y0, HH = 3072, -1150, 1980
    w, h = TW // ds, HH // ds
    rng = np.random.default_rng(seed)
    yy = (Y0 + np.arange(h) * ds).astype(np.float32)[:, None]
    n_big = _fbm2(w, h, (170, 26), rng, 5, periodic=True)
    n_det = _fbm2(w, h, (36, 9), rng, 3, periodic=True)
    n_mask = _fbm2(w, h, (260, 60), rng, 2, periodic=True)

    def band(c, th):
        return np.exp(-((yy - c) / th) ** 2)

    env = (band(HORIZON_Y - 110, 55) * 1.0 + band(HORIZON_Y - 250, 60) * 0.8
           + band(HORIZON_Y - 470, 90) * 0.75 + band(HORIZON_Y - 820, 120) * 0.55)
    dens = (n_big * 0.75 + n_det * 0.25) * env * (0.5 + n_mask)
    dens = np.clip((dens - 0.36) * 3.2, 0, 1) ** 1.3
    dens = gaussian_filter(dens, (1.0, 2.0), mode="wrap")
    lit = dens * (1 - np.roll(dens, 3, axis=0))
    lit = gaussian_filter(lit, 1.0, mode="wrap")
    und = dens * (1 - np.roll(dens, -3, axis=0))
    und = gaussian_filter(und, 1.0, mode="wrap")
    return _a8(dens), _a8(np.clip(lit * 1.8, 0, 1)), _a8(np.clip(und * 1.8, 0, 1)), TW, Y0, ds


def draw_clouds(ctx, phase, t=0.0, cam_y=0.0, cam_x=0.0, seed=3, alpha=1.0, speed=5.0,
                horizon_y=None):
    """Soft stratus bands + high wisps, moon-rimmed at night, pink/peach at dawn.
    Drifts slowly right (`speed` px/s)."""
    C = sky_colors(phase)
    dens, lit, und, TW, Y0, ds = _cloud_masks(seed)
    hy = HORIZON_Y - cam_y * SKY_PAR if horizon_y is None else horizon_y
    oy = Y0 - cam_y * SKY_PAR + (hy - (HORIZON_Y - cam_y * SKY_PAR))
    ox = (t * speed - cam_x * SKY_PAR) % TW - TW
    sun = C["sun"]
    small = cairo.ImageSurface(cairo.FORMAT_ARGB32, W // ds + 2, H // ds + 2)
    c2 = cairo.Context(small)
    ox /= ds
    oy /= ds
    while ox < W / ds:
        for m, col, a in ((dens, C["cloud"], 0.82), (und, C["cloud_under"], 0.35 + 0.55 * sun),
                          (lit, C["cloud_lit"], 0.75 - 0.25 * sun)):
            c2.set_source_rgba(*col, a)
            c2.mask_surface(m, round(ox), round(oy))
        ox += TW / ds
    small.flush()
    ctx.save()
    ctx.scale(ds, ds)
    ctx.set_source_surface(small, 0, 0)
    pat = ctx.get_source()
    pat.set_filter(cairo.FILTER_BILINEAR)
    pat.set_extend(cairo.EXTEND_PAD)
    ctx.paint_with_alpha(alpha)
    ctx.restore()


# =========================================================================== streaks & particles
def _tail_polygon(ctx, pts, widths):
    """Tapered ribbon around a polyline (pts head->tail) with per-point widths."""
    n = len(pts)
    left, right = [], []
    for i in range(n):
        x, y = pts[i]
        if i == 0:
            dx, dy = pts[1][0] - x, pts[1][1] - y
        elif i == n - 1:
            dx, dy = x - pts[i - 1][0], y - pts[i - 1][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        l = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / l, dx / l
        w = widths[i] * 0.5
        left.append((x + nx * w, y + ny * w))
        right.append((x - nx * w, y - ny * w))
    ctx.new_path()
    # round head cap
    x0, y0 = pts[0]
    ctx.arc(x0, y0, widths[0] * 0.5, 0, TAU)
    ctx.new_sub_path()
    ctx.move_to(*left[0])
    for p in left[1:]:
        ctx.line_to(*p)
    for p in reversed(right):
        ctx.line_to(*p)
    ctx.close_path()


def draw_shooting_star(ctx, head_xy, dir_angle, length, t=0.0, intensity=1.0, path=None,
                       width=1.0, color="#FFE9A8", sparkles=True, seed=0):
    """Comet-like shooting star.

    head_xy: head position; dir_angle: direction of travel (rad, screen space,
    0 = right, +pi/2 = down); length: tail length in px.  Alternatively pass
    `path` = list of recent head positions (oldest first, head last) to get a
    curved tail (e.g. an arcing, sputtering dive).  intensity 0..1+ scales
    brightness and head size; width scales thickness.
    """
    hx, hy = head_xy
    if intensity <= 0.003:
        return
    if path is not None and len(path) >= 2:
        pts = [tuple(p) for p in reversed(path)]
        pts[0] = (hx, hy)
        # resample by arclength into 32 points
        seg = [0.0]
        for i in range(1, len(pts)):
            seg.append(seg[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
        tot = seg[-1] or 1.0
        L = min(tot, length)
        res = []
        j = 0
        for k in range(33):
            s = L * k / 32
            while j < len(seg) - 2 and seg[j + 1] < s:
                j += 1
            u = (s - seg[j]) / max(1e-6, seg[j + 1] - seg[j])
            res.append((lerp(pts[j][0], pts[j + 1][0], u), lerp(pts[j][1], pts[j + 1][1], u)))
        pts = res
    else:
        dx, dy = math.cos(dir_angle), math.sin(dir_angle)
        pts = [(hx - dx * length * k / 32, hy - dy * length * k / 32) for k in range(33)]
    tx, ty = pts[-1]
    c = _c(color)
    I = intensity
    ctx.save()
    for wmax, cc, a, pw in ((34 * width * I ** 0.5, WINDOW_GLOW, 0.10, 1.2),
                            (14 * width, c, 0.28, 1.4),
                            (5.5 * width, c, 0.75, 1.7),
                            (2.2 * width, (1, 1, 1), 1.0, 2.2)):
        widths = [max(0.3, wmax * (1 - k / 32) ** pw) for k in range(33)]
        _tail_polygon(ctx, pts, widths)
        g = cairo.LinearGradient(hx, hy, tx, ty)
        g.add_color_stop_rgba(0, *cc, min(1, a * I))
        g.add_color_stop_rgba(0.25, *cc, min(1, a * I * 0.6))
        g.add_color_stop_rgba(1, *cc, 0)
        ctx.set_source(g)
        ctx.fill()
    # sparkles shed along the tail
    if sparkles:
        rng = np.random.default_rng(seed)
        n = 22
        off = rng.random(n)
        side = rng.normal(0, 1, n)
        sz = rng.uniform(0.5, 1.0, n)
        fr = rng.uniform(4, 11, n)
        for k in range(n):
            f = (off[k] + t * 1.3) % 1.0
            f = f ** 1.4
            idx = min(31, int(f * 32))
            u = f * 32 - idx
            px = lerp(pts[idx][0], pts[idx + 1][0], u)
            py = lerp(pts[idx][1], pts[idx + 1][1], u)
            ddx, ddy = pts[min(32, idx + 1)][0] - pts[idx][0], pts[min(32, idx + 1)][1] - pts[idx][1]
            l = math.hypot(ddx, ddy) or 1
            spread = side[k] * (4 + 30 * f) * width
            px += -ddy / l * spread
            py += ddx / l * spread
            tw = 0.5 + 0.5 * math.sin(t * fr[k] + k)
            a = I * (1 - f) ** 1.2 * (0.4 + 0.6 * tw)
            draw_glint(ctx, px, py, (5 + 9 * sz[k] * (1 - f)) * width, a, c, rot=k * 0.3, diag=0.0)
    # head
    R = 60 * width * I ** 0.7
    rg = radial_gradient(hx, hy, 0, R, [(0, (1, 1, 1), min(1, 0.95 * I)), (0.08, c, 0.8 * min(1, I)),
                                        (0.3, WINDOW_GLOW, 0.22 * min(1, I)), (1, WINDOW_GLOW, 0)])
    ctx.set_source(rg)
    ctx.arc(hx, hy, R, 0, TAU)
    ctx.fill()
    draw_glint(ctx, hx, hy, 46 * width * I ** 0.6, min(1, I), c, rot=dir_angle * 0.0 + 0.12, diag=0.5)
    ctx.restore()


def draw_sparkles(ctx, t, center, n=70, speed=520.0, life=1.7, gravity=160.0, seed=0,
                  color="#FFD66B", size=1.0, spread=TAU, direction=-math.pi / 2, drag=1.6,
                  flash=True):
    """Golden glitter burst. t = seconds since the burst (t<0 draws nothing).
    Particles fly out (cone `spread` around `direction`), slow by drag, fall
    under gravity, twinkle and fade."""
    if t < 0:
        return
    cx, cy = center
    rng = np.random.default_rng(seed)
    ang = direction + (rng.random(n) - 0.5) * spread
    sp = speed * (0.25 + 0.75 * rng.random(n) ** 0.6)
    lf = life * (0.45 + 0.55 * rng.random(n))
    sz = size * (0.5 + rng.random(n) ** 2 * 1.2)
    fr = rng.uniform(6, 16, n)
    ph = rng.uniform(0, TAU, n)
    pal = [_c(color), rgb("#FFF6D6"), rgb("#FFC94D"), rgb("#FFE9A8")]
    ci = rng.integers(0, 4, n)
    ctx.save()
    if flash and t < 0.5:
        a = (1 - t / 0.5) ** 2
        R = 220 * size * (0.6 + t)
        rg = radial_gradient(cx, cy, 0, R, [(0, (1, 1, 0.92), 0.9 * a), (0.25, _c(color), 0.45 * a),
                                            (1, _c(color), 0)])
        ctx.set_source(rg)
        ctx.arc(cx, cy, R, 0, TAU)
        ctx.fill()
    k = drag
    for i in range(n):
        if t > lf[i]:
            continue
        dist = sp[i] * (1 - math.exp(-k * t)) / k
        x = cx + math.cos(ang[i]) * dist
        y = cy + math.sin(ang[i]) * dist + 0.5 * gravity * t * t
        f = t / lf[i]
        tw = 0.55 + 0.45 * math.sin(t * fr[i] + ph[i])
        a = (1 - f) ** 1.3 * tw
        draw_glint(ctx, x, y, 12 * sz[i] * (1 - 0.4 * f), a, pal[ci[i]], rot=ph[i], diag=0.4)
    ctx.restore()


def draw_fireflies(ctx, t, n=24, region=(0, 0, W, H), seed=1, color="#C8F56A", size=1.0,
                   flow=(0.0, 0.0), alpha=1.0, wander=60.0):
    """Wandering, pulsing fireflies inside region=(x0, y0, x1, y1).
    flow=(vx, vy) px/s makes them stream (wrapping inside the region)."""
    x0, y0, x1, y1 = region
    rng = np.random.default_rng(seed)
    bx = rng.random(n)
    by = rng.random(n)
    fr = rng.uniform(0.35, 0.8, n)
    ph = rng.uniform(0, TAU, n)
    sp = rng.uniform(0.6, 1.4, n)
    sz = size * rng.uniform(0.6, 1.25, n)
    c = _c(color)
    core = mix(c, (1, 1, 0.9), 0.65)
    ww, hh = x1 - x0, y1 - y0
    ctx.save()
    for i in range(n):
        s = t * 0.22 * sp[i]
        x = bx[i] * ww + wander * (noise1(s + i * 3.7, seed) + 0.5 * math.sin(s * 2.1 + i))
        y = by[i] * hh + wander * 0.7 * (noise1(s + i * 5.3, seed + 9) + 0.4 * math.cos(s * 1.7 + i))
        x = (x + flow[0] * t * sp[i]) % (ww + 80) - 40 + x0
        y = (y + flow[1] * t * sp[i]) % (hh + 80) - 40 + y0
        p = 0.5 + 0.5 * math.sin(t * TAU * fr[i] * 0.5 + ph[i])
        b = (0.12 + 0.88 * smoothstep(0.3, 0.95, p)) * alpha
        R = 26 * sz[i]
        rg = radial_gradient(x, y, 0, R, [(0, c, 0.55 * b), (0.25, c, 0.18 * b), (1, c, 0)])
        ctx.set_source(rg)
        ctx.arc(x, y, R, 0, TAU)
        ctx.fill()
        ctx.set_source_rgba(*core, min(1, b * 1.1))
        ctx.arc(x, y, 2.3 * sz[i], 0, TAU)
        ctx.fill()
    ctx.restore()


def draw_dust_motes(ctx, t, region=(0, 0, W, H), n=60, seed=5, color="#FFE6B8", size=1.0,
                    alpha=0.5, light=None, drift=(4.0, -6.0)):
    """Slow drifting soft motes; light=(cx, cy, r) restricts them to a beam/pool."""
    x0, y0, x1, y1 = region
    rng = np.random.default_rng(seed)
    bx, by = rng.random(n), rng.random(n)
    sz = size * rng.uniform(0.8, 2.6, n)
    fr = rng.uniform(0.3, 1.2, n)
    ph = rng.uniform(0, TAU, n)
    c = _c(color)
    ww, hh = x1 - x0, y1 - y0
    for i in range(n):
        x = (bx[i] * ww + drift[0] * t + 18 * math.sin(t * 0.3 * fr[i] + ph[i])) % ww + x0
        y = (by[i] * hh + drift[1] * t + 14 * math.cos(t * 0.23 * fr[i] + ph[i] * 1.3)) % hh + y0
        a = alpha * (0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * fr[i] * 2 + ph[i])))
        if light is not None:
            lx, ly, lr = light
            a *= clamp(1 - math.hypot(x - lx, y - ly) / lr) ** 1.2
        if a < 0.01:
            continue
        rg = radial_gradient(x, y, 0, sz[i] * 2.5, [(0, c, a), (0.4, c, a * 0.4), (1, c, 0)])
        ctx.set_source(rg)
        ctx.arc(x, y, sz[i] * 2.5, 0, TAU)
        ctx.fill()


def draw_light_pool(ctx, x, y, rx, ry=None, color="#FFB85C", strength=0.6, op=cairo.OPERATOR_SCREEN):
    """Soft elliptical light (e.g. Kira's glow on the grass, a window spill).
    Uses SCREEN blending so it brightens what is below."""
    ry = rx if ry is None else ry
    c = _c(color)
    ctx.save()
    ctx.set_operator(op)
    ctx.translate(x, y)
    ctx.scale(1.0, ry / rx)
    rg = radial_gradient(0, 0, 0, rx, [(0, c, strength), (0.35, c, strength * 0.5),
                                       (0.7, c, strength * 0.15), (1, c, 0)])
    ctx.set_source(rg)
    ctx.arc(0, 0, rx, 0, TAU)
    ctx.fill()
    ctx.restore()


# =========================================================================== shared props
def _ribbon(ctx, pts, widths):
    """Tapered filled ribbon along a polyline."""
    n = len(pts)
    left, right = [], []
    for i in range(n):
        x, y = pts[i]
        a = pts[max(0, i - 1)]
        b = pts[min(n - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        l = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / l, dx / l
        w = widths[i] * 0.5
        left.append((x + nx * w, y + ny * w))
        right.append((x - nx * w, y - ny * w))
    ctx.new_path()
    ctx.move_to(*left[0])
    for p in left[1:]:
        ctx.line_to(*p)
    for p in reversed(right):
        ctx.line_to(*p)
    ctx.close_path()


def _bez(p0, p1, p2, p3, n=16):
    out = []
    for i in range(n + 1):
        u = i / n
        v = 1 - u
        out.append((v ** 3 * p0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u ** 3 * p3[0],
                    v ** 3 * p0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u ** 3 * p3[1]))
    return out


def draw_torii(ctx, x, base_y, w, h=None, phase=0.0, base_r=None, color=None, alpha=1.0,
               light=0.0, plaque=True):
    """Vermilion myōjin torii, front view, centred at x.

    base_y: ground under the left pillar (base_r: under the right one, for
    slopes / stairs). w: pillar span-ish width (kasagi is ~1.3w). h: pillar
    height (default 1.0*w).  light: extra warm lantern light 0..1 on the
    lower pillars.  Colours follow the time-of-day phase.
    """
    C = sky_colors(phase)
    col = _c(color) if color is not None else C["torii"]
    h = w * 1.0 if h is None else h
    bl = base_y
    br = base_y if base_r is None else base_r
    top = min(bl, br) - h
    lit = mix(col, C["rim"], 0.35)
    shade = _darken(col, 0.45)
    blk = mix((0.07, 0.05, 0.08), C["near"], 0.4)
    blk_lit = mix(blk, C["rim"], 0.35)
    ctx.save()
    if alpha < 1:
        ctx.push_group()
    px, pwb, pwt = w * 0.36, w * 0.085, w * 0.072
    for sgn, b in ((-1, bl), (1, br)):
        xc = x + sgn * px
        # pillar: lit left half + shaded right half
        ctx.move_to(xc - pwb / 2, b)
        ctx.line_to(xc - pwt / 2, top)
        ctx.line_to(xc + pwt / 2, top)
        ctx.line_to(xc + pwb / 2, b)
        ctx.close_path()
        ctx.set_source_rgb(*col)
        ctx.fill()
        ctx.move_to(xc + pwb * 0.12, b)
        ctx.line_to(xc + pwt * 0.12, top)
        ctx.line_to(xc + pwt / 2, top)
        ctx.line_to(xc + pwb / 2, b)
        ctx.close_path()
        ctx.set_source_rgba(*shade, 0.75)
        ctx.fill()
        ctx.move_to(xc - pwb / 2, b)
        ctx.line_to(xc - pwt / 2, top)
        ctx.line_to(xc - pwt / 2 + max(1.0, w * 0.012), top)
        ctx.line_to(xc - pwb / 2 + max(1.0, w * 0.014), b)
        ctx.close_path()
        ctx.set_source_rgba(*lit, 0.8)
        ctx.fill()
        # black nemaki at the foot
        ctx.rectangle(xc - pwb * 0.62, b - h * 0.07, pwb * 1.24, h * 0.07 + 1)
        ctx.set_source_rgb(*blk)
        ctx.fill()
        if light > 0:
            rg = radial_gradient(xc, b - h * 0.2, 0, h * 0.45, [(0, WINDOW, 0.35 * light), (1, WINDOW, 0)])
            ctx.set_source(rg)
            ctx.rectangle(xc - pwb, b - h * 0.65, pwb * 2, h * 0.65)
            ctx.fill()
    # nuki (tie beam)
    ny = top + h * 0.27
    ctx.rectangle(x - w * 0.52, ny, w * 1.04, h * 0.055)
    ctx.set_source_rgb(*col)
    ctx.fill()
    ctx.rectangle(x - w * 0.52, ny + h * 0.035, w * 1.04, h * 0.02)
    ctx.set_source_rgba(*shade, 0.7)
    ctx.fill()
    # gakuzuka + plaque
    ctx.rectangle(x - w * 0.03, top + h * 0.1, w * 0.06, h * 0.18)
    ctx.set_source_rgb(*col)
    ctx.fill()
    if plaque:
        ctx.rectangle(x - w * 0.055, top + h * 0.12, w * 0.11, h * 0.12)
        ctx.set_source_rgb(*blk)
        ctx.fill()
    # shimaki (red beam under the kasagi), slightly curved
    ctx.move_to(x - w * 0.6, top + h * 0.035)
    ctx.curve_to(x - w * 0.2, top + h * 0.075, x + w * 0.2, top + h * 0.075, x + w * 0.6, top + h * 0.035)
    ctx.line_to(x + w * 0.6, top + h * 0.1)
    ctx.curve_to(x + w * 0.2, top + h * 0.13, x - w * 0.2, top + h * 0.13, x - w * 0.6, top + h * 0.1)
    ctx.close_path()
    ctx.set_source_rgb(*col)
    ctx.fill()
    # kasagi (black top beam with upswept ends)
    ctx.move_to(x - w * 0.7, top - h * 0.06)
    ctx.curve_to(x - w * 0.45, top - h * 0.005, x + w * 0.45, top - h * 0.005, x + w * 0.7, top - h * 0.06)
    ctx.line_to(x + w * 0.64, top + h * 0.035)
    ctx.curve_to(x + w * 0.3, top + h * 0.06, x - w * 0.3, top + h * 0.06, x - w * 0.64, top + h * 0.035)
    ctx.close_path()
    ctx.set_source_rgb(*blk)
    ctx.fill()
    ctx.move_to(x - w * 0.7, top - h * 0.06)
    ctx.curve_to(x - w * 0.45, top - h * 0.005, x + w * 0.45, top - h * 0.005, x + w * 0.7, top - h * 0.06)
    ctx.set_line_width(max(1.0, h * 0.012))
    ctx.set_source_rgba(*blk_lit, 0.9)
    ctx.stroke()
    if alpha < 1:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(alpha)
    ctx.restore()


def draw_toro(ctx, x, base_y, h, phase=0.0, lit=1.0, t=0.0, seed=0):
    """Stone lantern (tōrō), front view, standing on (x, base_y), height h.
    lit 0..1 lights the fire box (with flicker)."""
    C = sky_colors(phase)
    st = C["stone"]
    sl = C["stone_lit"]
    sd = _darken(st, 0.35)
    u = h / 100.0
    ctx.save()

    def box(cx, y, bw, bh, taper=0.0):
        ctx.move_to(cx - bw / 2, y + bh)
        ctx.line_to(cx - bw / 2 + taper, y)
        ctx.line_to(cx + bw / 2 - taper, y)
        ctx.line_to(cx + bw / 2, y + bh)
        ctx.close_path()

    # base, pole, platform
    for (bw, y, bh, tp) in ((34, 92, 8, 2), (12, 60, 32, 1), (30, 52, 8, 3)):
        box(x, base_y - (100 - y) * u, bw * u, bh * u, tp * u)
        ctx.set_source_rgb(*st)
        ctx.fill()
        box(x, base_y - (100 - y) * u, bw * u, bh * u, tp * u)
        ctx.save()
        ctx.clip()
        ctx.rectangle(x + bw * u * 0.15, base_y - 200 * u, bw * u, 300 * u)
        ctx.set_source_rgba(*sd, 0.8)
        ctx.fill()
        ctx.rectangle(x - bw * u / 2, base_y - (100 - y) * u, bw * u, 1.6 * u)
        ctx.set_source_rgba(*sl, 0.7)
        ctx.fill()
        ctx.restore()
    # fire box
    fy0, fy1 = base_y - 48 * u, base_y - 30 * u
    ctx.rectangle(x - 11 * u, fy0 - 0 * u, 22 * u, fy1 - fy0 + 0 * u)
    ctx.set_source_rgb(*sd)
    ctx.fill()
    if lit > 0:
        f = lit * (0.85 + 0.15 * noise1(t * 6 + seed * 3.1, 11))
        ctx.rectangle(x - 7 * u, fy0 + 3 * u, 14 * u, fy1 - fy0 - 5 * u)
        ctx.set_source_rgba(*mix(WINDOW, WINDOW_HOT, 0.4), f)
        ctx.fill()
        rg = radial_gradient(x, (fy0 + fy1) / 2, 0, 42 * u, [(0, WINDOW, 0.4 * f), (0.4, WINDOW_GLOW, 0.12 * f),
                                                            (1, WINDOW_GLOW, 0)])
        ctx.set_source(rg)
        ctx.arc(x, (fy0 + fy1) / 2, 42 * u, 0, TAU)
        ctx.fill()
    else:
        ctx.rectangle(x - 7 * u, fy0 + 3 * u, 14 * u, fy1 - fy0 - 5 * u)
        ctx.set_source_rgb(*_darken(sd, 0.4))
        ctx.fill()
    # roof (kasa) with upturned corners + hōju knob
    ry = base_y - 48 * u
    ctx.move_to(x - 26 * u, ry - 3 * u)
    ctx.curve_to(x - 18 * u, ry - 2 * u, x - 10 * u, ry - 10 * u, x - 4 * u, ry - 16 * u)
    ctx.line_to(x + 4 * u, ry - 16 * u)
    ctx.curve_to(x + 10 * u, ry - 10 * u, x + 18 * u, ry - 2 * u, x + 26 * u, ry - 3 * u)
    ctx.line_to(x + 24 * u, ry + 1 * u)
    ctx.line_to(x - 24 * u, ry + 1 * u)
    ctx.close_path()
    ctx.set_source_rgb(*st)
    ctx.fill_preserve()
    ctx.save()
    ctx.clip()
    ctx.rectangle(x - 30 * u, ry - 20 * u, 30 * u, 5 * u + 20 * u)
    ctx.set_source_rgba(*sl, 0.35)
    ctx.fill()
    ctx.restore()
    ctx.arc(x, ry - 19 * u, 4 * u, 0, TAU)
    ctx.set_source_rgb(*st)
    ctx.fill()
    ctx.restore()


# ---------------------------------------------------------------- the giant camphor tree
@lru_cache(maxsize=4)
def _kusu_geom(seed=5):
    """Camphor tree geometry in local units: trunk base at (0,0), canopy
    ≈1100 wide, ≈560 tall, top at y≈-900."""
    rng = np.random.default_rng(seed)
    limbs = []
    ends = [(-470, -560), (-300, -700), (-120, -780), (60, -800), (230, -730), (430, -600), (560, -480)]
    for ex, ey in ends:
        sx = rng.uniform(-14, 14)
        p0 = (sx, -250)
        p1 = (sx + ex * 0.15, -380 + rng.uniform(-30, 30))
        p2 = (ex * 0.65 + rng.uniform(-40, 40), ey * 0.85)
        p3 = (ex, ey)
        limbs.append((p0, p1, p2, p3, rng.uniform(34, 46)))
    clumps = []

    def sub(r):
        n = int(rng.integers(8, 12))
        a0 = rng.uniform(0, TAU)
        subs = [(0.0, 0.0, r * 0.74)]
        for k in range(n):
            a = a0 + k * TAU / n + rng.uniform(-0.2, 0.2)
            d = r * rng.uniform(0.52, 0.68)
            subs.append((math.cos(a) * d, math.sin(a) * d, r * rng.uniform(0.27, 0.4)))
        return subs

    cy0, rx, ry = -640, 560, 300
    # interior mass (back)
    for i in range(26):
        a = rng.uniform(0, TAU)
        d = math.sqrt(rng.random()) * 0.62
        cx, cy = math.cos(a) * rx * d, cy0 + math.sin(a) * ry * d
        r = rng.uniform(105, 150)
        clumps.append((cx, cy - 300, cx, cy, r, sub(r), 0))
    # silhouette ring along the dome top
    for k in range(17):
        th = math.pi + math.pi * (k + rng.uniform(-0.3, 0.3)) / 16
        f = rng.uniform(0.8, 0.92)
        cx, cy = math.cos(th) * rx * f, cy0 + math.sin(th) * ry * f + 10
        r = rng.uniform(100, 138)
        clumps.append((cx, cy, cx, cy, r, sub(r), 1))
    # hanging lower fringe (front)
    for k in range(11):
        cx = -500 + k * 100 + rng.uniform(-25, 25)
        cy = -440 + 60 * (abs(cx) / 540) ** 2 * -1 + rng.uniform(-25, 20)
        r = rng.uniform(78, 112)
        clumps.append((cx, cy + 200, cx, cy, r, sub(r), 2))
    clumps.sort(key=lambda c: (c[6] == 2, c[0 + 1]))
    return dict(limbs=limbs, clumps=[(c[2], c[3], c[4], c[5], c[6]) for c in clumps])


def _clump_path(ctx, x, y, subs, sc, dx=0.0, dy=0.0):
    for (sx, sy, sr) in subs:
        ctx.new_sub_path()
        ctx.arc(x + (sx + dx) * sc, y + (sy + dy) * sc, sr * sc, 0, TAU)


def _draw_clump(L, x, y, r, subs, sc, big=True):
    """One cauliflower clump: body (a) + mid-tone (r) + lit crescent (g)."""
    ctx = L.ctx
    ctx.new_path()
    _clump_path(ctx, x, y, subs, sc)
    L.fill(0, 0, 0)
    for (ddx, ddy, ch) in ((0.08, 0.5, "r"), (0.22, 0.3, "g")):
        ctx.save()
        ctx.new_path()
        _clump_path(ctx, x, y, subs, sc)
        ctx.clip()
        ctx.push_group()
        ctx.set_source_rgba(1 if ch == "r" else 0, 1 if ch == "g" else 0, 0, 1)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_CLEAR)
        ctx.new_path()
        _clump_path(ctx, x, y, subs, sc, r * ddx, r * ddy)
        ctx.fill()
        ctx.pop_group_to_source()
        ctx.paint()
        ctx.restore()


def _draw_kusu_trunk(L, x, y, sc, geom, rng):
    ctx = L.ctx
    for (p0, p1, p2, p3, wd) in geom["limbs"]:
        pts = _bez(*[(x + px * sc, y + py * sc) for px, py in (p0, p1, p2, p3)], n=20)
        widths = [wd * sc * (1 - 0.65 * i / 20) for i in range(21)]
        _ribbon(ctx, pts, widths)
        L.fill(0, 0, 1)
        # a couple of twigs
        for k in (8, 13):
            bx, by = pts[k]
            ang = math.atan2(p3[1] - p0[1], p3[0] - p0[0]) + rng.choice([-1, 1]) * rng.uniform(0.5, 0.9)
            tw = [(bx, by), (bx + math.cos(ang) * 70 * sc, by + math.sin(ang) * 70 * sc - 20 * sc),
                  (bx + math.cos(ang) * 130 * sc, by + math.sin(ang) * 130 * sc - 30 * sc)]
            _ribbon(ctx, tw, [widths[k] * 0.5, widths[k] * 0.3, 2 * sc])
            L.fill(0, 0, 1)
    # trunk with root flare
    ctx.new_path()
    ctx.move_to(x - 150 * sc, y + 6 * sc)
    ctx.curve_to(x - 95 * sc, y - 4 * sc, x - 70 * sc, y - 40 * sc, x - 62 * sc, y - 110 * sc)
    ctx.curve_to(x - 58 * sc, y - 170 * sc, x - 50 * sc, y - 230 * sc, x - 40 * sc, y - 290 * sc)
    ctx.line_to(x + 44 * sc, y - 290 * sc)
    ctx.curve_to(x + 54 * sc, y - 230 * sc, x + 60 * sc, y - 170 * sc, x + 64 * sc, y - 110 * sc)
    ctx.curve_to(x + 72 * sc, y - 40 * sc, x + 100 * sc, y - 4 * sc, x + 160 * sc, y + 6 * sc)
    ctx.close_path()
    L.fill(0, 0, 1)
    # bark: lit streaks on the left, dark grooves
    for i in range(9):
        fx = -50 + i * 12 + rng.uniform(-3, 3)
        pts = [(x + (fx * (1 + 0.8 * (1 - k / 10) ** 3) + 5 * math.sin(k * 0.9 + i)) * sc, y - (k * 29) * sc)
               for k in range(11)]
        wd = [(2.2 + 1.5 * rng.random()) * sc] * 11
        _ribbon(ctx, pts, wd)
        if fx < -18:
            L.fill(0, 0.55, 0.45)
        else:
            L.fill(0, 0, 0.45)


def _draw_kusu_canopy(L, x, y, sc, geom, which=None):
    for (cx, cy, r, subs, grp) in geom["clumps"]:
        if which is not None and grp not in which:
            continue
        _draw_clump(L, x + cx * sc, y + cy * sc, r, subs, sc)


# ---------------------------------------------------------------- town
PADX = 260
TOWN_LAYERS = ("far", "hill", "mid", "near")
TOWN_PAR = {"far": 0.82, "hill": 0.9, "mid": 1.0, "near": 1.12}
_TOWN_BOX = {"far": (520, 1200), "hill": (-160, 1200), "mid": (560, 1200), "near": (420, 1200)}
# landmark positions (screen space at cam 0; hill layer)
TOWN_TREE = (1745.0, 356.0)          # trunk base of the camphor tree
TOWN_TREE_TOP = (1735.0, 60.0)       # approx. top of canopy
TOWN_SHRINE = (1585.0, 352.0)
TOWN_STEPS = ((1150.0, 800.0), (1497.0, 362.0))   # bottom -> top of the stone steps
TOWN_IMPACT = (560.0, 860.0)         # a good spot in the mid layer for the star to land


def _hill_top_y(x):
    return 352 + 432 * (1 - smoothstep(1040, 1580, x)) ** 1.25 + 7 * noise1(x / 55.0, 5) * (1 - smoothstep(1500, 1600, x))


def _wall_windows(L, x, top, w, h, rng, s, wins, p_lit, style):
    """Scatter windows on a wall; lit ones are recorded for per-frame drawing."""
    ctx = L.ctx
    ww = rng.uniform(5.5, 9) * s
    wh = rng.uniform(4.5, 6.5) * s
    if w < ww * 1.6 or h < wh * 1.5:
        return
    n = max(1, min(4, int(w / (ww * 2.2))))
    slots = rng.choice(n, size=int(rng.integers(1, n + 1)), replace=False)
    for k in slots:
        cx = x + (k + 0.5) * w / n + rng.uniform(-0.1, 0.1) * w / n
        wx, wy = cx - ww / 2, top + h * rng.uniform(0.25, 0.4)
        wy = min(wy, top + h - wh - 1.5 * s)
        ctx.rectangle(wx, wy, ww, wh)
        L.fill(0, 0.22, 0.25)
        if rng.random() < p_lit:
            kind = rng.choice(3, p=[0.8, 0.15, 0.05])     # warm / fluorescent / tv
            wins.append((wx, wy, ww, wh, float(rng.uniform(0.15, 0.9)), float(rng.uniform(0, 100)),
                         int(kind), style, float(rng.uniform(0.75, 1.0))))


def _roof(L, x, ye, w, rh, oh, inset, s, tiles=False, ridge=True):
    ctx = L.ctx
    xl, xr = x - oh, x + w + oh
    yr = ye - rh
    rl, rr = x + inset, x + w - inset
    up = rh * 0.08

    def outline():
        ctx.new_path()
        ctx.move_to(xl, ye - up)
        ctx.curve_to(xl + (rl - xl) * 0.55, ye - rh * 0.14, rl - (rl - xl) * 0.12, yr + rh * 0.32, rl, yr)
        ctx.line_to(rr, yr)
        ctx.curve_to(rr + (xr - rr) * 0.12, yr + rh * 0.32, xr - (xr - rr) * 0.55, ye - rh * 0.14, xr, ye - up)
        ctx.line_to(xr - s * 0.4, ye + s * 1.1)
        ctx.curve_to(x + w * 0.7, ye + s * 0.9, x + w * 0.3, ye + s * 0.9, xl + s * 0.4, ye + s * 1.1)
        ctx.close_path()

    outline()
    L.fill(1, 0, 0)
    if tiles:
        ctx.save()
        outline()
        ctx.clip()
        step = 3.4 * s
        xx = xl
        ctx.set_line_width(max(0.6, 0.38 * s))
        while xx < xr:
            ctx.move_to(xx, yr - 2)
            ctx.line_to(xx + (xx - (x + w / 2)) * 0.06, ye + 2 * s)
            xx += step
        L.stroke(max(0.6, 0.38 * s), 1, 0.38, 0)
        # tile courses (subtle horizontal bands)
        yy = yr + 4 * s
        while yy < ye:
            ctx.move_to(xl, yy)
            ctx.line_to(xr, yy)
            yy += 5 * s
        L.stroke(max(0.5, 0.25 * s), 1, 0.15, 0.35)
        ctx.restore()
        # round eave tile ends
        xx = xl + step * 0.5
        while xx < xr - step * 0.3:
            ctx.new_sub_path()
            ctx.arc(xx, ye + s * 0.55, s * 0.9, 0, TAU)
            xx += step
        L.fill(1, 0.6, 0)
    # shadow band under the eave
    ctx.rectangle(x, ye + s * 1.0, w, s * 2.2)
    L.fill(0, 0, 0.8)
    if ridge and rr - rl > 2 * s:
        ctx.rectangle(rl - s * 0.8, yr - s * 1.6, rr - rl + s * 1.6, s * 1.9)
        L.fill(1, 0.25, 0)
        for ex in (rl - s * 0.8, rr + s * 0.8):
            ctx.new_path()
            ctx.move_to(ex - s * 1.3, yr + s * 0.2)
            ctx.line_to(ex - s * 0.9, yr - s * 3.2)
            ctx.line_to(ex + s * 0.9, yr - s * 3.4)
            ctx.line_to(ex + s * 1.3, yr + s * 0.2)
            ctx.close_path()
            L.fill(1, 0.3, 0)


def _house(L, x, base, w, hw, rng, s, wins, kind=None, tiles=False, ext=20.0, p_lit=0.3, style=0):
    ctx = L.ctx
    if kind is None:
        kind = rng.choice(["hip", "hip2", "gable", "apato"], p=[0.38, 0.37, 0.17, 0.08])
    if kind == "hip":
        ye = base - hw
        ctx.rectangle(x, ye, w, hw + ext)
        L.fill()
        _wall_windows(L, x, ye, w, hw, rng, s, wins, p_lit, style)
        _roof(L, x, ye, w, min(w * 0.3, 16 * s + w * 0.12), w * 0.07 + s, w * 0.24, s, tiles)
    elif kind == "hip2":
        h1 = hw * 0.55
        ye1 = base - h1
        uw = w * rng.uniform(0.55, 0.8)
        ux = x + (w - uw) * rng.uniform(0.1, 0.9)
        h2 = hw * rng.uniform(0.5, 0.62)
        ye2 = ye1 - h2
        ctx.rectangle(ux, ye2, uw, h2 + 4 * s)
        L.fill()
        _wall_windows(L, ux, ye2, uw, h2, rng, s, wins, p_lit, style)
        _roof(L, ux, ye2, uw, min(uw * 0.3, 14 * s + uw * 0.1), uw * 0.08 + s, uw * 0.24, s, tiles)
        ctx.rectangle(x, ye1, w, h1 + ext)
        L.fill()
        _wall_windows(L, x, ye1, w, h1, rng, s, wins, p_lit, style)
        rh1 = min(h1 * 0.4, 7 * s + w * 0.05)
        _roof(L, x, ye1, w, rh1, w * 0.06 + s, rh1 * 1.6, s, tiles, ridge=False)
    elif kind == "gable":
        ye = base - hw
        ctx.rectangle(x, ye, w, hw + ext)
        L.fill()
        _wall_windows(L, x, ye, w, hw, rng, s, wins, p_lit, style)
        ap = ye - w * 0.42
        oh = w * 0.1
        ctx.new_path()
        ctx.move_to(x - oh, ye + s)
        ctx.curve_to(x + w * 0.12, ye - w * 0.1, x + w * 0.38, ap + w * 0.06, x + w / 2, ap)
        ctx.curve_to(x + w * 0.62, ap + w * 0.06, x + w * 0.88, ye - w * 0.1, x + w + oh, ye + s)
        ctx.line_to(x + w + oh - 2 * s, ye + 2.4 * s)
        ctx.curve_to(x + w * 0.85, ye - w * 0.06, x + w * 0.6, ap + w * 0.1, x + w / 2, ap + 2.6 * s)
        ctx.curve_to(x + w * 0.4, ap + w * 0.1, x + w * 0.15, ye - w * 0.06, x - oh + 2 * s, ye + 2.4 * s)
        ctx.close_path()
        L.fill(1, 0.35, 0)
        # gable wall triangle + vent
        ctx.new_path()
        ctx.move_to(x + 1.5 * s, ye + 1)
        ctx.curve_to(x + w * 0.2, ye - w * 0.08, x + w * 0.4, ap + w * 0.12, x + w / 2, ap + 3 * s)
        ctx.curve_to(x + w * 0.6, ap + w * 0.12, x + w * 0.8, ye - w * 0.08, x + w - 1.5 * s, ye + 1)
        ctx.close_path()
        L.fill()
        ctx.rectangle(x + w / 2 - 2 * s, ap + w * 0.2, 4 * s, 3 * s)
        L.fill(0, 0, 0.7)
    else:  # apato: small flat-roofed apartment block with an outdoor corridor
        w = w * 1.25
        hh = hw * 1.45
        ye = base - hh
        ctx.rectangle(x, ye, w, hh + ext)
        L.fill()
        ctx.rectangle(x - s, ye - 1.5 * s, w + 2 * s, 2.2 * s)
        L.fill(0, 0.3, 0)
        for fl in range(2):
            fy = ye + hh * (0.08 + 0.5 * fl)
            n = max(2, int(w / (13 * s)))
            for k in range(n):
                wx = x + (k + 0.25) * w / n
                ctx.rectangle(wx, fy + hh * 0.08, w / n * 0.45, hh * 0.22)
                L.fill(0, 0.2, 0.3)
                if rng.random() < p_lit * 1.4:
                    wins.append((wx, fy + hh * 0.08, w / n * 0.45, hh * 0.22, float(rng.uniform(0.2, 0.9)),
                                 float(rng.uniform(0, 100)), int(rng.choice(3, p=[0.6, 0.35, 0.05])), 0,
                                 float(rng.uniform(0.75, 1))))
            ctx.rectangle(x, fy + hh * 0.34, w, 1.2 * s)
            L.fill(0, 0.4, 0)
    return w


def _bush(L, x, base, r, rng, s):
    ctx = L.ctx
    ctx.rectangle(x - 0.8 * s, base - r, 1.6 * s, r)
    L.fill(0, 0, 1)
    n = int(rng.integers(4, 8))
    subs = [(0.0, 0.0, r * 0.7)] + [(math.cos(a) * r * 0.55, math.sin(a) * r * 0.45, r * rng.uniform(0.35, 0.5))
                                    for a in rng.uniform(0, TAU, n)]
    _draw_clump(L, x, base - r * 1.2, r, subs, 1.0)


def _pole(L, x, base, top, s):
    ctx = L.ctx
    pw = 1.5 * s
    ctx.move_to(x - pw, base)
    ctx.line_to(x - pw * 0.65, top)
    ctx.line_to(x + pw * 0.65, top)
    ctx.line_to(x + pw, base)
    ctx.close_path()
    L.fill()
    pts = []
    for yy, hw in ((top + 4 * s, 13 * s), (top + 13 * s, 10 * s)):
        ctx.rectangle(x - hw, yy, 2 * hw, 1.5 * s)
        L.fill()
        for ex in (x - hw * 0.9, x - hw * 0.35, x + hw * 0.35, x + hw * 0.9):
            ctx.rectangle(ex - 0.6 * s, yy - 1.8 * s, 1.2 * s, 1.8 * s)
            L.fill(0, 0.4, 0)
            pts.append((ex, yy - 1.5 * s))
    # transformer drum
    ctx.rectangle(x + pw * 0.6, top + 26 * s, 5.5 * s, 9 * s)
    L.fill()
    ctx.rectangle(x + pw * 0.6, top + 26 * s, 5.5 * s, 1.2 * s)
    L.fill(0, 0.35, 0)
    # step bolts
    for k in range(6):
        yy = top + 40 * s + k * 7 * s
        ctx.rectangle(x - pw - 2 * s if k % 2 else x + pw, yy, 2 * s, 0.8 * s)
        L.fill()
    return pts


def _wires(L, a_pts, b_pts, sag, width):
    ctx = L.ctx
    for (x0, y0), (x1, y1) in zip(a_pts, b_pts):
        ctx.new_path()
        for i in range(25):
            u = i / 24
            ctx.line_to(lerp(x0, x1, u), lerp(y0, y1, u) + sag * 4 * u * (1 - u))
        L.stroke(width)


def _gen_far(L, rng, wins, ex):
    ctx = L.ctx
    xs = np.arange(-PADX, W + PADX + 1, 6)
    ctx.new_path()
    ctx.move_to(-PADX, 1200)
    for x in xs:
        y = 612 + 22 * noise1(x / 420.0, 21) + 9 * noise1(x / 95.0, 22) - 40 * smoothstep(900, 1500, x)
        ctx.line_to(x, y)
    ctx.line_to(W + PADX, 1200)
    ctx.close_path()
    L.fill(0, 0.62, 0)
    ex["lamps"] = []
    for j in range(9):
        base0 = 670 + j * 17
        x = -PADX + rng.uniform(0, 30)
        while x < W + PADX:
            w = rng.uniform(16, 40)
            hw = rng.uniform(7, 14)
            b = base0 - 0.05 * max(0, x) + rng.uniform(-2, 2)
            if rng.random() < 0.1:
                _bush(L, x + w / 2, b, rng.uniform(6, 11), rng, 1.0)
            else:
                w = _house(L, x, b, w, hw, rng, 1.0, wins, p_lit=0.3, style=0)
            x += w + rng.uniform(-3, 5)
    ctx.rectangle(-PADX, 830, W + 2 * PADX, 400)
    L.fill()


def _gen_hill(L, rng, wins, ex):
    ctx = L.ctx
    # hill body
    ctx.new_path()
    ctx.move_to(840, 1200)
    for x in range(840, W + PADX + 1, 5):
        ctx.line_to(x, _hill_top_y(x))
    ctx.line_to(W + PADX, 1200)
    ctx.close_path()
    L.fill()
    (sx0, sy0), (sx1, sy1) = TOWN_STEPS

    def step_dist(px, py):
        vx, vy = sx1 - sx0, sy1 - sy0
        u = clamp(((px - sx0) * vx + (py - sy0) * vy) / (vx * vx + vy * vy))
        return math.hypot(px - (sx0 + vx * u), py - (sy0 + vy * u)), u

    # forest clumps covering the hillside (skip the summit clearing & the path)
    pts = []
    for i in range(1100):
        x = rng.uniform(860, W + PADX)
        top = _hill_top_y(x)
        if 1460 < x < 2010 and rng.random() < 0.85:
            continue
        y = top + rng.uniform(2, 40) + rng.exponential(140)
        pts.append((y, x))
    pts.sort()
    for y, x in pts:
        d, u = step_dist(x, y)
        r = rng.uniform(14, 30) * (1 + (y - 350) / 700)
        if d < r + 10:
            continue
        subs = [(0.0, 0.0, r * 0.72)] + [(math.cos(a) * r * 0.6, math.sin(a) * r * 0.5, r * rng.uniform(0.3, 0.42))
                                         for a in np.linspace(0, TAU, 8, endpoint=False) + rng.uniform(0, 1)]
        _draw_clump(L, x, y, r, subs, 1.0)
    # the stone steps: a pale band with step lines
    n = 60
    band = []
    for k in range(n + 1):
        u = k / n
        band.append((lerp(sx0, sx1, u) + 6 * math.sin(u * 5), lerp(sy0, sy1, u)))
    _ribbon(ctx, band, [lerp(20, 9, k / n) for k in range(n + 1)])
    L.fill(0, 0.85, 0)
    for k in range(0, n, 1):
        x, y = band[k]
        wdt = lerp(20, 9, k / n) * 0.5
        ctx.rectangle(x - wdt, y - 0.8, wdt * 2, 1.2)
        L.fill(0, 0.35, 0.3)
    ex["path"] = band
    # a few trees that overlap the path to anchor it in the slope
    for u, side in ((0.18, 1), (0.42, -1), (0.63, 1), (0.8, -1)):
        x = lerp(sx0, sx1, u) + side * 26
        y = lerp(sy0, sy1, u) + 14
        r = 20
        subs = [(0.0, 0.0, r * 0.72)] + [(math.cos(a) * r * 0.6, math.sin(a) * r * 0.5, r * 0.4)
                                         for a in np.linspace(0, TAU, 8, endpoint=False)]
        _draw_clump(L, x, y, r, subs, 1.0)
    # little houses on the lower slope
    for x, w in ((880, 70), (960, 56), (1030, 62), (905, 50)):
        b = _hill_top_y(x + w / 2) + 70 + rng.uniform(0, 30)
        _house(L, x, b, w, rng.uniform(22, 30), rng, 1.8, wins, p_lit=0.5, style=0)
    # summit: shrine
    shx, shy = TOWN_SHRINE
    _shrine(L, shx - 40, shy + 2, 80, 1.2)
    # the camphor tree
    tx, ty = TOWN_TREE
    geom = _kusu_geom(5)
    sc = 0.46
    _draw_kusu_trunk(L, tx, ty, sc, geom, rng)
    _draw_kusu_canopy(L, tx, ty, sc, geom)
    # bushes around the summit
    for x in (1470, 1635, 1905, 1990, 2060):
        _bush(L, x, _hill_top_y(x) + 6, rng.uniform(9, 16), rng, 1.2)
    ex["torii"] = [(sx0, sy0, 34.0), (sx1, sy1 + 4, 20.0)]
    ex["shrine_lights"] = [(shx - 30, shy - 12), (shx + 30, shy - 12)]


def _shrine(L, x, base, w, s):
    """Small shrine hall with a sweeping irimoya roof (coded)."""
    ctx = L.ctx
    ctx.rectangle(x - w * 0.12, base - w * 0.08, w * 1.24, w * 0.08)
    L.fill(0, 0.5, 0)
    bh = w * 0.36
    by = base - w * 0.08 - bh
    ctx.rectangle(x, by, w, bh)
    L.fill()
    ctx.rectangle(x + w * 0.3, by + bh * 0.3, w * 0.4, bh * 0.7)
    L.fill(0, 0, 0.8)
    for k in range(5):
        ctx.rectangle(x + w * (0.08 + k * 0.21), by, w * 0.035, bh)
        L.fill(0, 0.3, 0)
    oh = w * 0.3
    rh = w * 0.52
    ye = by + 2
    ctx.new_path()
    ctx.move_to(x - oh, ye - w * 0.14)
    ctx.curve_to(x - oh * 0.3, ye - w * 0.01, x + w * 0.1, ye + w * 0.03, x + w * 0.3, ye + w * 0.03)
    ctx.line_to(x + w * 0.7, ye + w * 0.03)
    ctx.curve_to(x + w * 0.9, ye + w * 0.03, x + w + oh * 0.3, ye - w * 0.01, x + w + oh, ye - w * 0.14)
    ctx.curve_to(x + w + oh * 0.4, ye - w * 0.12, x + w * 0.9, ye - rh * 0.5, x + w * 0.78, ye - rh)
    ctx.line_to(x + w * 0.22, ye - rh)
    ctx.curve_to(x + w * 0.1, ye - rh * 0.5, x - oh * 0.4, ye - w * 0.12, x - oh, ye - w * 0.14)
    ctx.close_path()
    L.fill(1, 0, 0)
    # ridge + chigi
    ctx.rectangle(x + w * 0.18, ye - rh - s * 2.5, w * 0.64, s * 3)
    L.fill(1, 0.4, 0)
    for cx, sg in ((x + w * 0.2, -1), (x + w * 0.8, 1)):
        ctx.new_path()
        ctx.move_to(cx, ye - rh)
        ctx.line_to(cx + sg * w * 0.08, ye - rh - w * 0.14)
        L.stroke(max(1.0, s * 1.5), 1, 0.5, 0)
    # gable triangle (irimoya)
    ctx.new_path()
    ctx.move_to(x + w * 0.32, ye - rh * 0.42)
    ctx.line_to(x + w * 0.5, ye - rh * 0.95)
    ctx.line_to(x + w * 0.68, ye - rh * 0.42)
    ctx.close_path()
    L.fill(1, 0.45, 0.2)


def _gen_mid(L, rng, wins, ex):
    ctx = L.ctx
    ex["lamps"] = []
    rows = [(760, 70, 120, 20, 2.0, 0.5), (812, 80, 140, 24, 2.3, 0.6), (868, 90, 160, 28, 2.6, 0.7)]
    for base0, wmin, wmax, hmin, s, _ in rows:
        x = -PADX + rng.uniform(0, 40)
        while x < W + PADX:
            w = rng.uniform(wmin, wmax)
            b = base0 - 0.07 * max(0, x - 200) + rng.uniform(-4, 4)
            if b < _hill_top_y(x + w / 2) + 40 and x > 1000:
                x += w
                continue
            r = rng.random()
            if r < 0.14:
                _bush(L, x + w * 0.4, b, rng.uniform(16, 30), rng, s)
                w *= 0.7
            else:
                w = _house(L, x, b, w, rng.uniform(hmin, hmin * 1.6), rng, s, wins, tiles=True,
                           p_lit=0.32, style=1)
            x += w + rng.uniform(-6, 12)
    # poles + wires
    poles = []
    for x in (120, 700, 1260):
        b = 900 - 0.07 * max(0, x - 200)
        poles.append(_pole(L, x, b, b - 190, 2.0))
    for a, b in zip(poles[:-1], poles[1:]):
        _wires(L, a, b, 22, 1.1)
    # street lamps (light is drawn per frame)
    for x in (390, 980):
        b = 880 - 0.07 * max(0, x - 200)
        ctx.rectangle(x - 1.5, b - 110, 3, 110)
        L.fill()
        ctx.rectangle(x - 1.5, b - 110, 16, 3)
        L.fill()
        ex["lamps"].append((x + 13, b - 106))
    ctx.rectangle(-PADX, 900, W + 2 * PADX, 400)
    L.fill()


def _gen_near(L, rng, wins, ex):
    ctx = L.ctx
    ex["lamps"] = []
    # big foreground roofs
    specs = [(-190, 1000, 560, 70, "hip2"), (330, 1050, 300, 60, "hip"), (620, 1085, 330, 50, "hip"),
             (930, 1075, 300, 56, "gable"), (1260, 1030, 380, 80, "gable"),
             (1580, 990, 620, 64, "hip")]
    for x, b, w, hw, kind in specs:
        _house(L, x, b, w, hw, rng, 3.6, wins, kind=kind, tiles=True, p_lit=0.55, style=1, ext=200)
    for x, r in ((600, 50), (1210, 46)):
        _bush(L, x, 1080, r, rng, 3.0)
    # concrete block wall along the bottom
    ctx.rectangle(-PADX, 1050, W + 2 * PADX, 200)
    L.fill()
    for k in range(-PADX, W + PADX, 66):
        ctx.rectangle(k, 1050, 1.5, 30)
        L.fill(0, 0.18, 0)
    ctx.rectangle(-PADX, 1047, W + 2 * PADX, 4)
    L.fill(0, 0.45, 0)
    # tall poles with wires across the sky
    p1 = _pole(L, 250, 1080, 560, 3.0)
    p2 = _pole(L, 1320, 1080, 610, 3.0)
    left = [(-PADX, y + 40) for (_, y) in p1]
    right = [(W + PADX, y + 70) for (_, y) in p2]
    _wires(L, left, p1, 30, 1.5)
    _wires(L, p1, p2, 55, 1.5)
    _wires(L, p2, right, 30, 1.5)


_TOWN_GEN = {"far": _gen_far, "hill": _gen_hill, "mid": _gen_mid, "near": _gen_near}
_TOWN_RIM = {"far": (1, 2, 0.5), "hill": (2, 3, 0.7), "mid": (2, 3, 0.7), "near": (3, 4, 0.8)}


@lru_cache(maxsize=8)
def _town(layer, seed=11):
    y0, y1 = _TOWN_BOX[layer]
    rng = np.random.default_rng(seed * 101 + TOWN_LAYERS.index(layer))
    L = _Coded(W + 2 * PADX, y1 - y0, -PADX, y0)
    wins, ex = [], {}
    _TOWN_GEN[layer](L, rng, wins, ex)
    L.bake(rim=_TOWN_RIM[layer])
    return dict(sprite=L, wins=wins, **ex)


def _town_cols(layer, C, oy):
    fog, mist = C["fog"], C["mist"]
    rim = C["rim"]
    if layer == "far":
        base = mix(C["far"], fog, 0.1)
        return dict(a=linear_gradient(0, 600 + oy, 0, 860 + oy, [(0, base), (1, mix(base, fog, 0.55 + 0.3 * mist))]),
                    r=mix(base, rim, 0.14), g=mix(base, fog, 0.5), b=_darken(base, 0.18), rim=(rim, 0.45))
    if layer == "hill":
        base = C["hill"]
        return dict(a=linear_gradient(0, 300 + oy, 0, 900 + oy, [(0, base), (0.6, base), (1, mix(base, fog, 0.35 + 0.3 * mist))]),
                    r=mix(base, C["leaf_mid"], 0.45), g=mix(base, rim, 0.38), b=_darken(base, 0.35), rim=(rim, 0.85))
    if layer == "mid":
        base = C["mids"]
        return dict(a=linear_gradient(0, 700 + oy, 0, 960 + oy, [(0, base), (0.55, base), (1, mix(base, fog, 0.3 + 0.3 * mist))]),
                    r=C["roof"], g=mix(C["roof"], rim, 0.4), b=_darken(base, 0.45), rim=(rim, 0.75))
    base = C["near"]
    return dict(a=base, r=C["roof_near"], g=mix(C["roof_near"], rim, 0.33), b=_darken(base, 0.5), rim=(rim, 0.5))


def _draw_windows(ctx, wins, C, t, ox, oy, big):
    f = C["windows"]
    for (x, y, w, h, thr, sd, kind, style, br) in wins:
        on = smoothstep(thr - 0.1, thr + 0.1, f) * br
        if on < 0.02:
            continue
        x += ox
        y += oy
        fl = 1 + 0.05 * noise1(t * 1.7 + sd, 3)
        if kind == 2:   # television
            fl = 0.75 + 0.25 * noise1(t * 7 + sd, 4)
            c0, c1 = rgb("#CFE0FF"), rgb("#8FA8FF")
        elif kind == 1:  # fluorescent
            c0, c1 = rgb("#FFF4DE"), rgb("#FFE0B0")
        else:
            c0, c1 = WINDOW_HOT, WINDOW
        a = clamp(on * fl)
        if big and w > 6:
            R = max(w, h) * 1.6
            rg = radial_gradient(x + w / 2, y + h / 2, 0, R, [(0, c1, 0.28 * a), (0.5, WINDOW_GLOW, 0.08 * a), (1, WINDOW_GLOW, 0)])
            ctx.set_source(rg)
            ctx.rectangle(x + w / 2 - R, y + h / 2 - R, 2 * R, 2 * R)
            ctx.fill()
            g = linear_gradient(0, y, 0, y + h, [(0, c0, a), (1, c1, a)])
            ctx.set_source(g)
            ctx.rectangle(x, y, w, h)
            ctx.fill()
            if style == 1 and w > 12:
                ctx.set_source_rgba(*mix(c1, (0.3, 0.15, 0.1), 0.6), 0.55 * a)
                ctx.set_line_width(max(1.0, w * 0.05))
                n = max(2, int(w / 9))
                for k in range(1, n):
                    ctx.move_to(x + w * k / n, y)
                    ctx.line_to(x + w * k / n, y + h)
                ctx.move_to(x, y + h * 0.45)
                ctx.line_to(x + w, y + h * 0.45)
                ctx.stroke()
        else:
            ctx.set_source_rgba(*c1, 0.18 * a)
            ctx.rectangle(x - w * 0.8, y - h * 0.8, w * 2.6, h * 2.6)
            ctx.fill()
            ctx.set_source_rgba(*c0, a)
            ctx.rectangle(x, y, w, h)
            ctx.fill()


def _warm_dot(ctx, x, y, r, a, col=WINDOW):
    rg = radial_gradient(x, y, 0, r * 5, [(0, WINDOW_HOT, a), (0.15, col, a * 0.55), (0.45, WINDOW_GLOW, a * 0.12),
                                          (1, WINDOW_GLOW, 0)])
    ctx.set_source(rg)
    ctx.arc(x, y, r * 5, 0, TAU)
    ctx.fill()


def draw_town(ctx, layer, phase, t=0.0, cam_x=0.0, cam_y=0.0, seed=11, lights=True):
    """Layered hillside town. layer: 'far' | 'hill' | 'mid' | 'near' | 'all'.

    Screen layout at cam (0, 0): sky horizon y≈660; far rooftops y≈640-830;
    the hill rises on the right (x>1050) to a summit at y≈352 with the shrine
    (TOWN_SHRINE) and the giant camphor tree (TOWN_TREE, canopy up to y≈60);
    stone steps TOWN_STEPS climb the hill face; mid town y≈700-900 with poles;
    near roofs y≈880-1080 and tall poles/wires up to y≈560.
    cam_x/cam_y shift each layer with parallax TOWN_PAR (tilt: pass the same
    cam_y you give draw_sky/draw_stars).  Everything is baked once; per frame
    cost ≈4-12 ms per layer.
    """
    if layer == "all":
        for l in TOWN_LAYERS:
            draw_town(ctx, l, phase, t, cam_x, cam_y, seed, lights)
        return
    T = _town(layer, seed)
    C = sky_colors(phase)
    par = TOWN_PAR[layer]
    ox, oy = round(-cam_x * par), round(-cam_y * par)
    y0, y1 = _TOWN_BOX[layer]
    if y0 + oy > H or y1 + oy < 0:
        return
    T["sprite"].paint(ctx, _town_cols(layer, C, oy), ox, oy)
    if not lights:
        return
    wf = C["windows"]
    if layer == "hill":
        for (x, y, w) in T["torii"]:
            draw_torii(ctx, x + ox, y + oy, w, w * 0.9, phase=phase)
        path = T["path"]
        for k in range(3, len(path) - 2, 4):
            x, y = path[k]
            side = 1 if (k // 4) % 2 else -1
            fl = 0.8 + 0.2 * noise1(t * 3 + k, 8)
            _warm_dot(ctx, x + ox + side * 9, y + oy - 3, 1.3, 0.75 * wf * fl)
        for (x, y) in T["shrine_lights"]:
            _warm_dot(ctx, x + ox, y + oy, 1.8, 0.9 * wf)
    if layer in ("mid", "far"):
        for (x, y) in T.get("lamps", []):
            rg = radial_gradient(x + ox, y + oy, 0, 60, [(0, (0.92, 0.95, 1.0), 0.9 * wf), (0.08, (0.75, 0.85, 1.0), 0.35 * wf),
                                                         (1, (0.6, 0.7, 1.0), 0)])
            ctx.set_source(rg)
            ctx.arc(x + ox, y + oy, 60, 0, TAU)
            ctx.fill()
            g = linear_gradient(0, y + oy, 0, y + oy + 110, [(0, (0.8, 0.88, 1.0), 0.16 * wf), (1, (0.8, 0.88, 1.0), 0)])
            ctx.set_source(g)
            ctx.move_to(x + ox - 3, y + oy)
            ctx.line_to(x + ox + 3, y + oy)
            ctx.line_to(x + ox + 40, y + oy + 110)
            ctx.line_to(x + ox - 40, y + oy + 110)
            ctx.close_path()
            ctx.fill()
    _draw_windows(ctx, T["wins"], C, t, ox, oy, big=layer in ("mid", "near", "hill"))
