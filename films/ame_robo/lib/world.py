"""Environment library for 「あめふりロボ」 — rainy shopping street (商店街), the dagashi shopfront,
rain / drips / splashes, overcast→clearing sky, sunbeams, rainbow, after-rain glints.

Conventions
-----------
* 1920x1080 screen px, y down.  Everything draws onto a cairo context.
* ``weather``: 0 = heavy rain, overcast  →  1 = after rain, sunny & sparkling (continuous).
* ``cam = (cx, cy, zoom)``: world point (cx, cy) is placed at the screen centre, scaled by zoom.
  CAM_ID = (960, 540, 1) is identity (world == screen).  ``to_screen(x, y, cam)`` / ``from_screen``.
* Deterministic: all randomness from fixed seeds; caches are pure functions of their keys
  (safe for parallel frame rendering).

See films/ame_robo/docs/world_api.md for the full API description.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from functools import lru_cache

import cairo
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from common import core
from common.core import W, H, TAU, clamp, lerp, smoothstep, mix, rgb, noise1

CAM_ID = (960.0, 540.0, 1.0)


# =========================================================================== small helpers
def _c(c):
    return rgb(c) if isinstance(c, str) else tuple(c)


def _set(ctx, c, a=1.0):
    c = _c(c)
    ctx.set_source_rgba(c[0], c[1], c[2], a)


def _mul(c, k):
    c = _c(c)
    if isinstance(k, (int, float)):
        return tuple(clamp(v * k) for v in c)
    return tuple(clamp(v * kk) for v, kk in zip(c, k))


def _hash(*a):
    n = 2166136261
    for v in a:
        n = ((n ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    n ^= n >> 15
    n = (n * 2246822519) & 0xFFFFFFFF
    n ^= n >> 13
    return n / 4294967295.0


def to_screen(x, y, cam=CAM_ID, depth=1.0):
    """World -> screen.  depth<1 = farther parallax layer (moves/zooms less)."""
    cx, cy, z = cam
    if depth != 1.0:
        cx = 960 + (cx - 960) * depth
        cy = 540 + (cy - 540) * depth
        z = 1 + (z - 1) * depth
    return (x - cx) * z + 960, (y - cy) * z + 540


def from_screen(sx, sy, cam=CAM_ID):
    cx, cy, z = cam
    return (sx - 960) / z + cx, (sy - 540) / z + cy


def _apply_cam(ctx, cam, depth=1.0):
    cx, cy, z = cam
    if depth != 1.0:
        cx = 960 + (cx - 960) * depth
        cy = 540 + (cy - 540) * depth
        z = 1 + (z - 1) * depth
    ctx.translate(960, 540)
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)


def _camkey(cam):
    return (round(cam[0] * 2) / 2, round(cam[1] * 2) / 2, round(cam[2], 4))


def _surf_from_rgb(arr, alpha=None):
    """float RGB (h,w,3) [+ alpha (h,w)] -> premultiplied cairo surface."""
    h, w = arr.shape[:2]
    a = np.ones((h, w), np.float32) if alpha is None else alpha.astype(np.float32)
    rgba = np.empty((h, w, 4), np.float32)
    rgba[..., :3] = arr * a[..., None]
    rgba[..., 3] = a
    return core.rgba_surface(rgba)


def _surf_array(s):
    return core.surface_rgba(s)


def _blur_surface(s, sx, sy=None, ds=None):
    """Gaussian blur of a cairo surface (anisotropic allowed); returns a new surface."""
    sy = sx if sy is None else sy
    a = core.surface_rgba(s)
    h, w = a.shape[:2]
    ds = ds or max(1, min(6, int(min(sx, sy) / 4)))
    out = np.empty_like(a)
    for ch in range(4):
        im = Image.fromarray(a[..., ch], mode="F")
        if ds > 1:
            im = im.resize((max(1, w // ds), max(1, h // ds)), Image.BILINEAR)
        b = gaussian_filter(np.asarray(im), (sy / ds, sx / ds), mode="nearest")
        im2 = Image.fromarray(b, mode="F")
        if ds > 1:
            im2 = im2.resize((w, h), Image.BILINEAR)
        out[..., ch] = np.asarray(im2)
    return core.rgba_surface(out)


def _rgba_rows(s, y0=0, y1=None):
    """Premultiplied float32 RGBA of a cairo surface, rows y0:y1 only (fast path)."""
    s.flush()
    h, w = s.get_height(), s.get_width()
    a = np.ndarray((h, w, 4), np.uint8, s.get_data(), strides=(s.get_stride(), 4, 1))[y0:y1]
    out = np.empty(a.shape, np.float32)
    out[..., 0] = a[..., 2]
    out[..., 1] = a[..., 1]
    out[..., 2] = a[..., 0]
    out[..., 3] = a[..., 3]
    out *= 1.0 / 255.0
    return out


def _ablur(a, sy, sx, ds=1):
    """Anisotropic gaussian blur of an (h,w,c) float array, computed at 1/ds resolution."""
    h, w, c = a.shape
    out = np.empty_like(a)
    for ch in range(c):
        x = a[..., ch]
        if ds > 1:
            x = np.asarray(Image.fromarray(x, mode="F").resize((max(1, w // ds), max(1, h // ds)), Image.BILINEAR))
        b = gaussian_filter(x, (sy / ds, sx / ds), mode="nearest")
        if ds > 1:
            b = np.asarray(Image.fromarray(b, mode="F").resize((w, h), Image.BILINEAR))
        out[..., ch] = b
    return out


def _rows_to_surface(arr, full_h, y0):
    """Place float RGBA rows (starting at y0) into a new full-height cairo surface."""
    h, w = arr.shape[:2]
    full = np.zeros((full_h, w, 4), np.float32)
    full[y0:y0 + h] = arr
    return core.rgba_surface(full)


class _LRU:
    def __init__(self, n):
        self.n, self.d = n, OrderedDict()

    def get(self, k, build):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        v = build()
        self.d[k] = v
        while len(self.d) > self.n:
            self.d.popitem(last=False)
        return v


# --------------------------------------------------------------------------- text
@lru_cache(maxsize=128)
def _text_surf(text, size, color="#FFFFFF", font="maru", vertical=False, stroke=0,
               stroke_color="#000000", pad=6, spacing=1.04):
    """Small cairo surface with (horizontal or vertical) text; returns (surface, w, h)."""
    f = core._font(font, size)
    c = tuple(int(v * 255) for v in rgb(color))
    sc = tuple(int(v * 255) for v in rgb(stroke_color))
    if vertical:
        cw = size + 2 * stroke
        hh = int(len(text) * size * spacing) + 2 * stroke
        im = Image.new("RGBA", (cw + 2 * pad, hh + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for i, ch in enumerate(text):
            d.text((pad + cw / 2, pad + stroke + (i + 0.5) * size * spacing), ch, font=f, fill=c,
                   anchor="mm", stroke_width=stroke, stroke_fill=sc)
    else:
        dummy = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        tw = int(dummy.textlength(text, font=f)) + 2 * stroke
        im = Image.new("RGBA", (tw + 2 * pad, int(size * 1.25) + 2 * stroke + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((im.width / 2, im.height / 2), text, font=f, fill=c, anchor="mm",
               stroke_width=stroke, stroke_fill=sc)
    a = np.asarray(im).astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return core.rgba_surface(a), im.width, im.height


def _paint_in_quad(ctx, surf, w, h, p0, p1, p3, alpha=1.0, inset=0.0):
    """Paint surface (w x h) affinely so its corners land on p0 (top-left), p1 (top-right),
    p3 (bottom-left)."""
    ctx.save()
    m = cairo.Matrix((p1[0] - p0[0]) / w, (p1[1] - p0[1]) / w, (p3[0] - p0[0]) / h,
                     (p3[1] - p0[1]) / h, p0[0], p0[1])
    try:
        ctx.transform(m)
    except cairo.Error:
        ctx.restore()
        return
    ctx.set_source_surface(surf, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint_with_alpha(alpha)
    ctx.restore()


# =========================================================================== palette
_RAIN = dict(
    sky_top="#76879A", sky_mid="#93A2B1", sky_hor="#B7C2CC", sun_glow="#C9D2DA",
    cloud_dark="#5A6878", cloud_mid="#7C8A99", cloud_lit="#AAB6C2", cloud_rim="#C3CDD6",
    haze="#A3B0BD", amb="#8C9BAA", street="#4E5A66", street_hi="#71808D", pave="#5D6872",
    pave_dry="#8A8F92", rain="#D8E3EC", warm="#FFC777", vend="#CFF3FF", sun="#E9EEF2",
    sky_refl="#A9B6C2", char_rim="#D3E1EE", shadow="#2E3843", glint="#E8F2FA",
)
_SUN = dict(
    sky_top="#5AAAE6", sky_mid="#7EC3F0", sky_hor="#D8F0FF", sun_glow="#FFF1C8",
    cloud_dark="#9DB5CE", cloud_mid="#DCE8F3", cloud_lit="#FFFFFF", cloud_rim="#FFF6DD",
    haze="#CBE4F4", amb="#BFDDF2", street="#55626E", street_hi="#A8C4D8", pave="#76818A",
    pave_dry="#C9C2B5", rain="#F2F8FF", warm="#FFE6A8", vend="#E4F8FF", sun="#FFE6A8",
    sky_refl="#8CC8F0", char_rim="#FFE6A8", shadow="#3C4A63", glint="#FFFFFF",
)


@lru_cache(maxsize=256)
def _palette_q(q):
    w = q / 100.0
    k = smoothstep(0.0, 1.0, w)
    C = {key: mix(_RAIN[key], _SUN[key], k) for key in _RAIN}
    C.update(
        weather=w,
        sunk=smoothstep(0.35, 0.95, w),          # direct sunlight strength
        lamp=1.0 - smoothstep(0.25, 0.9, w),     # artificial lights (shop/lamps) dominance
        rain_amt=clamp(1 - w / 0.8) ** 1.3,      # rain density
        fog=lerp(55.0, 160.0, smoothstep(0, 1, w)),  # fog distance (m)
        refl=lerp(0.62, 0.72, w),                # wet reflectivity
        ripple=lerp(1.0, 0.18, smoothstep(0.2, 0.9, w)),
        glints=smoothstep(0.55, 1.0, w),
        char_mul=tuple(lerp(a, b, k) for a, b in zip((0.80, 0.86, 0.94), (1.03, 1.0, 0.95))),
    )
    return C


def palette(weather):
    """Colour/lighting dict for a weather value (RGB tuples 0..1 + scalars).

    Colours: sky_top sky_mid sky_hor sun_glow cloud_dark cloud_mid cloud_lit cloud_rim haze amb street
    street_hi pave pave_dry rain warm vend sun sky_refl char_rim shadow glint.
    Scalars: weather, sunk (sun strength), lamp (artificial light), rain_amt (density 0..1), fog (m),
    refl, ripple, glints; char_mul = RGB multiply tint suggested for characters."""
    return _palette_q(int(round(clamp(weather) * 100)))


def _shade(C, base, face="F", z=None, k=1.0):
    """Surface colour for a material `base` on a face: F (towards camera), L (left-side facade,
    faces +x), R (right-side facade, faces -x), T (top), U (underside).  Sun from upper-left
    behind the camera.  z (m) applies distance fog."""
    b = _c(base)
    s = C["sunk"]
    amb = C["amb"]
    rain_f = {"F": 0.92, "L": 0.86, "R": 0.9, "T": 1.0, "U": 0.62}[face]
    sun_f = {"F": 0.98, "L": 0.80, "R": 1.04, "T": 1.06, "U": 0.62}[face]
    rc = tuple(lerp(bb, aa, 0.2) * rain_f * 0.8 * bl for bb, aa, bl in zip(b, amb, (0.93, 0.98, 1.04)))
    if face in ("F", "R", "T"):
        sc = tuple(bb * sun_f * ww for bb, ww in zip(b, (1.04, 1.0, 0.9)))
    else:
        sc = tuple(lerp(bb, aa, 0.18) * sun_f for bb, aa in zip(b, (0.55, 0.7, 0.95)))
    c = tuple(lerp(r, q, s) * k for r, q in zip(rc, sc))
    if z is not None:
        f = 1 - math.exp(-max(0.0, z - 4) / C["fog"])
        c = mix(c, C["haze"], f * 0.92)
    return tuple(clamp(v) for v in c)


# =========================================================================== noise textures
def _noise2(w, h, cell, rng):
    gw, gh = int(math.ceil(w / cell)), int(math.ceil(h / cell))
    g = rng.random((gh, gw)).astype(np.float32)
    g = np.pad(g, ((0, 1), (0, 1)), mode="wrap")
    ys = np.arange(h) / cell
    xs = np.arange(w) / cell
    yi, xi = ys.astype(int), xs.astype(int)
    fy, fx = ys - yi, xs - xi
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = g[yi][:, xi]
    b = g[yi][:, xi + 1]
    c = g[yi + 1][:, xi]
    d = g[yi + 1][:, xi + 1]
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return top + (bot - top) * fy[:, None]


@lru_cache(maxsize=4)
def _cloud_tex(seed=3, size=1024):
    """Periodic fbm texture (size x size) in 0..1."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), np.float32)
    amp, tot = 1.0, 0.0
    for cell in (256, 128, 64, 32, 16, 8):
        acc += _noise2(size, size, cell, rng) * amp
        tot += amp
        amp *= 0.52
    acc /= tot
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    return gaussian_filter(acc, 1.2, mode="wrap")


# =========================================================================== sky
SKY_F = 1100.0          # focal length used for the cloud-deck perspective
SKY_RES = 4             # sky computed at 1/SKY_RES resolution, upscaled
SUN_DIR = (-0.55, 0.9)  # sun is up-left (screen direction from horizon centre), used for cloud light
_SKY_CACHE = _LRU(3)


def _sky_array(t, weather, cam, horizon_y, depth):
    """Sky RGB array (H/SKY_RES, W/SKY_RES, 3) in screen space."""
    C = palette(weather)
    w = weather
    cx, cy, z = cam
    cx = 960 + (cx - 960) * depth
    cy = 540 + (cy - 540) * depth
    z = 1 + (z - 1) * depth
    hs, ws = H // SKY_RES, W // SKY_RES
    sx = (np.arange(ws, dtype=np.float32) * SKY_RES + SKY_RES / 2 - 960) / z + cx
    sy = (np.arange(hs, dtype=np.float32) * SKY_RES + SKY_RES / 2 - 540) / z + cy
    v = (horizon_y - sy)                 # px above horizon (world)
    # ------------------------------------------------ base gradient
    e = np.clip(v / 900.0, 0, 1)
    top, mid, hor = (np.array(C[k], np.float32) for k in ("sky_top", "sky_mid", "sky_hor"))
    e1 = (np.clip(e / 0.4, 0, 1) ** 0.8)[:, None]
    e2 = np.clip((e - 0.4) / 0.6, 0, 1)[:, None]
    base = hor + (mid - hor) * e1 + (top - mid) * e2
    base = base[:, None, :]
    base = np.broadcast_to(base, (hs, ws, 3)).copy()
    # sun glow (upper-left, mostly off-frame)
    sunk = C["sunk"]
    if sunk > 0.01:
        sxp = (sx[None, :] - (-260)) / 1500.0
        syp = (sy[:, None] - (horizon_y - 900)) / 1100.0
        g = np.exp(-(sxp ** 2 + syp ** 2) * 2.2) * sunk
        base += (np.array(C["sun_glow"], np.float32) - base) * (g[..., None] * 0.75)
    # ------------------------------------------------ cloud decks (perspective planes)
    tex = _cloud_tex()
    n = tex.shape[0]
    msk = n - 1
    out = base.astype(np.float32)
    r1 = int(np.sum(v > -10))          # only rows above the horizon get clouds
    if r1 <= 0:
        return np.clip(out, 0, 1)
    full = out
    out = full[:r1]
    vv = np.maximum(v[:r1], 4.0)[:, None]
    above = np.clip(v[:r1] / 30.0, 0, 1)[:, None]
    sxc = (sx[None, :] - 960).astype(np.float32)
    decks = [  # height (rel), texture scale, drift speed, coverage rain->sun, softness
        dict(hgt=2.2, K=46.0, sp=0.020, cov0=0.0, cov1=0.56, soft=0.16, off=(0, 0)),
        dict(hgt=1.0, K=80.0, sp=0.055, cov0=0.30, cov1=0.68, soft=0.12, off=(311, 517)),
    ]
    sunk = C["sunk"]
    hz = np.array(C["haze"], np.float32)
    for di, d in enumerate(decks):
        depthz = (d["hgt"] * SKY_F / vv).astype(np.float32)          # (hs,1)
        tu = ((sxc * (depthz / SKY_F) * d["K"]) + (t * d["sp"] * 300 + d["off"][0])).astype(np.int32) & msk
        tvr = (depthz * d["K"] - t * d["sp"] * 60 + d["off"][1]).astype(np.int32) & msk
        tv = np.broadcast_to(tvr, tu.shape)
        dens = tex[tv, tu]
        cov = lerp(d["cov0"], d["cov1"], smoothstep(0.05, 0.95, w))
        a = np.clip((dens - cov) * (1.0 / d["soft"]), 0, 1)
        a = a * a * (3 - 2 * a)
        lit = np.clip((dens - tex[(tv + 6) & msk, (tu - 8) & msk]) * 10 + 0.5, 0, 1)
        thick = np.clip((dens - cov) * (1 / 0.3), 0, 1)
        far = np.clip(1 - vv / 220.0, 0, 1)
        if di == 1:
            cdark = np.array(C["cloud_dark"], np.float32)
            cmid = np.array(C["cloud_mid"], np.float32)
        else:
            cdark = np.array(mix(C["cloud_dark"], C["cloud_mid"], 0.45), np.float32)
            cmid = np.array(mix(C["cloud_mid"], C["cloud_lit"], 0.35), np.float32)
        clit = np.array(C["cloud_lit"], np.float32)
        crim = np.array(C["cloud_rim"], np.float32)
        sh = (thick * (1 - 0.6 * lit))[..., None]
        body = cmid + (cdark - cmid) * sh
        body += (clit - body) * ((lit * lit) * (0.3 + 0.55 * sunk))[..., None]
        edge = ((1 - thick) * lit * (0.25 + 0.75 * sunk))[..., None]
        body += (crim - body) * (edge * 0.85)
        body += (hz - body) * (far * 0.8)[..., None]
        a = a * (1 - far * 0.3) * above
        out += (body - out) * a[..., None]
    out = full
    # rain veil: heavy rain flattens everything towards haze
    r = C["rain_amt"]
    if r > 0:
        hz = np.array(C["haze"], np.float32)
        veil = np.clip(0.18 * r + 0.25 * r * np.clip(1 - v / 500.0, 0, 1), 0, 0.5)[:, None, None]
        out = out + (hz - out) * veil
    return np.clip(out, 0, 1)


def sky_array(t, weather, cam=CAM_ID, horizon_y=540.0, depth=0.35):
    key = (round(t, 4), round(weather, 4), _camkey(cam), round(horizon_y, 1), depth)
    return _SKY_CACHE.get(key, lambda: _sky_array(t, weather, cam, horizon_y, depth))


def _paint_small(ctx, arr, alpha=None, flip_axis=None):
    s = _surf_from_rgb(arr, alpha)
    ctx.save()
    if flip_axis is not None:
        ctx.translate(0, 2 * flip_axis)
        ctx.scale(1, -1)
    ctx.scale(W / arr.shape[1], H / arr.shape[0])
    ctx.set_source_surface(s, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_BILINEAR)
    ctx.get_source().set_extend(cairo.EXTEND_PAD)
    ctx.paint()
    ctx.restore()


def draw_sky(ctx, t, weather, cam=CAM_ID, horizon_y=540.0, depth=0.35, sunbeams=True):
    """Full-screen sky: gradient + two perspective cloud decks drifting slowly.
    weather 0: low heavy overcast; 0.5: breaking, brighter; 1: blue sky with white cumulus.
    horizon_y = world y of the horizon (street: STREET_HORIZON_Y, shopfront: SHOP_HORIZON_Y).
    depth = parallax factor of the sky relative to cam (0 = fixed to screen)."""
    arr = sky_array(t, weather, cam, horizon_y, depth)
    _paint_small(ctx, arr)
    C = palette(weather)
    if sunbeams and C["sunk"] > 0.02:
        draw_sunbeams(ctx, t, C["sunk"], origin=to_screen(-300, horizon_y - 1050, cam, depth))


def _lowres(ctx, fn, k=3, op=None, alpha=1.0):
    """Run fn(ctx2) on a 1/k resolution transparent surface (same screen coords) and paint it
    upscaled onto ctx — for soft, large shapes (beams, rainbow)."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W // k + 2, H // k + 2)
    c2 = cairo.Context(s)
    c2.scale(1 / k, 1 / k)
    fn(c2)
    ctx.save()
    if op is not None:
        ctx.set_operator(op)
    ctx.scale(k, k)
    ctx.set_source_surface(s, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_BILINEAR)
    ctx.paint_with_alpha(alpha)
    ctx.restore()


def draw_sunbeams(ctx, t, strength=1.0, origin=(-300, -500), n=11, color="#FFF0C4", length=2600,
                  spread=(0.25, 1.05), seed=5):
    """Crepuscular rays fanning from `origin` (screen px, usually off-frame up-left).
    spread = angle range (rad, 0 = +x axis, pi/2 = down).  Screen-blended, slowly shimmering.
    Rendered at 1/4 resolution (soft)."""
    if strength <= 0:
        return
    _lowres(ctx, lambda c2: _sunbeams(c2, t, strength, origin, n, color, length, spread, seed), 4)


def _sunbeams(ctx, t, strength, origin, n, color, length, spread, seed):
    ox, oy = origin
    c = _c(color)
    ctx.save()
    for i in range(n):
        a0 = lerp(spread[0], spread[1], (i + 0.5 * _hash(seed, i)) / n)
        wdt = 0.012 + 0.05 * _hash(seed, i, 2)
        fl = 0.55 + 0.45 * noise1(t * 0.35 + i * 3.1, seed)
        a = strength * (0.05 + 0.10 * _hash(seed, i, 3)) * fl
        if a <= 0.003:
            continue
        L = length * (0.6 + 0.4 * _hash(seed, i, 4))
        g = cairo.LinearGradient(ox, oy, ox + math.cos(a0) * L, oy + math.sin(a0) * L)
        g.add_color_stop_rgba(0, *c, 0)
        g.add_color_stop_rgba(0.12, *c, a)
        g.add_color_stop_rgba(0.55, *c, a * 0.6)
        g.add_color_stop_rgba(1, *c, 0)
        ctx.move_to(ox, oy)
        ctx.line_to(ox + math.cos(a0 - wdt) * L, oy + math.sin(a0 - wdt) * L)
        ctx.line_to(ox + math.cos(a0 + wdt) * L, oy + math.sin(a0 + wdt) * L)
        ctx.close_path()
        ctx.set_source(g)
        ctx.fill()
    ctx.restore()


_RAINBOW = [(0.00, "#8A5BD6", 0.0), (0.10, "#7A62E0", 0.55), (0.24, "#4D86F0", 0.75),
            (0.40, "#4CC7C9", 0.75), (0.55, "#7AD66B", 0.8), (0.70, "#F6E25A", 0.85),
            (0.84, "#F7A04A", 0.85), (0.94, "#EE5A5A", 0.7), (1.00, "#EE5A5A", 0.0)]


def draw_rainbow(ctx, t, appear, center=(1250, 1150), radius=900, width=None, alpha=0.55,
                 secondary=True, fade_y=None):
    """Soft translucent rainbow arc.  appear 0..1 grows the arc from the left foot and fades it in.
    center/radius in screen px (centre usually below the horizon).  fade_y: screen y where the bow
    fades out towards its feet (default: centre y - 0.18 r).  Rendered at 1/2 resolution."""
    if appear <= 0.001:
        return
    _lowres(ctx, lambda c2: _rainbow(c2, t, appear, center, radius, width, alpha, secondary, fade_y), 2)


def _rainbow(ctx, t, appear, center, radius, width, alpha, secondary, fade_y):
    """Soft translucent rainbow arc.  appear 0..1 grows the arc from the left foot and fades it in.
    center/radius in screen px (centre usually below the horizon).  fade_y: screen y where the bow
    fades out towards its feet (default: centre y - 0.18 r)."""
    if appear <= 0.001:
        return
    cx, cy = center
    width = width or radius * 0.085
    fade_y = cy - radius * 0.18 if fade_y is None else fade_y
    a = alpha * smoothstep(0, 0.6, appear) * (0.92 + 0.08 * math.sin(t * 1.3))
    ctx.save()
    ctx.push_group()
    # sky inside the bow is a bit brighter
    g = cairo.RadialGradient(cx, cy, radius * 0.55, cx, cy, radius - width * 0.5)
    g.add_color_stop_rgba(0, 1, 1, 1, 0)
    g.add_color_stop_rgba(1, 1, 1, 1, 0.16)
    ctx.arc(cx, cy, radius - width * 0.5, 0, TAU)
    ctx.set_source(g)
    ctx.fill()
    bows = [(radius, width, 1.0, False)]
    if secondary:
        bows.append((radius * 1.3, width * 1.3, 0.2, True))
    for r, wd, k, rev in bows:
        r0, r1 = r - wd / 2, r + wd / 2
        g = cairo.RadialGradient(cx, cy, r0, cx, cy, r1)
        for off, col, al in _RAINBOW:
            g.add_color_stop_rgba(1 - off if rev else off, *rgb(col), al * k)
        ctx.set_line_width(wd)
        ctx.arc(cx, cy, r, 0, TAU)
        ctx.set_source(g)
        ctx.stroke()
    ctx.pop_group_to_source()
    # mask: angular growth from the left foot + vertical fade at the feet
    ctx.push_group()
    sweep = math.pi * 1.0 * smoothstep(0, 1, appear)
    ang0 = math.pi
    ctx.move_to(cx, cy)
    ctx.arc(cx, cy, radius * 1.6, ang0 - 0.2, ang0 + sweep + 0.001)
    ctx.close_path()
    fg = cairo.LinearGradient(0, fade_y, 0, cy - radius * 0.6)
    fg.add_color_stop_rgba(0, 1, 1, 1, 0)
    fg.add_color_stop_rgba(1, 1, 1, 1, a)
    ctx.set_source(fg)
    ctx.fill()
    mask = ctx.pop_group()
    ctx.mask(mask)  # source = the rainbow group
    ctx.restore()


# =========================================================================== rain
_RAIN_LAYERS = {
    #        n,   len,  width, speed(px/s), alpha, parallax, blur-ish halo
    "bg":  (1500, 34, 1.0, 1500, 0.30, 0.25, 0.0),
    "mid": (520, 70, 1.6, 2300, 0.42, 0.6, 0.0),
    "fg":  (55, 210, 4.5, 3400, 0.30, 1.4, 1.0),
}


@lru_cache(maxsize=16)
def _rain_seeds(layer, seed):
    n = _RAIN_LAYERS[layer][0]
    rng = np.random.default_rng(seed * 31 + {"bg": 1, "mid": 2, "fg": 3}[layer])
    return (rng.random(n), rng.random(n), rng.random(n), rng.random(n))


def draw_rain(ctx, t, weather, layer="mid", cam=CAM_ID, wind=0.2, seed=1, density=1.0,
              color=None, region=None, dof=0.0):
    """Slanted rain streaks in screen space.  layer 'bg' (fine, far), 'mid', 'fg' (big, fast, sparse,
    soft).  Density ~ palette(weather)['rain_amt'] * density; stops completely around weather 0.8.
    wind = horizontal slant (dx/dy).  cam moves the rain field with parallax.  region=(x0,y0,x1,y1)
    restricts the field (e.g. only outside an umbrella).  dof>0 softens (close-up)."""
    C = palette(weather)
    dens = C["rain_amt"] * density
    if dens <= 0.002:
        return
    n0, L0, wd0, sp0, al0, par, halo = _RAIN_LAYERS[layer]
    rx, ry, rl, ra = _rain_seeds(layer, seed)
    n = int(n0 * min(1.0, dens) + 0.5)
    cx, cy, z = cam
    offx = -(cx - 960) * par * z
    offy = -(cy - 540) * par * z
    zz = 1 + (z - 1) * par
    col = _c(color or C["rain"])
    span_w, span_h = W + 400, H + 600
    x0r, y0r, x1r, y1r = region or (-200, -300, W + 200, H + 300)
    sp = sp0 * zz
    L = L0 * zz
    vx, vy = wind, 1.0
    nrm = math.hypot(vx, vy)
    vx, vy = vx / nrm, vy / nrm
    ctx.save()
    if region is not None:
        ctx.rectangle(x0r, y0r, x1r - x0r, y1r - y0r)
        ctx.clip()
    rx, ry, rl, ra = rx[:n], ry[:n], rl[:n], ra[:n]
    ln = L * (0.6 + 0.8 * rl)
    yy = (ry * span_h + t * sp * (0.85 + 0.3 * rl) + offy) % span_h - 300
    xx = (rx * span_w + offx + (yy + 300) * vx / vy) % span_w - 200
    # gusts: density varies in drifting sheets
    gx = xx * 0.004 - t * 0.6
    g = 0.55 + 0.45 * (0.6 * np.sin(gx * 2.1 + seed) * np.sin(gx * 0.77 + 1.3 * seed) + 0.4 * np.sin(gx * 3.3 - 0.5))
    keep = ra <= g + 0.25
    nb = 3
    bucket = np.minimum(nb - 1, (ra * nb).astype(int))
    wd = wd0 * zz * (1 + dof * 1.5)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    fine = layer == "bg"
    for b in range(nb):
        sel = keep & (bucket == b)
        if not sel.any():
            continue
        X0, Y0, LN = xx[sel], yy[sel], ln[sel]
        a = al0 * (0.45 + 0.55 * (b + 0.5) / nb) * min(1.0, 0.35 + dens) / (1 + dof * 1.2)
        hx, hy = X0 - vx * LN * 0.5, Y0 - vy * LN * 0.5
        tx, ty = X0 - vx * LN, Y0 - vy * LN
        if halo > 0 or dof > 0:
            ctx.set_line_width(wd * 2.6)
            for p in zip(X0.tolist(), Y0.tolist(), tx.tolist(), ty.tolist()):
                ctx.move_to(p[0], p[1])
                ctx.line_to(p[2], p[3])
            _set(ctx, col, a * 0.22)
            ctx.stroke()
        if fine:
            ctx.set_line_width(wd)
            for p in zip(X0.tolist(), Y0.tolist(), tx.tolist(), ty.tolist()):
                ctx.move_to(p[0], p[1])
                ctx.line_to(p[2], p[3])
            _set(ctx, col, a * 0.8)
            ctx.stroke()
            continue
        ctx.set_line_width(wd * 0.8)
        for p in zip(hx.tolist(), hy.tolist(), tx.tolist(), ty.tolist()):
            ctx.move_to(p[0], p[1])
            ctx.line_to(p[2], p[3])
        _set(ctx, col, a * 0.45)
        ctx.stroke()
        ctx.set_line_width(wd)
        for p in zip(X0.tolist(), Y0.tolist(), hx.tolist(), hy.tolist()):
            ctx.move_to(p[0], p[1])
            ctx.line_to(p[2], p[3])
        _set(ctx, col, a)
        ctx.stroke()
    ctx.restore()


def _pt_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def _pip_np(x, y, poly):
    """Vectorised point-in-polygon (even-odd)."""
    inside = np.zeros(x.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[i - 1]
        cond = (y1 > y) != (y2 > y)
        xint = (x2 - x1) * (y - y1) / ((y2 - y1) if y2 != y1 else 1e-9) + x1
        inside ^= cond & (x < xint)
    return inside


def draw_splashes(ctx, t, weather, region=(0, 900, W, H), seed=2, density=1.0, horizon_y=None,
                  ground_y_fn=None, color=None, scale=1.0, rings=True, crowns=True):
    """Crown splashes + expanding ripple rings where drops hit the ground.
    region: (x0,y0,x1,y1) rectangle or a polygon [(x,y),...] in screen px.
    horizon_y: if given, sizes shrink towards the horizon (perspective).
    ground_y_fn(x) -> y: scatter along a ground line instead (y = fn(x) + 0..(y1-y0) band)."""
    C = palette(weather)
    dens = C["rain_amt"] * density
    if dens <= 0.002:
        return
    col = _c(color or C["rain"])
    poly = None
    if isinstance(region[0], (tuple, list)):
        poly = [tuple(p) for p in region]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0, y0, x1, y1 = max(-50, min(xs)), max(-50, min(ys)), min(W + 50, max(xs)), min(H + 50, max(ys))
    else:
        x0, y0, x1, y1 = region
    area = max(1.0, (x1 - x0) * (y1 - y0))
    rate = 0.00008 * area * dens        # events per 1/24 s
    slot = 1 / 24.0
    life = 0.45
    k0 = int(math.floor((t - life) / slot))
    k1 = int(math.floor(t / slot))
    ev = []
    for k in range(k0, k1 + 1):
        rng = np.random.default_rng([seed & 0xFFFF, k & 0x7FFFFFFF])
        cnt = rng.poisson(rate)
        if cnt == 0:
            continue
        u = rng.random((cnt, 5))
        ev.append(np.column_stack([k * slot + u[:, 0] * slot, u[:, 1:]]))
    if not ev:
        return
    ev = np.concatenate(ev)
    age = t - ev[:, 0]
    ok = (age >= 0) & (age <= life)
    ev, age = ev[ok], age[ok]
    xs = x0 + (x1 - x0) * ev[:, 1]
    if ground_y_fn is not None:
        ys = np.array([ground_y_fn(x) for x in xs]) + ev[:, 2] * (y1 - y0)
    else:
        ys = y0 + (y1 - y0) * ev[:, 2]
    if poly is not None:
        ok = _pip_np(xs, ys, poly)
        xs, ys, ev, age = xs[ok], ys[ok], ev[ok], age[ok]
    if horizon_y is not None:
        ss = scale * np.clip((ys - horizon_y) / max(1.0, (H - horizon_y)), 0.08, 1.6)
    else:
        ss = np.full(xs.shape, float(scale))
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ps = age / life
    if rings:
        # bucket by alpha for batching
        for bkt in range(4):
            sel = np.nonzero((np.minimum(3, (ps * 4).astype(int)) == bkt))[0]
            if len(sel) == 0:
                continue
            pm = (bkt + 0.5) / 4
            for i in sel.tolist():
                x, y, s, p = xs[i], ys[i], ss[i], ps[i]
                rr = (6 + 40 * p ** 0.6) * s
                ctx.save()
                ctx.translate(x, y)
                ctx.scale(1, 0.32)
                ctx.new_sub_path()
                ctx.arc(0, 0, rr, 0, TAU)
                if p > 0.25:
                    ctx.new_sub_path()
                    ctx.arc(0, 0, rr * 0.55, 0, TAU)
                ctx.restore()
            ctx.set_line_width(max(0.7, 1.4 * float(np.median(ss[sel]))))
            _set(ctx, col, 0.4 * (1 - pm) ** 1.5)
            ctx.stroke()
    if crowns:
        sel = np.nonzero(ps < 0.42)[0]
        for i in sel.tolist():
            x, y, s = xs[i], ys[i], ss[i]
            q = ps[i] / 0.42
            hgt = 18 * s * math.sin(q * math.pi) * (0.7 + 0.6 * ev[i, 3])
            spr = 11 * s * q ** 0.7
            ctx.set_line_width(max(0.6, 1.3 * s))
            _set(ctx, col, 0.38 * (1 - q))
            for d in (-1, -0.45, 0.45, 1):
                ctx.move_to(x + d * spr * 0.4, y)
                ctx.curve_to(x + d * spr * 0.6, y - hgt * 0.8, x + d * spr, y - hgt,
                             x + d * spr * 1.1, y - hgt * 0.6)
            ctx.stroke()
            for d in (-1, 1, -0.3, 0.3):
                ctx.new_sub_path()
                ctx.arc(x + d * spr * 1.25, y - hgt * (1.15 if abs(d) < 1 else 0.85) - 3 * s * q,
                        max(0.7, 1.6 * s), 0, TAU)
            ctx.fill()
    ctx.restore()


def _drop(ctx, x, y, r, vy, col, a, sparkle=0.0):
    """A falling droplet elongated by its speed, with a highlight."""
    st = clamp(vy / 900.0, 0, 3.0)
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(1, 1 + st)
    ctx.arc(0, 0, r, 0, TAU)
    ctx.restore()
    _set(ctx, col, a)
    ctx.fill()
    if sparkle > 0:
        draw_glint(ctx, x, y, r * 6 * sparkle, a * sparkle)
    else:
        ctx.arc(x - r * 0.35, y - r * 0.3, r * 0.35, 0, TAU)
        _set(ctx, (1, 1, 1), a * 0.8)
        ctx.fill()


def draw_drips_from(ctx, t, points, ground_y=None, period=(0.7, 1.6), size=4.0, gravity=2600.0,
                    color=None, weather=0.0, seed=4, alpha=0.85, splash=True, sparkle=None,
                    rate=None):
    """Generic drips: at each point (x, y[, ground_y_i]) a bead swells, detaches, falls with gravity
    and (optionally) splashes at its ground y.  period = (min,max) seconds between drops per point;
    after the rain (weather>0.6) drips slow down and sparkle in the sun.  rate scales frequency."""
    C = palette(weather)
    col = _c(color or C["rain"])
    if rate is None:
        rate = lerp(1.0, 0.35, smoothstep(0.3, 1.0, weather))
    if rate <= 0:
        return
    spk = C["glints"] if sparkle is None else sparkle
    ctx.save()
    for i, p in enumerate(points):
        x, y = p[0], p[1]
        gy = p[2] if len(p) > 2 else ground_y
        per = lerp(period[0], period[1], _hash(seed, i)) / rate
        ph = _hash(seed, i, 7) * per
        lt = (t + ph) % per
        k = math.floor((t + ph) / per)
        grow = per * 0.7
        r = size * (0.7 + 0.5 * _hash(seed, i, k))
        if lt < grow:  # bead forming
            g = smoothstep(0, 1, lt / grow)
            rr = r * (0.35 + 0.75 * g)
            ctx.save()
            ctx.translate(x, y + rr * 0.6 * g)
            ctx.scale(1, 1 + 0.35 * g)
            ctx.arc(0, 0, rr, 0, TAU)
            ctx.restore()
            _set(ctx, col, alpha * 0.85)
            ctx.fill()
            ctx.arc(x - rr * 0.3, y + rr * 0.4 * g, rr * 0.3, 0, TAU)
            _set(ctx, (1, 1, 1), alpha * 0.7)
            ctx.fill()
            # previous drop may still be falling
            ft = lt + per - grow
        else:
            ft = lt - grow
            # tiny residual bead
            ctx.arc(x, y + 1, r * 0.3, 0, TAU)
            _set(ctx, col, alpha * 0.6)
            ctx.fill()
        if ft >= 0:
            yy = y + r + 0.5 * gravity * ft * ft
            vy = gravity * ft
            if gy is None or yy < gy:
                if gy is None and yy > H + 50:
                    continue
                _drop(ctx, x, yy, r * 0.8, vy, col, alpha, sparkle=spk * (0.6 + 0.4 * math.sin(t * 9 + i)))
            elif splash and gy is not None:
                tland = math.sqrt(2 * max(1.0, gy - y - r) / gravity)
                age = ft - tland
                if 0 <= age < 0.35:
                    q = age / 0.35
                    ctx.save()
                    ctx.translate(x, gy)
                    ctx.scale(1, 0.33)
                    ctx.arc(0, 0, 5 + 34 * q ** 0.6 * size / 4, 0, TAU)
                    ctx.restore()
                    ctx.set_line_width(1.5)
                    _set(ctx, col, 0.5 * (1 - q))
                    ctx.stroke()
                    if q < 0.5:
                        for d in (-1, 1, -0.4, 0.4):
                            hx = x + d * 14 * q * size / 4
                            hy = gy - math.sin(q / 0.5 * math.pi) * 16 * size / 4 * (1.2 - abs(d) * 0.4)
                            ctx.arc(hx, hy, 1.6 * size / 4, 0, TAU)
                            _set(ctx, col, 0.7 * (1 - q))
                            ctx.fill()
    ctx.restore()


def draw_awning_drips(ctx, t, weather, x0, x1, y, ground_y=None, spacing=38, seed=11, size=4.0,
                      streams=True, edge_fn=None, color=None):
    """Drips forming and falling off a horizontal edge (x0..x1 at y, or y = edge_fn(x)).
    Heavy rain also pours a few thin continuous streams from low points.  ground_y: splash line."""
    C = palette(weather)
    n = max(1, int((x1 - x0) / spacing))
    pts = []
    for i in range(n):
        x = lerp(x0, x1, (i + 0.5 + 0.35 * (_hash(seed, i) - 0.5)) / n)
        yy = edge_fn(x) if edge_fn else y
        pts.append((x, yy))
    heavy = C["rain_amt"]
    draw_drips_from(ctx, t, pts, ground_y=ground_y, period=(lerp(0.35, 0.9, 1 - heavy), lerp(0.9, 2.4, 1 - heavy)),
                    size=size, weather=weather, seed=seed, color=color)
    if streams and heavy > 0.15 and ground_y is not None:
        col = _c(color or C["rain"])
        ns = max(1, n // 7)
        for j in range(ns):
            x = lerp(x0, x1, (j + 0.5 + 0.4 * (_hash(seed, j, 5) - 0.5)) / ns)
            yy = edge_fn(x) if edge_fn else y
            a = 0.42 * heavy
            ctx.save()
            ctx.set_line_width(2.6 * heavy + 0.6)
            ctx.move_to(x, yy)
            steps = 18
            for s in range(1, steps + 1):
                q = s / steps
                py = lerp(yy, ground_y, q)
                px = x + 2.2 * math.sin(py * 0.05 - t * 30 + j) * q
                ctx.line_to(px, py)
            g = cairo.LinearGradient(0, yy, 0, ground_y)
            c = col
            g.add_color_stop_rgba(0, *c, a)
            g.add_color_stop_rgba(0.5, *c, a * 0.7)
            g.add_color_stop_rgba(1, *c, a * 0.35)
            ctx.set_source(g)
            ctx.stroke()
            # beads riding the stream
            for b in range(5):
                q = ((t * 2.4 + b / 5 + _hash(seed, j, b)) % 1.0)
                py = lerp(yy, ground_y, q * q)
                ctx.arc(x + 2.2 * math.sin(py * 0.05 - t * 30 + j) * q, py, 2.2, 0, TAU)
                _set(ctx, c, a * 1.4)
                ctx.fill()
            # splash at the bottom
            ph = (t * 7 + j) % 1.0
            ctx.save()
            ctx.translate(x, ground_y)
            ctx.scale(1, 0.3)
            ctx.arc(0, 0, 8 + 26 * ph, 0, TAU)
            ctx.restore()
            ctx.set_line_width(1.5)
            _set(ctx, c, 0.5 * (1 - ph) * heavy)
            ctx.stroke()
            ctx.restore()


def draw_canopy_rain(ctx, t, center, radius, weather=0.0, ry=None, ribs=8, ground_y=None,
                     rim_angles=(0.08, math.pi - 0.08), seed=21, intensity=1.0, color=None,
                     hits=True, tilt=0.0):
    """Water sheeting off the rim of a big umbrella (s04).
    center=(cx,cy) screen px of the canopy's rim ellipse centre, radius = rim half-width,
    ry = rim ellipse half-height (default 0.28 r).  Rib tips are spaced evenly over the visible
    (front) half of the rim between rim_angles (0 = right, pi = left, measured clockwise/down).
    Draws: drops bouncing on the canopy top, thin sheets/streams from each rib tip (scallop),
    rim drips, and splashes on ground_y.  tilt (rad) rotates the whole canopy."""
    C = palette(weather)
    heavy = C["rain_amt"] * intensity
    if heavy <= 0.003:
        return
    col = _c(color or C["rain"])
    cx, cy = center
    ry = radius * 0.28 if ry is None else ry
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(tilt)
    # rib-tip points along the front rim
    tips = []
    for i in range(ribs + 1):
        a = lerp(rim_angles[0], rim_angles[1], i / ribs)
        tips.append((math.cos(a) * radius, math.sin(a) * ry))
    gy = (ground_y - cy) if ground_y is not None else radius * 1.6
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    # 1) streams from scallop low points (between rib tips) -> sheets of water
    for i in range(ribs):
        (ax, ay), (bx, by) = tips[i], tips[i + 1]
        mx, my = (ax + bx) / 2, (ay + by) / 2 + ry * 0.12
        sheet_w = abs(bx - ax) * 0.35
        for s in range(3):
            off = (s - 1) * sheet_w * 0.4 + 2 * math.sin(t * 5 + i + s)
            x0 = mx + off
            a = 0.35 * heavy * (0.7 + 0.3 * _hash(seed, i, s))
            ctx.set_line_width((1.4 + 1.6 * heavy) * (1.4 if s == 1 else 0.9))
            ctx.move_to(x0, my)
            n = 14
            for k in range(1, n + 1):
                q = k / n
                py = lerp(my, gy, q)
                px = x0 + math.sin(py * 0.04 - t * 26 + i * 1.7 + s) * 2.5 * q + off * q * 0.25
                ctx.line_to(px, py)
            g = cairo.LinearGradient(0, my, 0, gy)
            g.add_color_stop_rgba(0, *col, a * 1.2)
            g.add_color_stop_rgba(0.6, *col, a * 0.6)
            g.add_color_stop_rgba(1, *col, a * 0.25)
            ctx.set_source(g)
            ctx.stroke()
            # beads riding down (break-up)
            for b in range(4):
                q = (t * (2.0 + 0.4 * s) + b / 4 + _hash(seed, i, s, b)) % 1.0
                py = lerp(my, gy, q * q)
                px = x0 + math.sin(py * 0.04 - t * 26 + i * 1.7 + s) * 2.5 * q * q + off * q * q * 0.25
                ctx.arc(px, py, 1.6 + 1.6 * heavy, 0, TAU)
                _set(ctx, col, a * 1.5)
                ctx.fill()
        # splash at ground under each sheet
        if ground_y is not None:
            for s in range(2):
                ph = (t * 6 + _hash(seed, i, s, 9)) % 1.0
                ctx.save()
                ctx.translate(mx + (s - 0.5) * sheet_w * 0.5, gy)
                ctx.scale(1, 0.3)
                ctx.arc(0, 0, 6 + 30 * ph, 0, TAU)
                ctx.restore()
                ctx.set_line_width(1.6)
                _set(ctx, col, 0.55 * (1 - ph) * heavy)
                ctx.stroke()
                if ph < 0.4:
                    q = ph / 0.4
                    for d in (-1, -0.4, 0.4, 1):
                        ctx.arc(mx + d * 18 * q, gy - math.sin(q * math.pi) * 22 * (1.1 - abs(d) * 0.4),
                                1.8, 0, TAU)
                        _set(ctx, col, 0.7 * (1 - q) * heavy)
                        ctx.fill()
    # 2) heavier drips from each rib tip
    pts = [(x, y + 3) for x, y in tips]
    ctx.restore()
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(tilt)
    draw_drips_from(ctx, t, pts, ground_y=gy, period=(0.25, 0.5), size=4.5, weather=weather,
                    seed=seed + 1, color=color, rate=1.0, sparkle=0.0)
    # 3) drops hitting the canopy top: little bounces on the upper half of the ellipse
    if hits:
        nh = int(26 * heavy)
        for k in range(nh):
            per = 0.28
            ph = _hash(seed, k, 3) * per
            lt = (t + ph) % per
            kk = math.floor((t + ph) / per)
            u = _hash(seed, k, kk) * 2 - 1
            v = _hash(seed, k, kk, 1)
            hx = u * radius * 0.92
            # canopy dome top: above the rim ellipse
            hy = -math.sqrt(max(0.0, 1 - u * u)) * ry * (0.2 + 1.8 * v) - ry * 0.1
            q = lt / per
            ctx.set_line_width(1.4)
            _set(ctx, col, 0.65 * (1 - q))
            for d in (-1, 1):
                ctx.move_to(hx, hy)
                ctx.line_to(hx + d * 9 * q, hy - 12 * math.sin(q * math.pi))
            ctx.stroke()
            ctx.arc(hx, hy - 15 * math.sin(q * math.pi) - 3, 1.4, 0, TAU)
            _set(ctx, col, 0.6 * (1 - q))
            ctx.fill()
        # misty spray halo above the canopy
        g = cairo.RadialGradient(0, -ry * 0.6, radius * 0.2, 0, -ry * 0.6, radius * 1.1)
        g.add_color_stop_rgba(0, *col, 0.0)
        g.add_color_stop_rgba(0.7, *col, 0.07 * heavy)
        g.add_color_stop_rgba(1, *col, 0.0)
        ctx.save()
        ctx.scale(1, 0.45)
        ctx.arc(0, -ry * 1.2, radius * 1.1, 0, TAU)
        ctx.restore()
        ctx.set_source(g)
        ctx.fill()
    ctx.restore()


# =========================================================================== glints & puddles
def draw_glint(ctx, x, y, size, alpha=1.0, color="#FFFFFF", rot=0.0, glow=True):
    """4-point star flare (sun sparkle on water)."""
    if alpha <= 0.01 or size <= 0.3:
        return
    c = _c(color)
    ctx.save()
    ctx.translate(x, y)
    if glow:
        g = cairo.RadialGradient(0, 0, 0, 0, 0, size * 0.8)
        g.add_color_stop_rgba(0, *c, 0.55 * alpha)
        g.add_color_stop_rgba(1, *c, 0)
        ctx.arc(0, 0, size * 0.8, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
    ctx.rotate(rot)
    for k, (L, wd) in enumerate(((size, size * 0.075), (size * 0.5, size * 0.06))):
        ctx.save()
        ctx.rotate(k * math.pi / 4)
        ctx.move_to(0, -L)
        ctx.curve_to(wd * 0.3, -wd, wd, -wd * 0.3, L, 0)
        ctx.curve_to(wd, wd * 0.3, wd * 0.3, wd, 0, L)
        ctx.curve_to(-wd * 0.3, wd, -wd, wd * 0.3, -L, 0)
        ctx.curve_to(-wd, -wd * 0.3, -wd * 0.3, -wd, 0, -L)
        ctx.close_path()
        _set(ctx, c, alpha * (1 if k == 0 else 0.7))
        ctx.fill()
        ctx.restore()
    ctx.arc(0, 0, size * 0.1, 0, TAU)
    _set(ctx, (1, 1, 1), alpha)
    ctx.fill()
    ctx.restore()


def draw_glints(ctx, t, weather, region=(0, 850, W, H), n=40, seed=8, size=14.0, horizon_y=None,
                color=None, strength=1.0):
    """Twinkling sun-glints on wet surfaces; fades in as weather -> 1 (palette 'glints').
    region rect or polygon (screen px); horizon_y scales size with perspective."""
    C = palette(weather)
    k = C["glints"] * strength
    if k <= 0.01:
        return
    poly = None
    if isinstance(region[0], (tuple, list)):
        poly = region
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    else:
        x0, y0, x1, y1 = region
    col = color or C["glint"]
    cnt = 0
    i = 0
    while cnt < n and i < n * 6:
        i += 1
        x = lerp(x0, x1, _hash(seed, i, 1))
        y = lerp(y0, y1, _hash(seed, i, 2) ** 0.8)
        if poly is not None and not _pt_in_poly(x, y, poly):
            continue
        cnt += 1
        per = 1.2 + 2.2 * _hash(seed, i, 3)
        ph = (t / per + _hash(seed, i, 4)) % 1.0
        tw = max(0.0, math.sin(ph * math.pi)) ** 6
        s = size * (0.5 + 0.8 * _hash(seed, i, 5))
        if horizon_y is not None:
            s *= clamp((y - horizon_y) / max(1.0, H - horizon_y), 0.15, 1.5)
        draw_glint(ctx, x, y, s * (0.4 + 0.6 * tw), k * (0.25 + 0.75 * tw), col,
                   rot=0.2 * _hash(seed, i, 6))


def _blob_path(ctx, x, y, w, h, seed, n=14, wob=0.22):
    pts = []
    for i in range(n):
        a = i / n * TAU
        r = 1 + wob * (noise1(i * 0.9, seed) * 0.8 + 0.2 * math.sin(a * 3 + seed))
        pts.append((x + math.cos(a) * w * 0.5 * r, y + math.sin(a) * h * 0.5 * r))
    # smooth closed curve (Catmull-Rom -> Bezier)
    ctx.move_to(*pts[0])
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        ctx.curve_to(p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6,
                     p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6, p2[0], p2[1])
    ctx.close_path()


def draw_puddle(ctx, x, y, w, h, t, reflect_fn=None, weather=0.0, seed=1, ripples=True,
                sky=True, alpha=1.0):
    """A puddle (irregular soft ellipse) centred (x, y), size w x h screen px.
    reflect_fn(ctx): called clipped to the puddle with a vertical mirror about y already applied —
    so draw the scene normally (screen coords) and it appears reflected.  Without it, the puddle
    reflects a sky gradient.  Rain ripples when weather < 0.8; a bright rim when sunny."""
    C = palette(weather)
    ctx.save()
    _blob_path(ctx, x, y, w, h, seed)
    path = ctx.copy_path()
    ctx.clip()
    if sky:
        g = cairo.LinearGradient(0, y - h / 2, 0, y + h / 2)
        g.add_color_stop_rgba(0, *mix(C["sky_refl"], C["street"], 0.15), alpha)
        g.add_color_stop_rgba(1, *mix(C["sky_refl"], C["street"], 0.55), alpha)
        ctx.set_source(g)
        ctx.paint()
    if reflect_fn is not None:
        ctx.save()
        ctx.translate(0, 2 * y)
        ctx.scale(1, -1)
        reflect_fn(ctx)
        ctx.restore()
    if ripples and C["rain_amt"] > 0.01:
        for i in range(int(3 + 5 * C["rain_amt"])):
            per = 0.6 + 0.5 * _hash(seed, i)
            ph = (t / per + _hash(seed, i, 1)) % 1.0
            k = math.floor(t / per + _hash(seed, i, 1))
            rx = x + (_hash(seed, i, k) - 0.5) * w * 0.8
            ryy = y + (_hash(seed, i, k, 2) - 0.5) * h * 0.6
            ctx.save()
            ctx.translate(rx, ryy)
            ctx.scale(1, h / w * 1.1)
            ctx.arc(0, 0, 4 + w * 0.12 * ph, 0, TAU)
            ctx.restore()
            ctx.set_line_width(1.3)
            _set(ctx, C["rain"], 0.45 * (1 - ph) * alpha)
            ctx.stroke()
    ctx.restore()
    # rim: darker wet edge + highlight on the far edge
    ctx.save()
    ctx.append_path(path)
    ctx.set_line_width(max(1.0, h * 0.05))
    _set(ctx, C["shadow"], 0.25 * alpha)
    ctx.stroke()
    ctx.restore()


# =========================================================================== 3D projection helper
class _Proj:
    """Shift-lens perspective (verticals stay vertical, one-point perspective along +Z).
    X right (m), Y up (m), Z forward (m).  mirror=True projects the reflection in the ground plane."""

    def __init__(self, f, vpx, vpy, eye, cam_x=0.0, mirror=False):
        self.f, self.vpx, self.vpy, self.eye, self.cx, self.mirror = f, vpx, vpy, eye, cam_x, mirror

    def __call__(self, X, Y, Z):
        if self.mirror:
            Y = -Y
        Z = max(Z, 0.3)
        return (self.vpx + self.f * (X - self.cx) / Z, self.vpy + self.f * (self.eye - Y) / Z)

    def s(self, Z):
        return self.f / max(Z, 0.3)

    def poly(self, ctx, pts):
        x, y = self(*pts[0])
        ctx.move_to(x, y)
        for p in pts[1:]:
            ctx.line_to(*self(*p))
        ctx.close_path()

    def line(self, ctx, pts):
        ctx.move_to(*self(*pts[0]))
        for p in pts[1:]:
            ctx.line_to(*self(*p))


def _fillp(ctx, P, pts, col, a=1.0):
    P.poly(ctx, pts)
    _set(ctx, col, a)
    ctx.fill()


def _quadX(X, y0, y1, z0, z1):
    """Quad on a plane X = const (side facades)."""
    return [(X, y0, z0), (X, y0, z1), (X, y1, z1), (X, y1, z0)]


def _quadZ(Z, x0, x1, y0, y1):
    """Quad on a plane Z = const (faces the camera)."""
    return [(x0, y0, Z), (x1, y0, Z), (x1, y1, Z), (x0, y1, Z)]


def _text_on_quad(ctx, P, text, pts_tl_tr_bl, size_px, color, vertical=False, font="maru", alpha=1.0,
                  stroke=0, stroke_color="#000000", fit=0.86):
    """Draw text inside a 3D quad given by top-left, top-right, bottom-left 3D points."""
    tl, tr, bl = (P(*p) for p in pts_tl_tr_bl)
    size = int(max(8, min(160, size_px)))
    surf, w, h = _text_surf(text, size, color, font, vertical, stroke, stroke_color)
    # fit text box into quad, centred
    qw = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    qh = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    if qw < 3 or qh < 3:
        return
    k = min(qw * fit / w, qh * fit / h)
    ux, uy = (tr[0] - tl[0]) / qw, (tr[1] - tl[1]) / qw
    vx, vy = (bl[0] - tl[0]) / qh, (bl[1] - tl[1]) / qh
    ox = (qw - w * k) / 2
    oy = (qh - h * k) / 2
    p0 = (tl[0] + ux * ox + vx * oy, tl[1] + uy * ox + vy * oy)
    p1 = (p0[0] + ux * w * k, p0[1] + uy * w * k)
    p3 = (p0[0] + vx * h * k, p0[1] + vy * h * k)
    _paint_in_quad(ctx, surf, w, h, p0, p1, p3, alpha)


# =========================================================================== street (s01 / wide s06)
STREET_F = 1450.0
STREET_HORIZON_Y = 290.0          # world y of horizon / vanishing point
STREET_VP = (960.0, STREET_HORIZON_Y)
STREET_EYE = 5.6                  # camera height (m)
STREET_PLATE = (-220.0, -760.0, 2140.0, 1280.0)   # cached world rect
STREET_SCALE = 1.2                # plate resolution (supports zoom up to ~1.2 crisply)
STREET_CAM_HIGH = (960.0, 540.0, 1.0)     # establishing high-angle view (s01 end / s06 wide)
STREET_CAM_START = (960.0, -170.0, 1.1)   # crane start: up in the rain above the wires & rooftops
STREET_FX = 3.6                   # facade distance from street centre (m)
STREET_EDGE = 3.0                 # pavement / gutter line (m)

_WALLS = ["#E6DCC6", "#A9C7BF", "#8E6E58", "#C2BDB3", "#C4DACB", "#DDAE95", "#9DAFBE", "#77594A",
          "#EDE7DA", "#D9C9A3", "#B7A6C4", "#9FB59B"]
_AWN = [("#C94A43", "#F3E8D2"), ("#3F8F6B", "#F1EEE4"), ("#3E6FA8", "#F1EEE4"), ("#E08A3C", "#F6E9CF"),
        ("#2F8C8C", "#F3E8D2"), ("#D9A93B", "#7A4E36"), ("#B5466E", "#F6E9E4"), ("#55657E", "#E8E2D2")]
_HSIGNS = [  # text, box colour, text colour, lit
    ("だがし", "#F4EFE3", "#C8323A", True), ("パン", "#E3A43C", "#FFFFFF", True),
    ("くすり", "#FFFFFF", "#2E6FB0", True), ("とうふ", "#3C5A7A", "#FFFFFF", False),
    ("ほん", "#FFFFFF", "#3B8A5A", True), ("きっさ", "#6B3E2E", "#FFE3B0", True),
    ("はな", "#F7D7DE", "#C2456A", False), ("さかな", "#2D6FA0", "#FFFFFF", True),
    ("ゆ", "#F4EFE3", "#C8323A", True), ("やおや", "#4E8B4A", "#FFFFFF", False)]
_BOARDS = ["にじいろ", "ふくや", "おもちゃ", "コーヒー", "まるや", "でんき", "クリーニング", "はなや", "とけい",
           "めがね", "やきとり", "たばこ"]


@lru_cache(maxsize=2)
def _street_layout(seed=7):
    rng = np.random.default_rng(seed)
    shops = []
    hs_i = 0
    bd_i = 0
    for side in (-1, 1):
        z = 3.2 if side < 0 else 2.8
        i = 0
        while z < 118:
            w = rng.uniform(4.0, 6.5)
            sh = dict(side=side, z0=z, z1=z + w, i=i)
            sh["xf"] = STREET_FX + rng.choice([0, 0, 0.12, 0.3, -0.1])
            sh["floors"] = 3 if rng.random() < 0.18 else 2
            sh["h"] = (6.3 if sh["floors"] == 2 else 9.2) + rng.uniform(-0.3, 0.6)
            sh["wall"] = _WALLS[int(rng.integers(len(_WALLS)))]
            sh["kind"] = rng.choice(["shutter", "shutter", "shutter", "open", "glass", "open"])
            sh["awn"] = _AWN[int(rng.integers(len(_AWN)))] if rng.random() < 0.72 else None
            sh["awn_dep"] = rng.uniform(0.8, 1.25)
            sh["roof"] = "tile" if rng.random() < 0.38 and sh["floors"] == 2 else "flat"
            sh["win_lit"] = [rng.random() < 0.4 for _ in range(6)]
            sh["ac"] = rng.random() < 0.4
            sh["board"] = _BOARDS[bd_i % len(_BOARDS)] if (rng.random() < 0.75 and z > 7.5) else None
            if sh["board"]:
                bd_i += 1
            sh["board_col"] = rng.choice(["#F4EFE3", "#2E4A6B", "#7A3B2E", "#2F6B4F", "#F2D36B"])
            sh["hsign"] = None
            if rng.random() < 0.42 and z < 70:
                sh["hsign"] = _HSIGNS[1 + hs_i % (len(_HSIGNS) - 1)]
                hs_i += 1
            sh["rail"] = rng.random() < 0.35
            shops.append(sh)
            z += w
            i += 1
    # hero: the dagashi shop on the left, red/cream awning + lit 「だがし」 sign
    for sh in shops:
        if sh["side"] < 0 and 13 < sh["z0"] < 22:
            sh.update(kind="open", awn=("#C94A43", "#F3E8D2"), hsign=_HSIGNS[0], wall="#D9C9A3",
                      roof="tile", floors=2, h=6.4, board="だがし", board_col="#F4EFE3")
            break
    return shops


def _street_props():
    poles = [(-1, 6.5), (1, 9.0), (-1, 24.0), (1, 27.0), (-1, 42.0), (1, 45.0), (-1, 60.0), (1, 63.0),
             (-1, 80.0), (1, 84.0)]
    lamps = [(-1, 13.5), (1, 17.0), (-1, 32.0), (1, 35.5), (-1, 50.5), (1, 54.0)]
    puddles = [(-1.2, 10.5, 1.6, 2.4), (1.4, 13.0, 1.1, 1.8), (0.2, 19.0, 2.2, 3.5), (-1.9, 27.0, 1.2, 2.5),
               (1.6, 33.0, 1.5, 3.0), (-0.5, 45.0, 1.8, 4.0), (0.9, 8.6, 0.9, 1.0), (-2.4, 16.0, 0.6, 1.4)]
    return poles, lamps, puddles


def _wire_pts(a, b, sag, n=24):
    out = []
    for k in range(n + 1):
        q = k / n
        out.append((lerp(a[0], b[0], q), lerp(a[1], b[1], q) - sag * 4 * q * (1 - q), lerp(a[2], b[2], q)))
    return out


def _draw_street_ground(ctx, P, C):
    """Opaque ground base (below the horizon)."""
    hy = P.vpy
    g = cairo.LinearGradient(0, hy, 0, hy + 1000)
    g.add_color_stop_rgb(0, *mix(C["street"], C["haze"], 0.75))
    g.add_color_stop_rgb(0.12, *mix(C["street"], C["haze"], 0.35))
    g.add_color_stop_rgb(0.45, *C["street"])
    g.add_color_stop_rgb(1, *_mul(C["street"], 0.9))
    ctx.rectangle(-400, hy, 3000, 1400)
    ctx.set_source(g)
    ctx.fill()
    # pavement strips (lighter tiles) from gutter to facade
    for s in (-1, 1):
        pts = [(s * STREET_EDGE, 0, 3.0), (s * STREET_EDGE, 0, 160), (s * 5, 0, 160), (s * 5, 0, 3.0)]
        P.poly(ctx, pts)
        g = cairo.LinearGradient(0, hy, 0, hy + 900)
        g.add_color_stop_rgb(0, *mix(C["pave"], C["haze"], 0.6))
        g.add_color_stop_rgb(0.3, *C["pave"])
        g.add_color_stop_rgb(1, *_mul(C["pave"], 0.92))
        ctx.set_source(g)
        ctx.fill()


def _draw_street_ground_details(ctx, P, C):
    """Joints, gutters, manhole, puddle rims (drawn over the reflection)."""
    hy = P.vpy
    # brick-paved centre: faint joint lines across + along
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    for k in range(3, 70):
        z = 2.0 + k * 1.6
        s = P.s(z)
        if s < 6:
            break
        P.line(ctx, [(-STREET_EDGE, 0, z), (STREET_EDGE, 0, z)])
        ctx.set_line_width(max(0.6, 0.03 * s))
        _set(ctx, C["shadow"], 0.18 * min(1, s / 60))
        ctx.stroke()
    for x in (-1.5, 0.0, 1.5):
        P.line(ctx, [(x, 0, 3.0), (x, 0, 140)])
        ctx.set_line_width(1.2)
        _set(ctx, C["shadow"], 0.12)
        ctx.stroke()
    # tile joints on pavement
    for s in (-1, 1):
        for k in range(40):
            z = 3 + k * 0.9
            sc = P.s(z)
            if sc < 10:
                break
            P.line(ctx, [(s * STREET_EDGE, 0, z), (s * 4.0, 0, z)])
            ctx.set_line_width(max(0.5, 0.025 * sc))
            _set(ctx, C["shadow"], 0.2)
            ctx.stroke()
        # gutter: dark line + light curb edge
        P.line(ctx, [(s * STREET_EDGE, 0, 3.0), (s * STREET_EDGE, 0, 150)])
        ctx.set_line_width(3)
        _set(ctx, C["shadow"], 0.5)
        ctx.stroke()
        P.line(ctx, [(s * (STREET_EDGE + 0.12), 0.06, 3.0), (s * (STREET_EDGE + 0.12), 0.06, 150)])
        ctx.set_line_width(1.6)
        _set(ctx, C["street_hi"], 0.35)
        ctx.stroke()
        # drain grates along gutter
        for z in (9.0, 21.0, 36.0):
            pts = [(s * 2.75, 0, z), (s * 2.98, 0, z), (s * 2.98, 0, z + 0.7), (s * 2.75, 0, z + 0.7)]
            _fillp(ctx, P, pts, C["shadow"], 0.55)
    # manhole
    pts = [(0.9 + 0.38 * math.cos(a), 0, 12.5 + 0.38 * math.sin(a)) for a in np.linspace(0, TAU, 24)]
    P.poly(ctx, pts)
    _set(ctx, mix(C["street"], C["shadow"], 0.4), 0.8)
    ctx.fill_preserve()
    ctx.set_line_width(2)
    _set(ctx, C["street_hi"], 0.35)
    ctx.stroke()


def _puddle_poly(P, X, Z, rx, rz, seed, n=18):
    pts = []
    for i in range(n):
        a = i / n * TAU
        r = 1 + 0.2 * noise1(i * 0.8, seed) + 0.08 * math.sin(3 * a + seed)
        pts.append((X + math.cos(a) * rx * r, 0, Z + math.sin(a) * rz * r))
    return pts


def _draw_shop(L, P, C, sh, emit):
    """Draw one shop (facade, side wall, ground floor, awning, signs, windows, roof)."""
    s = sh["side"]
    face = "L" if s < 0 else "R"
    z0, z1, xf, h = sh["z0"], sh["z1"], sh["xf"], sh["h"]
    zm = (z0 + z1) / 2
    X = s * xf
    sc = P.s(zm)
    wall = _shade(C, sh["wall"], face, zm)
    wall_side = _shade(C, sh["wall"], "F", z0, 0.9)
    lamp = C["lamp"]
    # side wall facing the camera (visible where it pokes out)
    _fillp(L, P, _quadZ(z0, X, s * (xf + 14), 0, h), wall_side)
    # facade
    pts = _quadX(X, 0, h, z0, z1)
    P.poly(L, pts)
    _set(L, wall)
    L.fill()
    # grime / rain darkening gradient (bottom & top)
    y_top = P(X, h, zm)[1]
    y_bot = P(X, 0, zm)[1]
    g = cairo.LinearGradient(0, y_top, 0, y_bot)
    g.add_color_stop_rgba(0, *C["shadow"], 0.18)
    g.add_color_stop_rgba(0.25, *C["shadow"], 0.0)
    g.add_color_stop_rgba(0.75, *C["shadow"], 0.0)
    g.add_color_stop_rgba(1, *C["shadow"], 0.22)
    P.poly(L, pts)
    L.set_source(g)
    L.fill()
    # vertical edge lines (shop separation)
    L.set_line_width(max(1.0, 0.05 * sc))
    P.line(L, [(X, 0, z0), (X, h, z0)])
    _set(L, C["shadow"], 0.35)
    L.stroke()
    za, zb = z0 + 0.25, z1 - 0.25
    # ---------------------------------------------------------------- ground floor
    kind = sh["kind"]
    gy1 = 2.55
    if kind == "shutter":
        sh_col = _shade(C, "#AEB5BA", face, zm)
        _fillp(L, P, _quadX(X, 0, gy1, za, zb), sh_col)
        step = 0.09 if sc > 40 else 0.18 if sc > 15 else 0.0
        if step:
            L.set_line_width(max(0.7, 0.022 * sc))
            yy = 0.08
            k = 0
            while yy < gy1 - 0.05:
                P.line(L, [(X, yy, za), (X, yy, zb)])
                _set(L, C["shadow"] if k % 2 == 0 else (1, 1, 1), 0.22 if k % 2 == 0 else 0.12)
                L.stroke()
                yy += step / 2
                k += 1
        # shutter box & bottom bar
        _fillp(L, P, _quadX(s * (xf - 0.08), gy1, gy1 + 0.3, za - 0.05, zb + 0.05), _mul(sh_col, 0.85))
        _fillp(L, P, _quadX(X, 0, 0.07, za, zb), _mul(sh_col, 0.6))
    else:
        # lit interior
        warm = mix(C["warm"], (1, 1, 1), 0.15)
        inner = mix(_shade(C, "#6B5A4A", face, zm), warm, 0.42 * (0.4 + 0.6 * lamp))
        _fillp(L, P, _quadX(X, 0, gy1, za, zb), inner)
        # shelves / goods
        rng = np.random.default_rng(int(z0 * 100) + s)
        for yy in (0.45, 1.05, 1.65):
            P.line(L, [(X, yy, za + 0.2), (X, yy, zb - 0.2)])
            L.set_line_width(max(1, 0.04 * sc))
            _set(L, _mul(inner, 0.55), 0.8)
            L.stroke()
            nz = int((zb - za) / 0.35)
            for k in range(nz):
                zz = za + 0.25 + k * 0.33
                if zz > zb - 0.3:
                    break
                c = rng.choice(["#E8665A", "#F2C14E", "#6BB7D6", "#8BC48A", "#F29BB6", "#FFFFFF"])
                hh = rng.uniform(0.18, 0.45)
                _fillp(L, P, _quadX(X, yy + 0.02, yy + hh, zz, zz + 0.24), _shade(C, c, face, zm), 0.85)
        if kind == "glass":
            # mullions + door
            L.set_line_width(max(1.2, 0.06 * sc))
            for q in np.linspace(za, zb, 5):
                P.line(L, [(X, 0, q), (X, gy1, q)])
            P.line(L, [(X, 2.1, za), (X, 2.1, zb)])
            _set(L, _shade(C, "#3A3A3A", face, zm), 0.9)
            L.stroke()
            # glass sheen (sky reflection)
            pts = _quadX(X, 0, gy1, za, zb)
            g = cairo.LinearGradient(*P(X, gy1, za), *P(X, 0, zb))
            g.add_color_stop_rgba(0, *C["sky_hor"], 0.35)
            g.add_color_stop_rgba(0.5, *C["sky_hor"], 0.05)
            g.add_color_stop_rgba(1, *C["sky_hor"], 0.2)
            P.poly(L, pts)
            L.set_source(g)
            L.fill()
        # emitters
        P.poly(emit, _quadX(X, 0.1, gy1 - 0.1, za + 0.1, zb - 0.1))
        _set(emit, C["warm"], 0.32 * (0.3 + 0.7 * lamp))
        emit.fill()
    # ---------------------------------------------------------------- awning
    if sh["awn"] is not None:
        c1, c2 = sh["awn"]
        dep = sh["awn_dep"]
        Xf = s * (xf - dep)
        ya, yb = 3.1, 2.62
        nst = max(2, int((z1 - z0) / 0.42))
        zs = np.linspace(z0 + 0.05, z1 - 0.05, nst + 1)
        for k in range(nst):
            col = c1 if k % 2 == 0 else c2
            top = _shade(C, col, "T", zm)
            _fillp(L, P, [(X, ya, zs[k]), (X, ya, zs[k + 1]), (Xf, yb, zs[k + 1]), (Xf, yb, zs[k])], top)
            # valance with scallop
            fc = _shade(C, col, face, zm, 0.92)
            pts = [(Xf, yb, zs[k]), (Xf, yb, zs[k + 1])]
            for q in np.linspace(1, 0, 6):
                pts.append((Xf, yb - 0.22 - 0.07 * math.sin(q * math.pi), lerp(zs[k], zs[k + 1], q)))
            _fillp(L, P, pts, fc)
        # wet sheen along the awning top
        g = cairo.LinearGradient(*P(X, ya, zm), *P(Xf, yb, zm))
        g.add_color_stop_rgba(0, 1, 1, 1, 0.0)
        g.add_color_stop_rgba(0.8, 1, 1, 1, 0.16)
        g.add_color_stop_rgba(1, 1, 1, 1, 0.0)
        P.poly(L, [(X, ya, z0), (X, ya, z1), (Xf, yb, z1), (Xf, yb, z0)])
        L.set_source(g)
        L.fill()
        # side triangle (facing camera)
        _fillp(L, P, [(X, ya, z0 + 0.05), (Xf, yb, z0 + 0.05), (Xf, yb - 0.25, z0 + 0.05), (X, yb - 0.1, z0 + 0.05)],
               _shade(C, c1, "F", z0, 0.85))
        # shadow under awning on facade
        _fillp(L, P, _quadX(X, ya - 0.6, ya, z0, z1), C["shadow"], 0.18)
    # ---------------------------------------------------------------- signboard above awning
    if sh["board"]:
        by0, by1 = 3.3, 3.95
        bc = _shade(C, sh["board_col"], face, zm)
        pts = _quadX(s * (xf - 0.06), by0, by1, za + 0.1, zb - 0.1)
        _fillp(L, P, pts, bc)
        dark = sum(_c(sh["board_col"])) < 1.6
        tc = "#FFFFFF" if dark else "#3A2E2A"
        tl = (s * (xf - 0.07), by1 - 0.06, zb - 0.25 if s > 0 else za + 0.25)
        tr = (s * (xf - 0.07), by1 - 0.06, za + 0.25 if s > 0 else zb - 0.25)
        bl = (s * (xf - 0.07), by0 + 0.06, tl[2])
        _text_on_quad(L, P, sh["board"], (tl, tr, bl), 0.5 * sc, tc, alpha=0.92)
    # ---------------------------------------------------------------- upper floors
    for fl in range(1, sh["floors"]):
        fy0 = 4.1 + (fl - 1) * 2.9
        nwin = 2 if (z1 - z0) > 4.6 else 1
        for wi in range(nwin):
            wz0 = lerp(za + 0.4, zb - 0.4, wi / nwin) + 0.15
            wz1 = lerp(za + 0.4, zb - 0.4, (wi + 1) / nwin) - 0.15
            lit = sh["win_lit"][(fl * 2 + wi) % 6]
            pts = _quadX(X, fy0, fy0 + 1.35, wz0, wz1)
            frame = _shade(C, "#EDEAE2", face, zm)
            _fillp(L, P, _quadX(X, fy0 - 0.08, fy0 + 1.43, wz0 - 0.08, wz1 + 0.08), frame)
            if lit:
                glass = mix(_shade(C, "#5F5A55", face, zm), C["warm"], 0.6 * (0.35 + 0.65 * lamp))
                _fillp(L, P, pts, glass)
                # curtains drawn to the sides
                cc = mix(glass, _shade(C, "#B0624A", face, zm), 0.45)
                _fillp(L, P, _quadX(X, fy0, fy0 + 1.35, wz0, lerp(wz0, wz1, 0.22)), cc)
                _fillp(L, P, _quadX(X, fy0, fy0 + 1.35, lerp(wz0, wz1, 0.78), wz1), cc)
                P.poly(emit, pts)
                _set(emit, C["warm"], 0.18 * (0.2 + 0.8 * lamp))
                emit.fill()
            else:
                g = cairo.LinearGradient(*P(X, fy0 + 1.35, wz0), *P(X, fy0, wz1))
                g.add_color_stop_rgb(0, *mix(C["sky_hor"], C["street"], 0.35))
                g.add_color_stop_rgb(1, *mix(C["sky_mid"], C["shadow"], 0.55))
                P.poly(L, pts)
                L.set_source(g)
                L.fill()
            # mullion + curtain hint
            P.line(L, [(X, fy0, (wz0 + wz1) / 2), (X, fy0 + 1.35, (wz0 + wz1) / 2)])
            L.set_line_width(max(1, 0.05 * sc))
            _set(L, frame)
            L.stroke()
            _fillp(L, P, _quadX(X, fy0 + 1.1, fy0 + 1.35, wz0, wz1), _mul(frame, 0.9), 0.5)
        if sh["rail"] and fl == 1:
            L.set_line_width(max(1, 0.035 * sc))
            for ry in (fy0 - 0.1, fy0 + 0.55):
                P.line(L, [(s * (xf - 0.25), ry, za), (s * (xf - 0.25), ry, zb)])
            for q in np.linspace(za, zb, max(3, int((zb - za) / 0.3))):
                P.line(L, [(s * (xf - 0.25), fy0 - 0.1, q), (s * (xf - 0.25), fy0 + 0.55, q)])
            _set(L, _shade(C, "#4D5560", face, zm), 0.85)
            L.stroke()
        if sh["ac"] and fl == 1:
            az = zb - 0.9
            ax = s * (xf - 0.35)
            _fillp(L, P, _quadX(ax, fy0 - 0.75, fy0 - 0.1, az, az + 0.75), _shade(C, "#E4E2DC", face, zm))
            _fillp(L, P, _quadZ(az, ax, X, fy0 - 0.75, fy0 - 0.1), _shade(C, "#CFCBC2", "F", az))
            cx_, cy_ = P(ax, fy0 - 0.42, az + 0.37)
            L.arc(cx_, cy_, 0.22 * P.s(az), 0, TAU)
            _set(L, C["shadow"], 0.35)
            L.fill()
    # ---------------------------------------------------------------- roof
    if sh["roof"] == "tile":
        ov = 0.75
        tile = _shade(C, "#56627A", "T", zm)
        _fillp(L, P, [(s * (xf - ov), h - 0.15, z0), (s * (xf - ov), h - 0.15, z1),
                      (s * (xf + 3.5), h + 1.9, z1), (s * (xf + 3.5), h + 1.9, z0)], tile)
        _fillp(L, P, [(X, h + 0.1, z0), (X, h + 0.1, z1), (s * (xf - ov), h - 0.15, z1), (s * (xf - ov), h - 0.15, z0)],
               _shade(C, "#4A3F38", "U", zm))
        L.set_line_width(max(1.2, 0.08 * sc))
        P.line(L, [(s * (xf - ov), h - 0.15, z0), (s * (xf - ov), h - 0.15, z1)])
        _set(L, _shade(C, "#3E475A", face, zm))
        L.stroke()
        _fillp(L, P, [(s * (xf - ov), h - 0.15, z0), (s * (xf + 3.5), h + 1.9, z0), (s * (xf + 3.5), h - 0.15, z0)],
               _shade(C, sh["wall"], "F", z0, 0.8))
    else:
        _fillp(L, P, _quadX(s * (xf - 0.12), h - 0.35, h, z0, z1), _shade(C, sh["wall"], face, zm, 0.82))
        _fillp(L, P, _quadZ(z0, s * (xf - 0.12), X, h - 0.35, h), _shade(C, sh["wall"], "F", z0, 0.75))
        P.line(L, [(s * (xf - 0.12), h - 0.35, z0), (s * (xf - 0.12), h - 0.35, z1)])
        L.set_line_width(max(1, 0.04 * sc))
        _set(L, C["shadow"], 0.4)
        L.stroke()
    # ---------------------------------------------------------------- hanging sign (frontal)
    if sh["hsign"] is not None:
        text, bcol, tcol, lit = sh["hsign"]
        zs_ = z0 + 0.55
        xa, xb = sorted((s * (xf - 0.15), s * (xf - 0.85)))
        n = len(text)
        y0_, y1_ = 3.55, 3.55 + 0.62 * n + 0.35
        # bracket
        P.line(L, [(s * xf, y1_ + 0.1, zs_), ((xa if s > 0 else xb), y1_ + 0.1, zs_)])
        L.set_line_width(max(1, 0.05 * P.s(zs_)))
        _set(L, _shade(C, "#3A3A3A", "F", zs_))
        L.stroke()
        box = _shade(C, bcol, "F", zs_)
        if lit:
            box = mix(box, _c(bcol), 0.45 * (0.3 + 0.7 * lamp))
        _fillp(L, P, _quadZ(zs_, xa, xb, y0_, y1_), box)
        # frame
        P.poly(L, _quadZ(zs_, xa, xb, y0_, y1_))
        L.set_line_width(max(1, 0.05 * P.s(zs_)))
        _set(L, _shade(C, "#5A4E46", "F", zs_), 0.8)
        L.stroke()
        _text_on_quad(L, P, text, ((xa, y1_ - 0.12, zs_), (xb, y1_ - 0.12, zs_), (xa, y0_ + 0.12, zs_)),
                      0.5 * P.s(zs_), tcol, vertical=True, fit=0.8)
        if lit:
            P.poly(emit, _quadZ(zs_, xa, xb, y0_, y1_))
            _set(emit, bcol, 0.3 * (0.25 + 0.75 * lamp))
            emit.fill()


def _draw_pole(L, P, C, s, z, top=10.5):
    X = s * 3.3
    sc = P.s(z)
    w = 0.32
    col = _shade(C, "#B9B6AE", "F", z)
    _fillp(L, P, _quadZ(z, X - w / 2, X + w / 2, 0, top), col)
    _fillp(L, P, _quadZ(z, X + (w / 2 if s > 0 else -w / 2) * 0.2, X + w / 2 * (1 if s > 0 else -1), 0, top),
           _mul(col, 0.85))
    # yellow/black guard stripes
    for k in range(5):
        _fillp(L, P, _quadZ(z - 0.01, X - w / 2, X + w / 2, 0.3 + k * 0.32, 0.46 + k * 0.32),
               _shade(C, "#E2B93B" if k % 2 == 0 else "#2C2C2C", "F", z))
    # crossarm + insulators
    _fillp(L, P, _quadZ(z, X - 0.9, X + 0.9, top - 0.9, top - 0.78), _shade(C, "#6C6F72", "F", z))
    for dx in (-0.75, -0.25, 0.25, 0.75):
        x_, y_ = P(X + dx, top - 0.72, z)
        L.arc(x_, y_, max(1, 0.05 * sc), 0, TAU)
        _set(L, _shade(C, "#D8DCDE", "F", z))
        L.fill()
    if int(z) % 3 == 0:  # transformer
        cx_, cy_ = P(X - s * 0.35, top - 2.4, z - 0.2)
        r = 0.28 * sc
        L.rectangle(cx_ - r, cy_ - r * 1.6, 2 * r, 3.2 * r)
        _set(L, _shade(C, "#9EA4A8", "F", z))
        L.fill()
    # small blue address sign
    _fillp(L, P, _quadZ(z - 0.02, X - w / 2 - 0.02, X + w / 2 + 0.02, 2.4, 3.2), _shade(C, "#2D5E9E", "F", z))


def _draw_lamp(L, P, C, s, z, emit, banner_idx=0):
    X = s * 3.25
    sc = P.s(z)
    col = _shade(C, "#59616A", "F", z)
    _fillp(L, P, _quadZ(z, X - 0.07, X + 0.07, 0, 5.0), col)
    # arm towards street
    P.line(L, [(X, 4.9, z), (X - s * 0.7, 5.1, z), (X - s * 0.85, 4.95, z)])
    L.set_line_width(max(1, 0.06 * sc))
    _set(L, col)
    L.stroke()
    lx, ly = P(X - s * 0.85, 4.85, z)
    r = max(1.5, 0.18 * sc)
    L.save()
    L.translate(lx, ly)
    L.scale(1, 0.75)
    L.arc(0, 0, r, 0, TAU)
    L.restore()
    lampc = mix(_shade(C, "#EDE6D6", "F", z), C["warm"], 0.8 * C["lamp"])
    _set(L, lampc)
    L.fill()
    emit.arc(lx, ly, r * 1.2, 0, TAU)
    _set(emit, C["warm"], 0.9 * C["lamp"])
    emit.fill()
    # banner flag (frontal) hanging from a small arm
    colors = [("#F28BA8", "#FFFFFF"), ("#6CC6C9", "#FFFFFF"), ("#F7C948", "#7A4E36")]
    bc, tc = colors[banner_idx % 3]
    xa, xb = sorted((X - s * 0.08, X - s * 0.62))
    P.line(L, [(X, 4.35, z), (X - s * 0.66, 4.35, z)])
    L.set_line_width(max(1, 0.03 * sc))
    _set(L, col)
    L.stroke()
    pts = [(xa, 4.3, z), (xb, 4.3, z), (xb, 2.75, z), ((xa + xb) / 2, 2.95, z), (xa, 2.75, z)]
    _fillp(L, P, pts, _shade(C, bc, "F", z))
    _text_on_quad(L, P, "にじいろ", ((xa, 4.2, z), (xb, 4.2, z), (xa, 3.0, z)), 0.3 * sc, tc, vertical=True, fit=0.8)


def _draw_vending(L, P, C, emit, z=10.6, side=1):
    s = side
    xf = STREET_FX
    xa = s * (xf - 0.75)
    hgt = 1.85
    z1 = z + 1.0
    body = _shade(C, "#E9EEF2", "F", z)
    # front (faces the street, plane X = xa)
    face = "R" if s > 0 else "L"
    _fillp(L, P, _quadX(xa, 0, hgt, z, z1), _shade(C, "#D9473F", face, z))
    disp = mix(_shade(C, "#BFE6F2", face, z), C["vend"], 0.7)
    _fillp(L, P, _quadX(xa, 0.85, hgt - 0.12, z + 0.08, z1 - 0.08), disp)
    rng = np.random.default_rng(3)
    for row, yy in enumerate((0.95, 1.25, 1.52)):
        for k in range(6):
            zz = z + 0.12 + k * 0.13
            c = rng.choice(["#E85A4F", "#3E7CC8", "#F2C14E", "#5DB86A", "#FFFFFF", "#F08A3C"])
            _fillp(L, P, _quadX(xa, yy, yy + 0.2, zz, zz + 0.07), c)
    _fillp(L, P, _quadX(xa, 0.15, 0.38, z + 0.2, z1 - 0.2), _shade(C, "#2A2A2A", face, z))
    # side (frontal)
    _fillp(L, P, _quadZ(z, xa, s * xf, 0, hgt), _mul(body, 0.92))
    _fillp(L, P, _quadZ(z, xa, s * xf, 1.0, 1.6), _shade(C, "#D9473F", "F", z))
    P.poly(emit, _quadX(xa, 0.85, hgt - 0.12, z + 0.08, z1 - 0.08))
    _set(emit, C["vend"], 0.9 * (0.45 + 0.55 * C["lamp"]))
    emit.fill()


def _draw_busstop(L, P, C, s=-1, z=8.0):
    X = s * 3.05
    sc = P.s(z)
    col = _shade(C, "#C8CCD0", "F", z)
    _fillp(L, P, _quadZ(z, X - 0.04, X + 0.04, 0, 2.5), col)
    # base
    bx, by = P(X, 0.05, z)
    L.save()
    L.translate(bx, by)
    L.scale(1, 0.35)
    L.arc(0, 0, 0.25 * sc, 0, TAU)
    L.restore()
    _set(L, _shade(C, "#8F959A", "F", z))
    L.fill()
    # timetable panel
    _fillp(L, P, _quadZ(z, X - 0.22, X + 0.22, 1.25, 1.95), _shade(C, "#F4F2EC", "F", z))
    for k in range(5):
        _fillp(L, P, _quadZ(z, X - 0.17, X + 0.17, 1.35 + k * 0.11, 1.39 + k * 0.11), C["shadow"], 0.35)
    # round sign
    cx_, cy_ = P(X, 2.55, z)
    r = 0.3 * sc
    L.arc(cx_, cy_, r, 0, TAU)
    _set(L, _shade(C, "#2E8B57", "F", z))
    L.fill()
    L.arc(cx_, cy_, r * 0.78, 0, TAU)
    _set(L, _shade(C, "#FFFFFF", "F", z))
    L.fill()
    _text_on_quad(L, P, "バス", ((X - 0.22, 2.7, z), (X + 0.22, 2.7, z), (X - 0.22, 2.4, z)), 0.2 * sc,
                  "#2E6B47", fit=0.9)


def _draw_arch(L, P, C, emit, z=72.0):
    col = _shade(C, "#E8E2D4", "F", z)
    for X in (-4.3, 4.3):
        _fillp(L, P, _quadZ(z, X - 0.25, X + 0.25, 0, 7.6), col)
    _fillp(L, P, _quadZ(z, -4.8, 4.8, 6.0, 7.4), _shade(C, "#3D6E9E", "F", z))
    _fillp(L, P, _quadZ(z - 0.01, -4.5, 4.5, 6.15, 7.25), _shade(C, "#FFFDF6", "F", z))
    # rainbow motif on top
    cx_, cy_ = P(0, 7.4, z)
    r = 1.6 * P.s(z)
    for k, c in enumerate(["#EE5A5A", "#F7A04A", "#F6E25A", "#7AD66B", "#4D86F0", "#8A5BD6"]):
        L.arc(cx_, cy_, r - k * 0.12 * P.s(z), math.pi, TAU)
        L.set_line_width(0.12 * P.s(z))
        _set(L, _shade(C, c, "F", z))
        L.stroke()
    _text_on_quad(L, P, "にじいろ商店街", ((-4.4, 7.15, z), (4.4, 7.15, z), (-4.4, 6.25, z)), 0.8 * P.s(z),
                  "#2F5F8F", fit=0.86)
    for k in range(13):
        x_, y_ = P(-4.4 + k * 8.8 / 12, 6.05, z)
        emit.arc(x_, y_, 0.08 * P.s(z), 0, TAU)
        _set(emit, C["warm"], 0.9 * C["lamp"])
        emit.fill()


def _draw_wires(L, P, C, near=None):
    """Electric wires between poles (+ across street).  near: None=all, True only z<12, False only z>=12."""
    poles, _, _ = _street_props()
    L.set_line_cap(cairo.LINE_CAP_ROUND)
    col = _shade(C, "#2E333A", "F", 20)
    for s in (-1, 1):
        ps = [z for (ss, z) in poles if ss == s]
        for a, b in zip(ps[:-1], ps[1:]):
            if near is not None and ((a < 12) != near):
                continue
            for dx, dy, sag in ((-0.75, 9.78, 0.55), (0.75, 9.78, 0.6), (0.0, 8.9, 0.75), (-0.2, 7.0, 0.9)):
                X = s * 3.3 + dx
                pts = _wire_pts((X, dy, a), (X, dy, b), sag)
                P.line(L, pts)
                L.set_line_width(max(0.8, 0.028 * P.s(a)))
                _set(L, col, 0.85)
                L.stroke()
    # cross-street wires + drop lines
    for (s, z) in poles:
        if s > 0 or z > 70:
            continue
        if near is not None and ((z < 12) != near):
            continue
        zz = z + 2.5
        pts = _wire_pts((-3.3, 8.9, z), (3.3, 8.9, zz), 0.8)
        P.line(L, pts)
        L.set_line_width(max(0.8, 0.028 * P.s(z)))
        _set(L, col, 0.85)
        L.stroke()
        for s2 in (-1, 1):
            zz = z + (4 if s2 < 0 else 7)
            P.line(L, _wire_pts((s2 * 3.3, 7.0, z if s2 < 0 else z + 2.5), (s2 * 3.65, 5.8, zz), 0.25, 10))
            L.set_line_width(max(0.6, 0.02 * P.s(z)))
            _set(L, col, 0.7)
            L.stroke()


def _draw_bunting(L, P, C, z0, z1, y=5.4, seed=0):
    a = (-3.25, y, z0)
    b = (3.25, y, z1)
    pts = _wire_pts(a, b, 0.6, 40)
    P.line(L, pts)
    L.set_line_width(max(0.7, 0.02 * P.s(z0)))
    _set(L, _shade(C, "#EDEDED", "F", z0), 0.7)
    L.stroke()
    cols = ["#EE6B6B", "#F7C948", "#6CC6C9", "#8FD18B", "#F29BB6", "#7B9BE0"]
    for k in range(1, 40, 1):
        p, q = pts[k - 1], pts[k]
        if k % 2:
            continue
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 - 0.42, (p[2] + q[2]) / 2)
        col = _shade(C, cols[(k // 2 + seed) % len(cols)], "F", mid[2])
        _fillp(L, P, [p, q, mid], col)


def _paint_plate(ctx, surf, cam, rect, ps, alpha=1.0, dx=0.0, filt=cairo.FILTER_BILINEAR):
    ctx.save()
    _apply_cam(ctx, cam)
    ctx.translate(rect[0] + dx, rect[1])
    ctx.scale(1 / ps, 1 / ps)
    ctx.set_source_surface(surf, 0, 0)
    ctx.get_source().set_filter(filt)
    ctx.paint_with_alpha(alpha)
    ctx.restore()


def _plate_surface(rect, ps):
    w = int((rect[2] - rect[0]) * ps)
    h = int((rect[3] - rect[1]) * ps)
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    c = cairo.Context(s)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    c.scale(ps, ps)
    c.translate(-rect[0], -rect[1])
    return s, c


_STREET_KEYS = (0.0, 0.5, 1.0)
_STREET_CACHE = _LRU(3)


def _street_plates(wkey):
    """Build cached street layers for weather key: dict(ground, refl, front, near, glow)."""
    C = palette(wkey)
    rect, ps = STREET_PLATE, STREET_SCALE
    P = _Proj(STREET_F, STREET_VP[0], STREET_VP[1], STREET_EYE)
    Pm = _Proj(STREET_F, STREET_VP[0], STREET_VP[1], STREET_EYE, mirror=True)
    shops = _street_layout()
    poles, lamps, puddles = _street_props()

    # ---- ground
    g_s, g = _plate_surface(rect, ps)
    _draw_street_ground(g, P, C)

    def scene(L, PP, emit, far_only=False):
        # farthest first
        _draw_arch(L, PP, C, emit)
        items = []
        for sh in shops:
            items.append((sh["z0"], "shop", sh))
        for (s, z) in poles:
            items.append((z, "pole", (s, z)))
        for k, (s, z) in enumerate(lamps):
            items.append((z, "lamp", (s, z, k)))
        items.append((10.6, "vend", None))
        items.append((8.0, "bus", None))
        items.sort(key=lambda it: -it[0])
        for z, kind, d in items:
            if kind == "shop":
                _draw_shop(L, PP, C, d, emit)
            elif kind == "pole" and z >= 12:
                _draw_pole(L, PP, C, *d)
            elif kind == "lamp":
                _draw_lamp(L, PP, C, d[0], d[1], emit, d[2])
            elif kind == "vend":
                _draw_vending(L, PP, C, emit)
            elif kind == "bus":
                _draw_busstop(L, PP, C)
        _draw_wires(L, PP, C, near=False)
        for k, (za, zb) in enumerate(((15, 22), (33, 40), (51, 58))):
            _draw_bunting(L, PP, C, za, zb, seed=k)

    # ---- front layer: ground details, shops, props
    f_s, f = _plate_surface(rect, ps)
    e_s, e = _plate_surface(rect, ps)
    _draw_street_ground_details(f, P, C)
    # puddle rims (subtle) + darker water
    for k, (X, Z, rx, rz) in enumerate(puddles):
        P.poly(f, _puddle_poly(P, X, Z, rx, rz, k))
        f.set_line_width(2.0)
        _set(f, C["street_hi"], 0.25)
        f.stroke()
    # warm light spill on the wet ground in front of open shops (additive layer)
    for sh in shops:
        if sh["kind"] != "shutter" and sh["z0"] < 60:
            s = sh["side"]
            zm = (sh["z0"] + sh["z1"]) / 2
            pts = [(s * sh["xf"], 0, sh["z0"]), (s * sh["xf"], 0, sh["z1"]),
                   (s * (sh["xf"] - 2.6), 0, sh["z1"] + 0.6), (s * (sh["xf"] - 2.6), 0, sh["z0"] - 0.6)]
            x0_, y0_ = P(s * sh["xf"], 0, zm)
            x1_, y1_ = P(s * (sh["xf"] - 2.6), 0, zm)
            gg = cairo.LinearGradient(x0_, y0_, x1_, y1_)
            gg.add_color_stop_rgba(0, *C["warm"], 0.28 * (0.3 + 0.7 * C["lamp"]))
            gg.add_color_stop_rgba(1, *C["warm"], 0.0)
            P.poly(e, pts)
            e.set_source(gg)
            e.fill()
    scene(f, P, e)

    # ---- near layer (parallax-ish foreground): nearest poles + wires
    n_s, n = _plate_surface(rect, ps)
    for (s, z) in poles:
        if z < 12:
            _draw_pole(n, P, C, s, z)
    _draw_wires(n, P, C, near=True)

    # ---- glow: blurred emitters (screen-blended at draw time)
    ea = _rgba_rows(e_s)
    glow = _ablur(ea, 26 * ps, 26 * ps, 6) * 1.0 + _ablur(ea, 7 * ps, 7 * ps, 2) * 0.5
    # bake glow additively into the front plate: premultiplied colour added with alpha untouched
    # composites as  dst + glow  under OVER (saturating).
    fa = _rgba_rows(f_s)
    fa[..., :3] = np.minimum(fa[..., :3] + glow[..., :3], 1.0)
    f_s = core.rgba_surface(fa)

    # ---- reflection: mirrored scene (+ mirrored emitters), vertical smear, puddles sharper
    r_s, r = _plate_surface(rect, ps)
    re_s, re = _plate_surface(rect, ps)
    r.rectangle(rect[0], STREET_HORIZON_Y, rect[2] - rect[0], rect[3] - STREET_HORIZON_Y)
    r.clip()
    scene(r, Pm, re)
    for (s, z) in poles:
        if z < 12:
            _draw_pole(r, Pm, C, s, z)
    y0 = int((STREET_HORIZON_Y - rect[1]) * ps)
    fh = r_s.get_height()
    ra = _rgba_rows(r_s, y0)
    rea = _rgba_rows(re_s, y0)
    soft = _ablur(ra, 9 * ps, 2.2 * ps, 2)
    sharp = _ablur(ra, 1.6 * ps, 0.8 * ps, 1)
    streak = _ablur(rea, 34 * ps, 3.0 * ps, 3)
    # puddle mask
    m_s, m = _plate_surface(rect, ps)
    for k, (X, Z, rx, rz) in enumerate(puddles):
        P.poly(m, _puddle_poly(P, X, Z, rx, rz, k))
        _set(m, (1, 1, 1))
        m.fill()
    pm = _rgba_rows(m_s, y0)[..., 3:4]
    refl = soft * (1 - pm) + sharp * pm
    refl[..., :3] += streak[..., :3] * 1.6 + _ablur(rea, 14 * ps, 14 * ps, 4)[..., :3] * 0.5
    refl[..., 3:] = np.maximum(refl[..., 3:], np.clip(streak[..., :3].max(-1, keepdims=True) * 1.3, 0, 1))
    np.clip(refl, 0, 1, out=refl)
    refl_s = _rows_to_surface(refl, fh, y0)
    # puddles: deeper water -> darker base
    for k, (X, Z, rx, rz) in enumerate(puddles):
        P.poly(g, _puddle_poly(P, X, Z, rx, rz, k))
        _set(g, C["shadow"], 0.3)
        g.fill()
    return dict(ground=g_s, refl=refl_s, front=f_s, near=n_s)


def _street_layers(wkey):
    return _STREET_CACHE.get(wkey, lambda: _street_plates(wkey))


def _weights(weather, keys):
    w = clamp(weather)
    for a, b in zip(keys[:-1], keys[1:]):
        if w <= b + 1e-9:
            k = (w - a) / (b - a)
            if k < 0.02:
                return [(a, 1.0)]
            if k > 0.98:
                return [(b, 1.0)]
            return [(a, 1 - k), (b, k)]
    return [(keys[-1], 1.0)]


def street_to_screen(X, Y, Z, cam=STREET_CAM_HIGH):
    """3D street point (m) -> screen px under cam."""
    P = _Proj(STREET_F, STREET_VP[0], STREET_VP[1], STREET_EYE)
    return to_screen(*P(X, Y, Z), cam)


def street_ground_poly(cam=STREET_CAM_HIGH, z_far=80.0):
    """Screen polygon of the walkable street (gutter to gutter) — for splashes / glints."""
    P = _Proj(STREET_F, STREET_VP[0], STREET_VP[1], STREET_EYE)
    pts = [(-STREET_FX + 0.3, 0, 3.5), (-STREET_FX + 0.3, 0, z_far), (STREET_FX - 0.3, 0, z_far),
           (STREET_FX - 0.3, 0, 3.5)]
    return [to_screen(*P(*p), cam) for p in pts]


_PASSERS = [  # X lane, Z0, speed (m/s, + = away), umbrella colour, coat colour, size, phase
    (-1.6, 14.0, 1.1, "#E9EEF2", "#3B4656", 1.0, 0.0),
    (1.2, 30.0, -1.2, "#2D4A7A", "#4A3F3A", 1.05, 0.3),
    (0.3, 22.0, 1.0, "#D24A4A", "#2F3A4A", 1.0, 0.6),
    (2.2, 44.0, -1.0, "#F2C94C", "#5A6B7A", 0.75, 0.1),   # kid, yellow
    (-2.3, 52.0, 0.9, "#E9EEF2", "#6B5A4A", 1.0, 0.8),
    (-0.6, 64.0, -1.1, "#3F8F6B", "#3B4656", 1.0, 0.45),
    (1.8, 10.0, 0.95, "#8E6AB0", "#2E3440", 1.0, 0.2),
]


def _passer_state(i, t):
    X, Z0, v, uc, cc, sz, ph = _PASSERS[i]
    zmin, zmax = 5.0, 92.0
    span = zmax - zmin
    Z = zmin + ((Z0 - zmin) + v * t) % span
    return X + 0.15 * math.sin(t * 0.7 + i), Z, v, uc, cc, sz, ph


def draw_passersby(ctx, t, weather, cam=STREET_CAM_HIGH, mirror=False, alpha=1.0):
    """Stylised pedestrians with umbrellas walking along the street (s01 depth). Umbrellas fold away
    as weather -> 1."""
    C = palette(weather)
    P = _Proj(STREET_F, STREET_VP[0], STREET_VP[1], STREET_EYE, mirror=mirror)
    ctx.save()
    _apply_cam(ctx, cam)
    order = sorted(range(len(_PASSERS)), key=lambda i: -_passer_state(i, t)[1])
    open_k = 1 - smoothstep(0.55, 0.85, weather)
    for i in order:
        X, Z, v, uc, cc, sz, ph = _passer_state(i, t)
        sc = P.s(Z)
        fade = smoothstep(5.0, 7.0, Z) * (1 - smoothstep(80, 92, Z)) * alpha
        if fade <= 0.01:
            continue
        hgt = 1.6 * sz
        step = (t * 1.9 * abs(v) + ph) * TAU
        bob = abs(math.sin(step)) * 0.03
        body = _shade(C, cc, "F", Z)
        fx, fy = P(X, 0, Z)
        # legs
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_width(max(1.0, 0.13 * sc))
        for k in (-1, 1):
            sw = math.sin(step) * 0.18 * k
            hx, hy = P(X + 0.07 * k, 0.8 * sz + bob, Z)
            ctx.move_to(hx, hy)
            fxx, fyy = P(X + 0.07 * k, 0.04, Z + sw)
            ctx.line_to(fxx, fyy)
        _set(ctx, _mul(body, 0.7), fade)
        ctx.stroke()
        # body (rounded coat)
        x0_, y0_ = P(X - 0.24 * sz, 1.45 * sz + bob, Z)
        x1_, y1_ = P(X + 0.24 * sz, 0.72 * sz + bob, Z)
        rr = (x1_ - x0_) * 0.4
        ctx.new_path()
        ctx.move_to(x0_ + rr, y0_)
        ctx.line_to(x1_ - rr, y0_)
        ctx.curve_to(x1_, y0_, x1_, y0_ + rr, x1_ + (x1_ - x0_) * 0.08, y1_)
        ctx.line_to(x0_ - (x1_ - x0_) * 0.08, y1_)
        ctx.curve_to(x0_, y0_ + rr, x0_, y0_, x0_ + rr, y0_)
        _set(ctx, body, fade)
        ctx.fill()
        # head
        hx, hy = P(X, 1.55 * sz + bob, Z)
        ctx.arc(hx, hy, 0.11 * sz * sc, 0, TAU)
        _set(ctx, _shade(C, "#3A3030", "F", Z), fade)
        ctx.fill()
        if open_k > 0.02:
            # umbrella dome
            ux, uy = P(X + 0.05, (1.95 + 0.05 * open_k) * sz + bob, Z)
            ur = 0.55 * sz * sc * (0.3 + 0.7 * open_k)
            uh = ur * 0.55
            ucol = _shade(C, uc, "T", Z)
            clear = uc == "#E9EEF2"
            ctx.move_to(ux - ur, uy + uh * 0.35)
            ctx.curve_to(ux - ur, uy - uh, ux + ur, uy - uh, ux + ur, uy + uh * 0.35)
            # scalloped rim
            for k in range(4, 0, -1):
                xa = ux - ur + (k / 4) * 2 * ur
                xb = ux - ur + ((k - 1) / 4) * 2 * ur
                ctx.curve_to(xa - ur * 0.12, uy + uh * 0.1, xb + ur * 0.12, uy + uh * 0.1, xb, uy + uh * 0.35)
            ctx.close_path()
            _set(ctx, ucol, fade * (0.55 if clear else 0.97))
            ctx.fill_preserve()
            ctx.set_line_width(max(0.6, 0.02 * sc))
            _set(ctx, (1, 1, 1) if clear else _mul(ucol, 0.7), fade * 0.6)
            ctx.stroke()
            # highlight
            ctx.move_to(ux - ur * 0.6, uy - uh * 0.25)
            ctx.curve_to(ux - ur * 0.4, uy - uh * 0.6, ux - ur * 0.05, uy - uh * 0.7, ux + ur * 0.15, uy - uh * 0.68)
            ctx.set_line_width(max(0.8, 0.05 * sc))
            _set(ctx, (1, 1, 1), fade * 0.35)
            ctx.stroke()
            # shaft
            ctx.move_to(ux, uy - uh * 0.6)
            ctx.line_to(ux, hy + 0.1 * sc)
            ctx.set_line_width(max(0.6, 0.025 * sc))
            _set(ctx, _mul(ucol, 0.5), fade)
            ctx.stroke()
        else:
            # folded umbrella in hand
            ax, ay = P(X + 0.26 * sz, 0.95 * sz + bob, Z)
            bx, by = P(X + 0.3 * sz, 0.1, Z)
            ctx.move_to(ax, ay)
            ctx.line_to(bx, by)
            ctx.set_line_width(max(1, 0.06 * sc))
            _set(ctx, _shade(C, uc, "F", Z), fade)
            ctx.stroke()
    ctx.restore()


def draw_street(ctx, t, weather, cam=STREET_CAM_HIGH, sky=True, passersby=True, near=True,
                rainbow=0.0, rainbow_center=(1350, 700), rainbow_radius=820, drips=True):
    """The rainy 商店街 seen from a high angle (s01; wide after-rain views for s06).
    Draws: sky (+ optional rainbow) → wet ground with rippling mirrored reflections (shops, lights,
    sky, puddles) → shops/props → passers-by → nearest poles & wires (near=True) → awning drips.
    Rain itself is separate (draw_rain / draw_splashes, with street_ground_poly(cam) as region).
    cam: STREET_CAM_START → STREET_CAM_HIGH for the crane-down.  rainbow: 0..1 appear
    (centre/radius in WORLD px, sky-relative)."""
    C = palette(weather)
    rect, ps = STREET_PLATE, STREET_SCALE
    hz = to_screen(0, STREET_HORIZON_Y, cam)[1]
    # ---- sky (+ rainbow)
    if sky:
        draw_sky(ctx, t, weather, cam, horizon_y=STREET_HORIZON_Y, depth=1.0)
        if rainbow > 0:
            rc = to_screen(*rainbow_center, cam)
            draw_rainbow(ctx, t, rainbow, rc, rainbow_radius * cam[2])
    ws = _weights(weather, _STREET_KEYS)
    layers = [(_street_layers(k), a) for k, a in ws]
    # ---- ground base
    for i, (Ls, a) in enumerate(layers):
        _paint_plate(ctx, Ls["ground"], cam, rect, ps, 1.0 if i == 0 else a)
    # ---- reflections (group: mirrored sky + mirrored scene with ripple) masked by fresnel
    if hz < H:
        ctx.save()
        ctx.rectangle(0, max(0, hz), W, H - max(0, hz))
        ctx.clip()
        ctx.push_group()
        arr = sky_array(t, weather, cam, STREET_HORIZON_Y, 1.0)
        _paint_small(ctx, arr, flip_axis=hz)
        if rainbow > 0:
            ctx.save()
            ctx.translate(0, 2 * hz)
            ctx.scale(1, -1)
            rc = to_screen(*rainbow_center, cam)
            draw_rainbow(ctx, t, rainbow, rc, rainbow_radius * cam[2], alpha=0.4)
            ctx.restore()
        amp = 7.0 * C["ripple"] * cam[2]
        y = max(0.0, hz)
        strip = 3
        while y < H:
            depthk = clamp((y - hz) / (H - hz + 1e-6))
            dx = amp * (0.25 + depthk) * (0.6 * noise1(y * 0.11 + t * 9.0, 3) + 0.4 * math.sin(y * 0.37 - t * 6.0))
            ctx.save()
            ctx.rectangle(0, y, W, strip)
            ctx.clip()
            for i, (Ls, a) in enumerate(layers):
                _paint_plate(ctx, Ls["refl"], cam, rect, ps, 1.0 if i == 0 else a, dx=dx / cam[2])
            ctx.restore()
            y += strip
        ctx.pop_group_to_source()
        g = cairo.LinearGradient(0, hz, 0, H + 80)
        r0 = C["refl"]
        g.add_color_stop_rgba(0, 0, 0, 0, r0 * 1.0)
        g.add_color_stop_rgba(0.35, 0, 0, 0, r0 * 0.85)
        g.add_color_stop_rgba(1, 0, 0, 0, r0 * 0.55)
        ctx.mask(g)
        ctx.restore()
        if passersby:
            ctx.save()
            ctx.rectangle(0, max(0, hz), W, H)
            ctx.clip()
            draw_passersby(ctx, t, weather, cam, mirror=True, alpha=0.35)
            ctx.restore()
    # ---- shops & props
    for i, (Ls, a) in enumerate(layers):
        _paint_plate(ctx, Ls["front"], cam, rect, ps, 1.0 if i == 0 else a)
    if passersby:
        draw_passersby(ctx, t, weather, cam)
    # ---- drips from awnings (a few, near ones)
    if drips:
        _street_drips(ctx, t, weather, cam)
    if near:
        for i, (Ls, a) in enumerate(layers):
            _paint_plate(ctx, Ls["near"], cam, rect, ps, 1.0 if i == 0 else a)
    # ---- rain haze down the street
    if C["rain_amt"] > 0:
        vx, vy = to_screen(960, STREET_HORIZON_Y, cam)
        g = cairo.RadialGradient(vx, vy, 10, vx, vy, 700 * cam[2])
        g.add_color_stop_rgba(0, *C["haze"], 0.32 * C["rain_amt"])
        g.add_color_stop_rgba(1, *C["haze"], 0.0)
        ctx.rectangle(0, 0, W, H)
        ctx.set_source(g)
        ctx.fill()


def _street_drips(ctx, t, weather, cam):
    P = _Proj(STREET_F, STREET_VP[0], STREET_VP[1], STREET_EYE)
    pts = []
    for sh in _street_layout():
        if sh["awn"] is None or sh["z0"] > 30:
            continue
        s = sh["side"]
        Xf = s * (sh["xf"] - sh["awn_dep"])
        n = int((sh["z1"] - sh["z0"]) / 0.9)
        for k in range(n):
            z = sh["z0"] + 0.4 + k * 0.9
            x, y = to_screen(*P(Xf, 2.36, z), cam)
            gx, gy = to_screen(*P(Xf, 0, z), cam)
            if -50 < x < W + 50 and y < H:
                pts.append((x, y, gy))
    sz = 2.6 * cam[2]
    draw_drips_from(ctx, t, pts, weather=weather, size=sz, gravity=2600 * cam[2], seed=33, alpha=0.7)


# =========================================================================== shopfront (s02–s06)
# Ground-level, near-frontal view of the closed dagashi shop 「ねこや」.  World coords == screen at
# CAM_ID.  A side street on the right recedes to the vanishing point and opens a window of sky.
SHOP_F = 2890.0
SHOP_VP = (1660.0, 536.0)
SHOP_HORIZON_Y = 536.0
SHOP_EYE = 1.0                      # camera height (m)
SHOP_WALL_Z = 9.0
SHOP_AWN_Z = 7.8                    # awning front plane (drip line)
SHOP_X0, SHOP_X1 = -4.7, -0.25      # dagashi facade extent (m)
SHOP_AWN_X = (-4.95, -0.12)


def _sp(mirror=False):
    return _Proj(SHOP_F, SHOP_VP[0], SHOP_VP[1], SHOP_EYE, mirror=mirror)


_SP = _sp()
GROUND_Y = 880.0                                  # where characters stand (Z = 8.4 m)
GROUND_Z = SHOP_F * SHOP_EYE / (GROUND_Y - SHOP_HORIZON_Y)
WALL_BASE_Y = _SP(0, 0, SHOP_WALL_Z)[1]           # ≈ 857
DRIP_GROUND_Y = _SP(0, 0, SHOP_AWN_Z)[1]          # ≈ 906  (front edge of the dry patch)
AWNING_WALL_Y = _SP(0, 2.5, SHOP_WALL_Z)[1]       # ≈ 54
AWNING_TOP_Y = _SP(0, 2.2, SHOP_AWN_Z)[1]         # ≈ 91  (top of the valance)
AWNING_EDGE_Y = _SP(0, 1.93, SHOP_AWN_Z)[1]       # ≈ 191 (bottom of the valance scallops = drip edge)
AWNING_X0 = _SP(SHOP_AWN_X[0], 0, SHOP_AWN_Z)[0]  # ≈ -174 (left end is off-screen)
AWNING_X1 = _SP(SHOP_AWN_X[1], 0, SHOP_AWN_Z)[0]  # ≈ 1616
SHOP_PX_PER_M = (GROUND_Y - SHOP_HORIZON_Y) / SHOP_EYE   # ≈ 344 px per metre at GROUND_Y
SHOPFRONT_CAM = CAM_ID
SHOP_RAINBOW = ((2350.0, 1050.0), 1250.0)         # default rainbow centre/radius (world px)

# prop anchors (world == screen px at CAM_ID)
DOOR_RECT = (_SP(-3.25, 0, 9)[0], _SP(0, 2.05, 9)[1], _SP(-1.85, 0, 9)[0], WALL_BASE_Y)   # x0,y0,x1,y1
WINDOW_RECT = (_SP(-4.5, 0, 9)[0], _SP(0, 1.85, 9)[1], _SP(-3.6, 0, 9)[0], _SP(0, 0.95, 9)[1])
BENCH_SEAT = (_SP(-4.45, 0, 8.75)[0], _SP(0, 0.42, 8.75)[1], _SP(-3.05, 0, 8.75)[0])     # x0, seat y, x1
GACHAPON_X = (_SP(-1.78, 0, 8.7)[0], _SP(-1.05, 0, 8.7)[0])
VENDING_RECT = (_SP(-0.98, 0, 8.6)[0], _SP(0, 1.83, 8.6)[1], _SP(-0.28, 0, 8.6)[0], _SP(0, 0, 8.6)[1])
SIDE_STREET_X = _SP(SHOP_X1, 0, 9)[0]             # ≈ 1580: right of this is the side street


def ground_y(x=None):
    """Pavement line where characters stand (flat): GROUND_Y for any x."""
    return GROUND_Y


def shop_scale(y_feet):
    """Relative character scale for feet at screen-world y (1.0 at GROUND_Y) — for moving in depth."""
    return (y_feet - SHOP_HORIZON_Y) / (GROUND_Y - SHOP_HORIZON_Y)


def shop_to_screen(X, Y, Z, cam=CAM_ID):
    """3D shopfront point (m) -> screen."""
    return to_screen(*_SP(X, Y, Z), cam)


def awning_drip_points(cam=CAM_ID):
    """Scallop tips along the valance bottom (screen coords) — natural drip sources."""
    pts = []
    ax0, ax1 = SHOP_AWN_X
    nst = int(round((ax1 - ax0) / 0.35))
    for k in range(nst):
        x, y = _SP(lerp(ax0, ax1, (k + 0.5) / nst), 1.935, SHOP_AWN_Z)
        pts.append(to_screen(x, y, cam))
    return pts


def _scallop_pts(Xa, Xb, Y, Z, depth=0.03, w=0.35):
    pts = []
    n = max(1, int(round(abs(Xb - Xa) / w)))
    for k in range(n):
        for q in np.linspace(0, 1, 7):
            X = lerp(Xa + k * (Xb - Xa) / n, Xa + (k + 1) * (Xb - Xa) / n, q)
            pts.append((X, Y - depth * math.sin(q * math.pi), Z))
    return pts


def _leaf_clump(L, cx, cy, r, col, seed, n=9, lit=None):
    rng = np.random.default_rng(seed)
    for k in range(n):
        a = rng.uniform(-math.pi, 0.2)
        d = rng.uniform(0.2, 0.75) * r
        x, y = cx + math.cos(a) * d, cy + math.sin(a) * d * 0.9
        rr = r * rng.uniform(0.32, 0.5)
        L.save()
        L.translate(x, y)
        L.rotate(rng.uniform(0, math.pi))
        L.scale(1, 0.55)
        L.arc(0, 0, rr, 0, TAU)
        L.restore()
        _set(L, _mul(col, rng.uniform(0.8, 1.1)))
        L.fill()
        if lit is not None:
            L.save()
            L.translate(x - rr * 0.15, y - rr * 0.2)
            L.scale(1, 0.4)
            L.arc(0, 0, rr * 0.45, 0, TAU)
            L.restore()
            _set(L, lit, 0.35)
            L.fill()


def _pot(L, P, C, X, Z, w=0.32, h=0.3, col="#C8714A", plant="leaf", seed=0, face="F"):
    x0, y0 = P(X - w / 2, h, Z)
    x1, _ = P(X + w / 2, h, Z)
    xb0, yb = P(X - w * 0.38, 0, Z)
    xb1, _ = P(X + w * 0.38, 0, Z)
    L.move_to(x0, y0)
    L.line_to(x1, y0)
    L.line_to(xb1, yb)
    L.line_to(xb0, yb)
    L.close_path()
    pc = _shade(C, col, face)
    _set(L, pc)
    L.fill()
    rim = P.s(Z) * 0.05
    L.rectangle(x0 - rim * 0.3, y0 - rim * 0.2, (x1 - x0) + rim * 0.6, rim * 1.2)
    _set(L, _mul(pc, 1.08))
    L.fill()
    cx = (x0 + x1) / 2
    r = (x1 - x0) * 0.8
    if plant == "leaf":
        _leaf_clump(L, cx, y0 - r * 0.15, r, _shade(C, "#4E8A55", face), seed, 11, _shade(C, "#BFE0A0", face))
    elif plant == "tall":
        rng = np.random.default_rng(seed)
        for k in range(9):
            a = -math.pi / 2 + rng.uniform(-0.7, 0.7)
            ln = r * rng.uniform(1.0, 1.9)
            L.move_to(cx, y0)
            L.curve_to(cx + math.cos(a) * ln * 0.3, y0 + math.sin(a) * ln * 0.5,
                       cx + math.cos(a) * ln * 0.8, y0 + math.sin(a) * ln * 0.9,
                       cx + math.cos(a) * ln, y0 + math.sin(a) * ln)
            L.set_line_width(max(2, r * 0.12))
            _set(L, _shade(C, "#5C9A5A" if k % 2 else "#3F7A4A", face))
            L.stroke()
    elif plant == "hydrangea":
        _leaf_clump(L, cx, y0 - r * 0.1, r * 1.1, _shade(C, "#4A7F57", face), seed, 9, None)
        rng = np.random.default_rng(seed + 5)
        for k, (dx, dy) in enumerate(((-0.35, -0.45), (0.3, -0.55), (0.0, -0.85), (-0.55, -0.15), (0.5, -0.2))):
            fx, fy = cx + dx * r, y0 + dy * r
            fr = r * 0.36
            base = ["#7D8FE0", "#9B7FD6", "#86A9EC", "#B58AD9", "#7FA0E8"][k]
            L.arc(fx, fy, fr, 0, TAU)
            _set(L, _shade(C, base, face, None, 0.85))
            L.fill()
            for j in range(16):
                a = rng.uniform(0, TAU)
                d = math.sqrt(rng.random()) * fr * 0.85
                L.arc(fx + math.cos(a) * d, fy + math.sin(a) * d * 0.9, fr * 0.22, 0, TAU)
                _set(L, _shade(C, mix(base, "#FFFFFF", rng.uniform(0.0, 0.45)), face))
                L.fill()


def _shop_scene(L, P, C, emit, mirror=False):
    """Everything vertical in the shopfront view (the ground is drawn separately)."""
    sunk = C["sunk"]
    lamp = C["lamp"]
    warm = C["warm"]
    # ---------------------------------------------------------------- far: houses down the side street
    for (X0, X1, Z, hgt, col, roof) in ((2.4, 9.0, 44.0, 6.2, "#D8CDB8", "#5A6680"), (-1.0, 2.2, 60.0, 7.0, "#C9D3D6", "#6B5A52"),
                                        (2.3, 7.0, 30.0, 4.6, "#E4DCCB", "#4F5D78"), (-0.8, 1.6, 85.0, 9.5, "#BFC8CE", None)):
        _fillp(L, P, _quadZ(Z, X0, X1, 0, hgt), _shade(C, col, "F", Z))
        if roof:
            _fillp(L, P, [(X0 - 0.4, hgt, Z), (X1 + 0.4, hgt, Z), ((X0 + X1) / 2 + 0.3, hgt + 1.6, Z + 2),
                          ((X0 + X1) / 2 - 0.3, hgt + 1.6, Z + 2)], _shade(C, roof, "T", Z))
        for k in range(int((X1 - X0) / 1.6)):
            wx = X0 + 0.5 + k * 1.6
            lit = (k + int(Z)) % 3 == 0
            wc = mix(_shade(C, "#56606A", "F", Z), warm, 0.6 * lamp) if lit else _shade(C, "#7E93A6", "F", Z)
            _fillp(L, P, _quadZ(Z, wx, wx + 0.8, 1.2, 2.2), wc)
            if hgt > 5:
                _fillp(L, P, _quadZ(Z, wx, wx + 0.8, 3.8, 4.8), _shade(C, "#7E93A6", "F", Z))
    # trees behind the block wall
    for (X, Z, r, seed) in ((4.2, 24.0, 2.2, 3), (6.5, 36.0, 3.0, 4), (3.2, 50.0, 2.6, 5)):
        cx, cy = P(X, 3.6 + r * 0.6, Z)
        _leaf_clump(L, cx, cy, r * P.s(Z), _shade(C, "#5F8F6A", "F", Z), seed, 14, _shade(C, "#A9D49A", "F", Z) if sunk > 0.1 else None)
    # utility poles + wires in the side street
    for (X, Z) in ((1.75, 13.0), (1.75, 34.0), (1.75, 60.0)):
        sc = P.s(Z)
        _fillp(L, P, _quadZ(Z, X - 0.15, X + 0.15, 0, 9.5), _shade(C, "#B5B2AA", "F", Z))
        _fillp(L, P, _quadZ(Z, X - 0.85, X + 0.85, 8.7, 8.82), _shade(C, "#6C6F72", "F", Z))
        for k in range(4):
            _fillp(L, P, _quadZ(Z - 0.01, X - 0.15, X + 0.15, 0.3 + k * 0.3, 0.45 + k * 0.3),
                   _shade(C, "#E2B93B" if k % 2 == 0 else "#2C2C2C", "F", Z))
    L.set_line_cap(cairo.LINE_CAP_ROUND)
    for (za, zb) in ((13.0, 34.0), (34.0, 60.0), (13.0, 9.2)):
        for dx, yy, sag in ((-0.7, 8.85, 0.5), (0.7, 8.85, 0.55), (0.0, 7.9, 0.7)):
            xb = 1.75 + dx if zb > 10 else -0.3
            P.line(L, _wire_pts((1.75 + dx, yy, za), (xb, yy - (0 if zb > 10 else 2.0), zb), sag))
            L.set_line_width(max(1.0, 0.03 * P.s(min(za, zb))))
            _set(L, _shade(C, "#2E333A", "F", 15), 0.85)
            L.stroke()
    # wires crossing the sky towards camera-left (off frame)
    for yy, sag in ((8.4, 0.9), (8.0, 1.1)):
        P.line(L, _wire_pts((1.75, yy, 13.0), (-6.0, yy + 1.5, 9.3), sag))
        L.set_line_width(2.4)
        _set(L, _shade(C, "#2E333A", "F", 12), 0.8)
        L.stroke()
    # block wall + low house on the right of the side street
    _fillp(L, P, _quadX(2.0, 0, 1.45, 9.0, 40.0), _shade(C, "#BDB8AC", "R", 14))
    for k in range(1, 7):
        P.line(L, [(2.0, k * 0.2, 9.0), (2.0, k * 0.2, 40.0)])
        L.set_line_width(1.2)
        _set(L, C["shadow"], 0.2)
        L.stroke()
    for zz in np.arange(9.4, 40, 0.4):
        P.line(L, [(2.0, 0, zz), (2.0, 1.45, zz)])
        L.set_line_width(max(0.6, 0.02 * P.s(zz)))
        _set(L, C["shadow"], 0.12)
        L.stroke()
    _fillp(L, P, _quadX(2.0, 1.45, 1.55, 9.0, 40.0), _shade(C, "#D9D4C8", "T", 14))
    # the dagashi shop's side wall facing the side street (+X face)
    _fillp(L, P, _quadX(SHOP_X1, 0, 6.2, 9.0, 18.0), _shade(C, "#C9B79A", "L", 12))
    P.line(L, [(SHOP_X1 + 0.05, 0, 9.6), (SHOP_X1 + 0.05, 6.0, 9.6)])
    L.set_line_width(5)
    _set(L, _shade(C, "#8A8F92", "L", 10))
    L.stroke()
    # ---------------------------------------------------------------- left neighbour (shutter)
    Zn = 9.15
    _fillp(L, P, _quadZ(Zn, -9.0, SHOP_X0, 0, 7.0), _shade(C, "#A9C7BF", "F", Zn))
    _fillp(L, P, _quadZ(Zn, -9.0, SHOP_X0 - 0.15, 0, 2.3), _shade(C, "#AEB5BA", "F", Zn))
    for k in range(26):
        yy = 0.08 + k * 0.085
        P.line(L, [(-9.0, yy, Zn), (SHOP_X0 - 0.15, yy, Zn)])
        L.set_line_width(1.6)
        _set(L, C["shadow"] if k % 2 == 0 else (1, 1, 1), 0.2 if k % 2 == 0 else 0.1)
        L.stroke()
    # ---------------------------------------------------------------- dagashi facade
    Z = SHOP_WALL_Z
    X0, X1 = SHOP_X0, SHOP_X1
    plaster = _shade(C, "#EADFC8", "F", Z)
    wood = _shade(C, "#6B4A36", "F", Z)
    wood_l = _shade(C, "#8A6447", "F", Z)
    _fillp(L, P, _quadZ(Z, X0, X1, 0, 7.0), plaster)
    # lower wainscot boards
    _fillp(L, P, _quadZ(Z, X0, X1, 0, 0.85), wood_l)
    for xb in np.arange(X0 + 0.14, X1, 0.14):
        P.line(L, [(xb, 0.02, Z), (xb, 0.85, Z)])
        L.set_line_width(1.4)
        _set(L, wood, 0.6)
        L.stroke()
    _fillp(L, P, _quadZ(Z, X0, X1, 0.82, 0.9), wood)
    # posts
    for xp in (X0, -3.42, -1.0, X1 - 0.12):
        _fillp(L, P, _quadZ(Z - 0.01, xp, xp + 0.12, 0, 2.5), wood)
    # beam under the awning
    _fillp(L, P, _quadZ(Z - 0.01, X0, X1, 2.3, 2.5), wood)
    # ---- small window (warm light) with lattice
    wx0, wx1, wy0, wy1 = -4.5, -3.6, 0.95, 1.85
    inner = mix(_shade(C, "#6E5A48", "F", Z), mix(warm, "#E89A50", 0.3), 0.62 * (0.45 + 0.55 * lamp))
    _fillp(L, P, _quadZ(Z, wx0, wx1, wy0, wy1), inner)
    gx0, gy0 = P(wx0, wy1, Z)
    gx1, gy1 = P(wx1, wy0, Z)
    g = cairo.RadialGradient((gx0 + gx1) / 2, (gy0 + gy1) / 2, 5, (gx0 + gx1) / 2, (gy0 + gy1) / 2, (gx1 - gx0) * 0.8)
    g.add_color_stop_rgba(0, *mix(warm, (1, 1, 1), 0.3), 0.55 * (0.4 + 0.6 * lamp))
    g.add_color_stop_rgba(1, *warm, 0.0)
    P.poly(L, _quadZ(Z, wx0, wx1, wy0, wy1))
    L.set_source(g)
    L.fill()
    for xb in np.linspace(wx0, wx1, 8):
        _fillp(L, P, _quadZ(Z - 0.01, xb - 0.018, xb + 0.018, wy0, wy1), wood)
    for yb in (wy0, (wy0 + wy1) / 2, wy1):
        _fillp(L, P, _quadZ(Z - 0.01, wx0 - 0.04, wx1 + 0.04, yb - 0.03, yb + 0.03), wood)
    P.poly(emit, _quadZ(Z, wx0, wx1, wy0, wy1))
    _set(emit, warm, 0.35 * (0.35 + 0.65 * lamp))
    emit.fill()
    # flower box under the window
    _fillp(L, P, _quadZ(Z - 0.12, wx0 - 0.05, wx1 + 0.05, wy0 - 0.2, wy0 - 0.02), _shade(C, "#7A5A44", "F", Z))
    for k, xx in enumerate(np.linspace(wx0 + 0.1, wx1 - 0.1, 4)):
        cx, cy = P(xx, wy0 - 0.02, Z - 0.12)
        _leaf_clump(L, cx, cy, 0.17 * P.s(Z), _shade(C, "#4F8A57", "F", Z), 40 + k, 7, _shade(C, "#BFE0A0", "F", Z))
        L.arc(cx + 6, cy - 0.12 * P.s(Z), 0.035 * P.s(Z), 0, TAU)
        _set(L, _shade(C, ["#F07A8A", "#F6D35B", "#FFFFFF", "#E8735A"][k], "F", Z))
        L.fill()
    # ---- glass sliding doors
    dx0, dx1, dy1 = -3.25, -1.85, 2.05
    _fillp(L, P, _quadZ(Z, dx0 - 0.06, dx1 + 0.06, 0, dy1 + 0.08), wood)
    interior = mix(_shade(C, "#4F4038", "F", Z), mix(warm, "#D98A4A", 0.35), 0.38 * (0.4 + 0.6 * lamp))
    for k in range(2):
        px0 = lerp(dx0, dx1, k / 2) + 0.05
        px1 = lerp(dx0, dx1, (k + 1) / 2) - 0.05
        _fillp(L, P, _quadZ(Z, px0, px1, 0.4, dy1 - 0.05), interior)
        L.save()
        P.poly(L, _quadZ(Z, px0, px1, 0.4, dy1 - 0.05))
        L.clip()
        # shelves with candy jars behind the glass
        for sy in (0.75, 1.2, 1.62):
            _fillp(L, P, _quadZ(Z + 0.5, px0 - 0.05, px1 + 0.05, sy - 0.03, sy), _mul(interior, 0.6))
            rng = np.random.default_rng(int(sy * 100) + k)
            for j in range(5):
                jx = lerp(px0 + 0.06, px1 - 0.1, j / 4.5)
                jc = rng.choice(["#F07A8A", "#F6D35B", "#7FC8E8", "#9BD27A", "#F4A35A", "#C79BE8"])
                cx, cy = P(jx + 0.04, sy + 0.12, Z + 0.5)
                rr = 0.05 * P.s(Z + 0.5)
                L.rectangle(cx - rr, cy - rr * 1.4, rr * 2, rr * 2.4)
                _set(L, mix(_shade(C, jc, "F", Z), interior, 0.35))
                L.fill()
                L.rectangle(cx - rr * 0.8, cy - rr * 1.75, rr * 1.6, rr * 0.4)
                _set(L, _mul(interior, 0.5))
                L.fill()
        L.restore()
        # inner curtain (closed) on the right panel, half drawn on the left
        cfrac = 1.0 if k == 1 else 0.38
        cc = mix(_shade(C, "#E3CFA8", "F", Z), mix(warm, "#E8A05A", 0.3), 0.45 * lamp)
        _fillp(L, P, _quadZ(Z - 0.01, px1 - (px1 - px0) * cfrac, px1, 0.4, dy1 - 0.05), cc, 0.92)
        for fx in np.linspace(px1 - (px1 - px0) * cfrac + 0.04, px1 - 0.03, max(2, int(9 * cfrac))):
            P.line(L, [(fx, 0.42, Z - 0.012), (fx, dy1 - 0.07, Z - 0.012)])
            L.set_line_width(1.3)
            _set(L, _mul(cc, 0.82), 0.7)
            L.stroke()
        # glass sheen
        a0 = P(px0, dy1 - 0.05, Z)
        a1 = P(px1, 0.4, Z)
        g = cairo.LinearGradient(a0[0], a0[1], a1[0], a1[1])
        g.add_color_stop_rgba(0.0, 1, 1, 1, 0.0)
        g.add_color_stop_rgba(0.30, 1, 1, 1, 0.0)
        g.add_color_stop_rgba(0.33, 1, 1, 1, 0.22)
        g.add_color_stop_rgba(0.40, 1, 1, 1, 0.22)
        g.add_color_stop_rgba(0.43, 1, 1, 1, 0.0)
        g.add_color_stop_rgba(0.5, 1, 1, 1, 0.08)
        g.add_color_stop_rgba(0.55, 1, 1, 1, 0.0)
        P.poly(L, _quadZ(Z - 0.02, px0, px1, 0.4, dy1 - 0.05))
        L.set_source(g)
        L.fill()
        # lower wooden kick panel + frame bars
        _fillp(L, P, _quadZ(Z - 0.02, px0 - 0.02, px1 + 0.02, 0.02, 0.4), wood_l)
        _fillp(L, P, _quadZ(Z - 0.02, px0 - 0.02, px1 + 0.02, 1.2, 1.24), wood)
        P.poly(emit, _quadZ(Z, px0, px1, 0.4, dy1 - 0.05))
        _set(emit, warm, 0.14 * (0.3 + 0.7 * lamp))
        emit.fill()
    # paper notice 「おやすみ」 on the right door
    nx0, nx1, ny0, ny1 = -2.42, -2.1, 1.2, 1.72
    _fillp(L, P, _quadZ(Z - 0.03, nx0, nx1, ny0, ny1), _shade(C, "#FFFDF4", "F", Z))
    _text_on_quad(L, P, "おやすみ", ((nx0, ny1, Z - 0.03), (nx1, ny1, Z - 0.03), (nx0, ny0, Z - 0.03)),
                  0.08 * P.s(Z), "#3A3A40", vertical=True, font="klee", fit=0.8)
    for (tx, ty) in ((nx0 + 0.03, ny1 - 0.03), (nx1 - 0.03, ny1 - 0.03)):
        x_, y_ = P(tx, ty, Z - 0.03)
        L.rectangle(x_ - 6, y_ - 3, 12, 6)
        _set(L, (1, 1, 1), 0.7)
        L.fill()
    # ---- above the awning (seen when tilting up): signboard, 2F, roof
    _fillp(L, P, _quadZ(Z - 0.05, X0 + 0.2, X1 - 0.2, 2.62, 3.3), wood)
    _fillp(L, P, _quadZ(Z - 0.06, X0 + 0.28, X1 - 0.28, 2.68, 3.24), _shade(C, "#F4EDDC", "F", Z))
    _text_on_quad(L, P, "だがし  ねこや商店", ((X0 + 0.3, 3.2, Z - 0.06), (X1 - 0.3, 3.2, Z - 0.06), (X0 + 0.3, 2.72, Z - 0.06)),
                  0.4 * P.s(Z), "#3B2A22", font="klee", fit=0.86)
    _fillp(L, P, _quadZ(Z - 0.02, X0, X1, 3.35, 3.5), wood)
    for k, (wx0_, wx1_) in enumerate(((-4.2, -2.9), (-2.1, -0.8))):
        _fillp(L, P, _quadZ(Z - 0.01, wx0_ - 0.08, wx1_ + 0.08, 3.95, 5.35), wood)
        lit = k == 0
        gc = mix(_shade(C, "#5E6B76", "F", Z), warm, 0.65 * lamp) if lit else _shade(C, "#8DA2B3", "F", Z)
        _fillp(L, P, _quadZ(Z - 0.015, wx0_, wx1_, 4.03, 5.27), gc)
        for xb in np.linspace(wx0_, wx1_, 4)[1:-1]:
            _fillp(L, P, _quadZ(Z - 0.02, xb - 0.02, xb + 0.02, 4.03, 5.27), wood)
        _fillp(L, P, _quadZ(Z - 0.25, wx0_ - 0.15, wx1_ + 0.15, 3.85, 3.95), wood_l)
        if lit:
            P.poly(emit, _quadZ(Z, wx0_, wx1_, 4.03, 5.27))
            _set(emit, warm, 0.4 * lamp)
            emit.fill()
    # tiled roof eave
    tile = _shade(C, "#4F5A70", "U", Z)
    _fillp(L, P, [(X0 - 0.3, 5.75, Z), (X1 + 0.3, 5.75, Z), (X1 + 0.3, 5.75, Z - 0.7), (X0 - 0.3, 5.75, Z - 0.7)],
           _shade(C, "#4A3F38", "U", Z))
    _fillp(L, P, _quadZ(Z - 0.7, X0 - 0.3, X1 + 0.3, 5.75, 5.95), tile)
    for xb in np.arange(X0 - 0.3, X1 + 0.3, 0.22):
        x_, y_ = P(xb + 0.11, 5.75, Z - 0.7)
        L.arc(x_, y_, 0.06 * P.s(Z - 0.7), 0, math.pi)
        _set(L, _mul(tile, 0.8))
        L.fill()
    # ambient occlusion under the awning
    ya_ = P(0, 2.5, Z)[1]
    yb_ = P(0, 0.9, Z)[1]
    g = cairo.LinearGradient(0, ya_, 0, yb_)
    g.add_color_stop_rgba(0, *C["shadow"], 0.38 + 0.12 * sunk)
    g.add_color_stop_rgba(0.45, *C["shadow"], (0.12 + 0.14 * sunk))
    g.add_color_stop_rgba(1, *C["shadow"], 0.0)
    P.poly(L, _quadZ(Z - 0.04, X0, X1, 0.9, 2.5))
    L.set_source(g)
    L.fill()
    # ---------------------------------------------------------------- props at the wall base
    # bench
    bx0, bx1, bz0, bz1, by = -4.45, -3.05, 8.72, 9.0, 0.42
    for lx in (bx0 + 0.08, bx1 - 0.14):
        _fillp(L, P, _quadZ(bz0 + 0.03, lx, lx + 0.06, 0, by), _shade(C, "#56798A", "F", bz0))
        _fillp(L, P, _quadZ(bz1 - 0.04, lx, lx + 0.06, 0, by), _shade(C, "#4F6F7E", "F", bz1))
    top = _shade(C, "#8FB6C6", "T", bz0)
    P.poly(L, [(bx0, by, bz0), (bx1, by, bz0), (bx1, by, bz1), (bx0, by, bz1)])
    _set(L, top)
    L.fill()
    for k in range(1, 3):
        zz = lerp(bz0, bz1, k / 3)
        P.line(L, [(bx0, by, zz), (bx1, by, zz)])
        L.set_line_width(1.2)
        _set(L, C["shadow"], 0.3)
        L.stroke()
    _fillp(L, P, _quadZ(bz0, bx0, bx1, by - 0.06, by), _shade(C, "#6F98AA", "F", bz0))
    for lx in (bx0 + 0.08, bx1 - 0.14):
        _fillp(L, P, _quadZ(bz1 - 0.03, lx, lx + 0.06, by, by + 0.42), _shade(C, "#4F6F7E", "F", bz1))
    for yy in (by + 0.18, by + 0.34):
        _fillp(L, P, _quadZ(bz1 - 0.035, bx0, bx1, yy, yy + 0.08), _shade(C, "#86ADBE", "F", bz1))
    # umbrella stand (empty!)
    usx = -3.32
    _fillp(L, P, _quadZ(8.85, usx - 0.1, usx + 0.1, 0, 0.5), _shade(C, "#3E6FA8", "F", 8.85))
    cx, cy = P(usx, 0.5, 8.85)
    L.save()
    L.translate(cx, cy)
    L.scale(1, 0.18)
    L.arc(0, 0, 0.1 * P.s(8.85), 0, TAU)
    L.restore()
    _set(L, _shade(C, "#1F3550", "F", 8.85))
    L.fill()
    for xx in (usx - 0.1, usx + 0.1):
        _fillp(L, P, _quadZ(8.84, xx - 0.012, xx + 0.012, 0, 0.5), _shade(C, "#9DB4CF", "F", 8.85), 0.6)
    # potted plants
    _pot(L, P, C, -4.62, 8.8, 0.3, 0.32, "#C8714A", "hydrangea", 7)
    _pot(L, P, C, -1.92, 8.82, 0.22, 0.26, "#B9653F", "tall", 8)
    _pot(L, P, C, -3.55, 8.95, 0.18, 0.18, "#D08B5E", "leaf", 9)
    # gachapon machines
    for k, gx in enumerate((-1.78, -1.42)):
        gz = 8.72
        w = 0.34
        red = _shade(C, ["#D9473F", "#3E7CC8"][k], "F", gz)
        _fillp(L, P, _quadZ(gz, gx, gx + w, 0, 0.12), _shade(C, "#3A3A3A", "F", gz))
        _fillp(L, P, _quadZ(gz, gx, gx + w, 0.12, 0.66), red)
        _fillp(L, P, _quadZ(gz - 0.005, gx + 0.05, gx + w - 0.05, 0.42, 0.6), _shade(C, "#F4EFE3", "F", gz))
        cx, cy = P(gx + w / 2, 0.51, gz - 0.01)
        L.arc(cx, cy, 0.055 * P.s(gz), 0, TAU)
        _set(L, _shade(C, "#C9CDD2", "F", gz))
        L.fill()
        L.rectangle(cx - 0.05 * P.s(gz), cy - 0.008 * P.s(gz), 0.1 * P.s(gz), 0.016 * P.s(gz))
        _set(L, _shade(C, "#6A6E72", "F", gz))
        L.fill()
        _fillp(L, P, _quadZ(gz - 0.005, gx + 0.1, gx + w - 0.1, 0.16, 0.3), _shade(C, "#2A2A2A", "F", gz))
        # clear capsule case
        _fillp(L, P, _quadZ(gz, gx + 0.01, gx + w - 0.01, 0.66, 1.22), _shade(C, "#DDE9EF", "F", gz), 0.55)
        rng = np.random.default_rng(11 + k)
        for j in range(22):
            cxx = rng.uniform(gx + 0.05, gx + w - 0.05)
            cyy = 0.7 + (rng.random() ** 0.7) * 0.42
            if cyy > 0.7 + 0.42 * (1 - 0.0):
                continue
            x_, y_ = P(cxx, cyy, gz + 0.05)
            rr = 0.035 * P.s(gz)
            cc = rng.choice(["#F07A8A", "#F6D35B", "#7FC8E8", "#9BD27A", "#F4A35A", "#C79BE8", "#FFFFFF"])
            L.arc(x_, y_, rr, 0, math.pi)
            _set(L, _shade(C, "#F2F2F2", "F", gz))
            L.fill()
            L.arc(x_, y_, rr, math.pi, TAU)
            _set(L, _shade(C, cc, "F", gz))
            L.fill()
        x0_, y0_ = P(gx + 0.03, 1.2, gz)
        L.rectangle(x0_, y0_, 0.025 * P.s(gz), 0.45 * P.s(gz))
        _set(L, (1, 1, 1), 0.35)
        L.fill()
        _fillp(L, P, _quadZ(gz, gx - 0.01, gx + w + 0.01, 1.22, 1.3), red)
    # vending machine (glowing)
    vx0, vx1, vz, vh = -0.98, -0.28, 8.6, 1.83
    _fillp(L, P, _quadX(vx1, 0, vh, vz, 9.0), _shade(C, "#E9EEF2", "L", vz, 0.85))
    _fillp(L, P, _quadZ(vz, vx0, vx1, 0, vh), _shade(C, "#E2E7EB", "F", vz))
    disp = mix(_shade(C, "#BFE6F2", "F", vz), C["vend"], 0.45)
    _fillp(L, P, _quadZ(vz - 0.005, vx0 + 0.06, vx1 - 0.06, 0.9, vh - 0.08), disp)
    rng = np.random.default_rng(5)
    for row, yy in enumerate((0.98, 1.27, 1.54)):
        _fillp(L, P, _quadZ(vz - 0.006, vx0 + 0.08, vx1 - 0.08, yy - 0.03, yy - 0.01), _mul(disp, 0.75))
        for k in range(6):
            xx = vx0 + 0.1 + k * 0.088
            c = rng.choice(["#E85A4F", "#3E7CC8", "#F2C14E", "#5DB86A", "#FFFFFF", "#F08A3C", "#8A5A3C"])
            _fillp(L, P, _quadZ(vz - 0.007, xx, xx + 0.05, yy, yy + 0.19), _shade(C, c, "F", vz))
            x_, y_ = P(xx + 0.025, yy - 0.05, vz - 0.007)
            L.arc(x_, y_, 3, 0, TAU)
            _set(L, (0.3, 0.9, 0.5) if k % 2 else (1.0, 0.35, 0.3), 0.9)
            L.fill()
    _fillp(L, P, _quadZ(vz - 0.006, vx0 + 0.1, vx1 - 0.1, 0.15, 0.35), _shade(C, "#2A2A2A", "F", vz))
    _fillp(L, P, _quadZ(vz - 0.006, vx0 + 0.45, vx1 - 0.08, 0.5, 0.75), _shade(C, "#C9CDD2", "F", vz))
    P.poly(emit, _quadZ(vz, vx0 + 0.06, vx1 - 0.06, 0.9, vh - 0.08))
    _set(emit, C["vend"], 0.32 * (0.45 + 0.55 * lamp))
    emit.fill()
    # ---------------------------------------------------------------- awning
    Za, Zw = SHOP_AWN_Z, SHOP_WALL_Z
    ax0, ax1 = SHOP_AWN_X
    red, cream = "#C94A43", "#F3E8D2"
    nst = int(round((ax1 - ax0) / 0.35))
    for k in range(nst):
        xa = lerp(ax0, ax1, k / nst)
        xb = lerp(ax0, ax1, (k + 1) / nst)
        col = red if k % 2 == 0 else cream
        # underside: lit through the fabric when sunny
        uc = mix(_shade(C, col, "U", Za), _c(col), 0.35 * sunk)
        _fillp(L, P, [(xa, 2.5, Zw), (xb, 2.5, Zw), (xb, 2.2, Za), (xa, 2.2, Za)], uc)
    # side flap at the right end (faces +X)
    _fillp(L, P, [(ax1, 2.5, Zw), (ax1, 2.2, Za), (ax1, 1.95, Za)], _shade(C, red, "L", Za, 0.85))
    # valance: vertical stripes, scalloped edge, cream name panel
    for k in range(nst):
        xa = lerp(ax0, ax1, k / nst)
        xb = lerp(ax0, ax1, (k + 1) / nst)
        col = red if k % 2 == 0 else cream
        pts = [(xa, 2.2, Za), (xb, 2.2, Za)] + _scallop_pts(xb, xa, 1.99, Za, 0.06, w=(xb - xa))
        _fillp(L, P, pts, _shade(C, col, "F", Za))
    _fillp(L, P, _quadZ(Za, ax0, ax1, 2.185, 2.2), _shade(C, "#A8362F", "F", Za))
    _fillp(L, P, _quadZ(Za - 0.005, -4.0, -1.1, 2.0, 2.17), _shade(C, "#FBF3E0", "F", Za))
    P.poly(L, _quadZ(Za - 0.006, -4.0, -1.1, 2.0, 2.17))
    L.set_line_width(2.5)
    _set(L, _shade(C, "#A8362F", "F", Za))
    L.stroke()
    _text_on_quad(L, P, "だがし ・ ねこや", ((-3.95, 2.165, Za - 0.007), (-1.15, 2.165, Za - 0.007), (-3.95, 2.005, Za - 0.007)),
                  0.13 * P.s(Za), "#B83A33", fit=0.92)
    # little cat-face logo
    cx, cy = P(-2.55, 2.27, Za)
    cx, cy = P(-0.65, 2.085, Za)
    L.arc(cx, cy, 0.075 * P.s(Za), 0, TAU)
    _set(L, _shade(C, "#FBF3E0", "F", Za))
    L.fill()
    r = 0.055 * P.s(Za)
    L.arc(cx, cy, r, 0, TAU)
    _set(L, _shade(C, "#B83A33", "F", Za))
    L.fill()
    for d in (-1, 1):
        L.move_to(cx + d * r * 0.95, cy - r * 0.2)
        L.line_to(cx + d * r * 0.75, cy - r * 1.35)
        L.line_to(cx + d * r * 0.2, cy - r * 0.85)
        L.close_path()
        L.fill()
    # wet sheen on the valance
    g = cairo.LinearGradient(0, P(0, 2.2, Za)[1], 0, P(0, 1.95, Za)[1])
    g.add_color_stop_rgba(0, 1, 1, 1, 0.0)
    g.add_color_stop_rgba(0.6, 1, 1, 1, 0.10)
    g.add_color_stop_rgba(1, 1, 1, 1, 0.0)
    P.poly(L, _quadZ(Za - 0.01, ax0, ax1, 1.95, 2.2))
    L.set_source(g)
    L.fill()
    # sunlit: shadow of the awning on the wall + under-awning shade


def _shop_ground(L, P, C):
    """Pavement (dry under the awning, wet in front), road strip, side-street asphalt."""
    hz = P.vpy
    # side street + far ground (wet asphalt)
    g = cairo.LinearGradient(0, hz, 0, 1100)
    g.add_color_stop_rgb(0, *mix(C["street"], C["haze"], 0.6))
    g.add_color_stop_rgb(0.4, *C["street"])
    g.add_color_stop_rgb(1, *_mul(C["street"], 0.92))
    L.rectangle(-2000, hz, 6000, 2000)
    L.set_source(g)
    L.fill()
    # pavement in front of the shops (lighter tiles), wet
    pave = C["pave"]
    _fillp(L, P, [(-12, 0, 6.2), (SHOP_X1 + 0.05, 0, 6.2), (SHOP_X1 + 0.05, 0, 9.3), (-12, 0, 9.3)], pave)
    # dry patch under the awning (+ ragged wet edge)
    dry = C["pave_dry"]
    edge = []
    for X in np.linspace(SHOP_AWN_X[1], -12, 90):
        edge.append((X, 0, SHOP_AWN_Z + 0.06 + 0.06 * noise1(X * 3.1, 4) + 0.03 * noise1(X * 9, 5)))
    _fillp(L, P, [(-12, 0, 9.3), (SHOP_AWN_X[1] - 0.02, 0, 9.3)] + edge, dry)
    # wet darkening fringe just inside the drip line
    P.line(L, edge)
    L.set_line_width(7)
    _set(L, mix(dry, pave, 0.6), 0.6)
    L.stroke()
    # gutter + road strip
    _fillp(L, P, [(-12, 0, 5.0), (SHOP_X1 + 0.05, 0, 5.0), (SHOP_X1 + 0.05, 0, 6.2), (-12, 0, 6.2)], _mul(C["street"], 0.95))
    P.line(L, [(-12, 0.03, 6.2), (SHOP_X1 + 0.05, 0.03, 6.2)])
    L.set_line_width(4)
    _set(L, C["street_hi"], 0.4)
    L.stroke()


def _shop_ground_overlay(L, P, C, t=0.0):
    """Tile joints, puddle rims, sheen — drawn over the reflection."""
    # tile joints parallel to the wall
    for z in np.arange(6.2, 9.3, 0.45):
        P.line(L, [(-12, 0, z), (SHOP_X1, 0, z)])
        L.set_line_width(1.5 if z > 7.8 else 1.8)
        _set(L, C["shadow"], 0.16 if z > 7.8 else 0.25)
        L.stroke()
    for x in np.arange(-12, SHOP_X1, 0.45):
        P.line(L, [(x, 0, 6.2), (x, 0, 9.0)])
        L.set_line_width(1.4)
        _set(L, C["shadow"], 0.15)
        L.stroke()
    # side street gutter lines & centre markings
    for X in (SHOP_X1 + 0.2, 1.85):
        P.line(L, [(X, 0, 5.0), (X, 0, 120)])
        L.set_line_width(2.2)
        _set(L, C["shadow"], 0.4)
        L.stroke()
    for (X, Z, rx, rz, seed) in _SHOP_PUDDLES:
        pts = _puddle_poly(P, X, Z, rx, rz, seed, 24)
        P.poly(L, pts)
        L.set_line_width(2.0)
        _set(L, C["shadow"], 0.18)
        L.stroke()
        far = pts[2:11]
        if len(far) > 2:
            P.line(L, far)
            L.set_line_width(2.4)
            _set(L, C["glint"], 0.25 + 0.3 * C["sunk"])
            L.stroke()


_SHOP_PUDDLES = [(-3.6, 6.9, 0.9, 0.35, 1), (-1.1, 6.6, 0.75, 0.28, 2), (0.85, 11.0, 0.55, 2.2, 3),
                 (-5.0, 7.45, 0.6, 0.18, 4), (0.6, 6.0, 0.5, 0.3, 5)]


def _shop_wet_mask(L, P, C):
    """Alpha = wet reflectivity (0 on the dry patch / walls)."""
    hz = P.vpy
    g = cairo.LinearGradient(0, hz, 0, 1100)
    g.add_color_stop_rgba(0, 1, 1, 1, 0.95)
    g.add_color_stop_rgba(0.5, 1, 1, 1, 0.7)
    g.add_color_stop_rgba(1, 1, 1, 1, 0.55)
    # side street + road + wet pavement
    edge = []
    for X in np.linspace(-12, SHOP_AWN_X[1], 90):
        edge.append((X, 0, SHOP_AWN_Z + 0.06 + 0.06 * noise1(X * 3.1, 4) + 0.03 * noise1(X * 9, 5)))
    P.poly(L, [(-12, 0, 4.0)] + edge + [(SHOP_AWN_X[1], 0, 9.0), (SHOP_X1, 0, 9.0), (SHOP_X1, 0, 400), (6, 0, 400), (6, 0, 4.0)])
    L.set_source(g)
    L.fill()
    for (X, Z, rx, rz, seed) in _SHOP_PUDDLES:
        P.poly(L, _puddle_poly(P, X, Z, rx, rz, seed))
        _set(L, (1, 1, 1), 1.0)
        L.fill()


_SHOP_CACHE = _LRU(4)


def _shop_build(wq, ck, dof):
    """Render shopfront layers at exact cam: back (opaque, sky transparent), mirror (blurred),
    wet mask (A), overlay (+ baked glow).  Cached by (weather 0.01, cam, dof)."""
    C = palette(wq / 100.0)
    cam = ck
    P = _sp()
    Pm = _sp(mirror=True)

    def surf():
        s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        c = cairo.Context(s)
        c.set_line_join(cairo.LINE_JOIN_ROUND)
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        _apply_cam(c, cam)
        return s, c

    b_s, b = surf()
    e_s, e = surf()
    _shop_ground(b, P, C)
    _shop_scene(b, P, C, e)
    # overlay: joints, rims + light pools
    o_s, o = surf()
    _shop_ground_overlay(o, P, C)
    # warm light spill from door / window onto the ground
    for (xa, xb, k) in ((-3.3, -1.8, 0.32), (-4.6, -3.5, 0.22), (-1.0, -0.25, 0.3)):
        col = C["vend"] if xa > -1.5 else C["warm"]
        x0, y0 = P(xa, 0, SHOP_WALL_Z)
        x1, y1 = P(xb, 0, 6.6)
        gg = cairo.LinearGradient(0, y0, 0, y1)
        gg.add_color_stop_rgba(0, *col, k * (0.3 + 0.7 * C["lamp"]))
        gg.add_color_stop_rgba(1, *col, 0)
        P.poly(e, [(xa, 0, SHOP_WALL_Z), (xb, 0, SHOP_WALL_Z), (xb + 0.5, 0, 6.6), (xa - 0.5, 0, 6.6)])
        e.set_source(gg)
        e.fill()
    # mirrored scene
    m_s, m = surf()
    _shop_scene(m, Pm, C, cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 4, 4)), mirror=True)
    # wet mask
    w_s, wc = surf()
    _shop_wet_mask(wc, P, C)
    # rows that can show reflection
    wa = _rgba_rows(w_s)[..., 3]
    rows = np.nonzero(wa.max(axis=1) > 0.01)[0]
    y0 = int(rows[0]) if len(rows) else H
    # blur mirror (vertical smear) in the wet rows
    if y0 < H:
        ma = _rgba_rows(m_s, y0)
        k = cam[2]
        mb = _ablur(ma, 7 * k + dof * 0.5, 1.8 * k + dof * 0.5, 2)
        m_s = _rows_to_surface(mb, H, y0)
    # glow
    ea = _rgba_rows(e_s)
    glow = _ablur(ea, 24 * cam[2], 24 * cam[2], 6) + _ablur(ea, 6 * cam[2], 6 * cam[2], 2) * 0.5
    oa = _rgba_rows(o_s)
    oa[..., :3] = np.minimum(oa[..., :3] + glow[..., :3] * 0.75, 1.0)
    ba = None
    if dof > 0.5:
        ba = _ablur(_rgba_rows(b_s), dof, dof, max(1, int(dof / 4)))
        b_s = core.rgba_surface(ba)
        oa = _ablur(oa, dof, dof, max(1, int(dof / 4)))
    o_s = core.rgba_surface(oa)
    # wet mask as A8 surface for ctx.mask_surface
    wm = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    st = wm.get_stride()
    buf = np.ndarray((H, st), np.uint8, wm.get_data())
    buf[:, :W] = (np.clip(wa * C["refl"], 0, 1) * 255 + 0.5).astype(np.uint8)
    wm.mark_dirty()
    sky_vis = bool((_rgba_rows(b_s)[..., 3] < 0.99).any()) if ba is None else bool((ba[..., 3] < 0.99).any())
    return dict(back=b_s, mirror=m_s, wet=wm, over=o_s, y0=y0, sky=sky_vis)


def _shop_layers(weather, cam, dof):
    wq = int(round(clamp(weather) * 100))
    ck = _camkey(cam)
    dk = round(dof * 2) / 2
    return _SHOP_CACHE.get((wq, ck, dk), lambda: _shop_build(wq, ck, dk))


def draw_shopfront(ctx, t, weather, cam=CAM_ID, sky=True, dof=0.0, rainbow=0.0, rainbow_center=None,
                   rainbow_radius=None, ripples=True):
    """Ground-level view of the closed dagashi shop 「ねこや」 (s02–s06), everything BEHIND the
    characters: sky (visible over the side street on the right / when tilting up), facade, awning,
    props, dry patch, wet pavement with rippling reflections.  Then draw characters, then e.g.
    draw_shopfront_fg (drips + splashes) and draw_rain.
    dof: background blur radius px for close-ups (cached per cam).  rainbow 0..1."""
    C = palette(weather)
    Ls = _shop_layers(weather, cam, dof)
    hz = to_screen(0, SHOP_HORIZON_Y, cam)[1]
    rc = to_screen(*(rainbow_center or SHOP_RAINBOW[0]), cam)
    rr = (rainbow_radius or SHOP_RAINBOW[1]) * cam[2]
    if sky and Ls["sky"]:
        draw_sky(ctx, t, weather, cam, horizon_y=SHOP_HORIZON_Y, depth=1.0)
        if rainbow > 0:
            draw_rainbow(ctx, t, rainbow, rc, rr)
    ctx.set_source_surface(Ls["back"], 0, 0)
    ctx.paint()
    # reflection
    y0 = Ls["y0"]
    if y0 < H:
        ctx.save()
        ctx.rectangle(0, y0, W, H - y0)
        ctx.clip()
        ctx.push_group()
        if hz < H:
            arr = sky_array(t, weather, cam, SHOP_HORIZON_Y, 1.0)
            _paint_small(ctx, arr, flip_axis=hz)
            if rainbow > 0:
                ctx.save()
                ctx.translate(0, 2 * hz)
                ctx.scale(1, -1)
                draw_rainbow(ctx, t, rainbow, rc, rr, alpha=0.4)
                ctx.restore()
        amp = 5.0 * C["ripple"] * cam[2] if ripples else 0.0
        y = float(y0)
        strip = 3
        mir = Ls["mirror"]
        while y < H:
            dk = clamp((y - hz) / (H - hz + 1e-6))
            dx = amp * (0.3 + dk) * (0.6 * noise1(y * 0.09 / cam[2] + t * 8.0, 7) + 0.4 * math.sin(y * 0.3 / cam[2] - t * 5.0))
            ctx.save()
            ctx.rectangle(0, y, W, strip)
            ctx.clip()
            ctx.set_source_surface(mir, dx, 0)
            ctx.paint()
            ctx.restore()
            y += strip
        ctx.pop_group_to_source()
        ctx.mask_surface(Ls["wet"], 0, 0)
        ctx.restore()
    # puddles: deeper, clearer water tinted by the sky (blue after the rain)
    ctx.save()
    _apply_cam(ctx, cam)
    for (X, Z, rx, rz, seed) in _SHOP_PUDDLES:
        _SP.poly(ctx, _puddle_poly(_SP, X, Z, rx, rz, seed))
        y0_ = _SP(X, 0, Z + rz)[1]
        y1_ = _SP(X, 0, Z - rz)[1]
        g = cairo.LinearGradient(0, y0_, 0, y1_)
        g.add_color_stop_rgba(0, *C["sky_refl"], 0.10 + 0.30 * C["sunk"])
        g.add_color_stop_rgba(1, *C["sky_refl"], 0.0 + 0.12 * C["sunk"])
        ctx.set_source(g)
        ctx.fill()
    ctx.restore()
    ctx.set_source_surface(Ls["over"], 0, 0)
    ctx.paint()


def draw_shopfront_fg(ctx, t, weather, cam=CAM_ID, drips=True, splashes=True, glints=True, seed=3):
    """Things IN FRONT of the characters: the drip curtain from the awning (with splashes on the
    drip line), rain splashes on the wet pavement, after-rain glints."""
    C = palette(weather)
    z = cam[2]
    if splashes and C["rain_amt"] > 0:
        x0, y0 = to_screen(-200, DRIP_GROUND_Y + 12, cam)
        x1, y1 = to_screen(2200, H, cam)
        draw_splashes(ctx, t, weather, (x0, y0, x1, max(y0 + 1, min(H + 40, y1 + 40))), seed=seed,
                      horizon_y=to_screen(0, SHOP_HORIZON_Y, cam)[1], scale=1.2 * z)
        # side street
        poly = [to_screen(*p, cam) for p in ((SIDE_STREET_X + 20, 560), (1700, 560), (2400, 1100), (1600, 1100))]
        draw_splashes(ctx, t, weather, poly, seed=seed + 1, horizon_y=to_screen(0, SHOP_HORIZON_Y, cam)[1], scale=1.0 * z)
    if drips:
        pts = awning_drip_points(cam)
        gy = to_screen(0, DRIP_GROUND_Y + 8, cam)[1]
        pts = [(x, y, gy + 10 * _hash(5, i)) for i, (x, y) in enumerate(pts) if -40 < x < W + 40]
        draw_drips_from(ctx, t, pts, weather=weather, size=4.2 * z, gravity=2700 * z, seed=seed + 7,
                        period=(lerp(0.3, 1.0, C["weather"]), lerp(0.8, 2.2, C["weather"])), alpha=0.8)
        if C["rain_amt"] > 0.15:
            # a few heavier streams from the valance tips
            for j, i in enumerate((5, 13, 19)):
                if i < len(pts):
                    x, y, _ = pts[i]
                    draw_awning_drips(ctx, t, weather, x - 1, x + 1, y, ground_y=gy, spacing=100, seed=50 + j, size=4 * z)
    if glints:
        x0, y0 = to_screen(-100, DRIP_GROUND_Y + 15, cam)
        x1, y1 = to_screen(2100, H, cam)
        draw_glints(ctx, t, weather, (x0, y0, x1, y1), n=36, seed=seed + 2, size=18 * z,
                    horizon_y=to_screen(0, SHOP_HORIZON_Y, cam)[1])


def draw_ground_reflection(ctx, t, weather, draw_fn, cam=CAM_ID, axis_y=GROUND_Y, strength=0.75, blur=4.0):
    """Reflect characters in the wet pavement: draw_fn(ctx2) draws them (screen coords) onto a
    temporary layer, which is mirrored about the screen line of axis_y (world), blurred, rippled and
    masked to the wet areas.  Call after draw_shopfront, before the characters themselves."""
    C = palette(weather)
    Ls = _shop_layers(weather, cam, 0.0)
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c2 = cairo.Context(s)
    draw_fn(c2)
    ay = to_screen(0, axis_y, cam)[1]
    y0 = max(int(ay), Ls["y0"])
    if y0 >= H:
        return
    # mirror rows [2ay - H, ay] -> [ay, H]
    src0 = int(max(0, 2 * ay - H))
    a = _rgba_rows(s, src0, int(ay) + 1)
    a = a[::-1]
    a = _ablur(a, blur * 2, blur * 0.6, 1) if blur > 0 else a
    rs = _rows_to_surface(a[: H - int(ay)], H, int(ay))
    ctx.save()
    ctx.rectangle(0, y0, W, H - y0)
    ctx.clip()
    ctx.push_group()
    amp = 4.0 * C["ripple"] * cam[2]
    y = float(y0)
    while y < H:
        dx = amp * (0.6 * noise1(y * 0.09 + t * 8.0, 9) + 0.4 * math.sin(y * 0.3 - t * 5.0))
        ctx.save()
        ctx.rectangle(0, y, W, 3)
        ctx.clip()
        ctx.set_source_surface(rs, dx, 0)
        ctx.paint_with_alpha(strength)
        ctx.restore()
        y += 3
    ctx.pop_group_to_source()
    ctx.mask_surface(Ls["wet"], 0, 0)
    ctx.restore()
