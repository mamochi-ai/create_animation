"""s01 — Opening: rainy 商店街 from a high angle, crane down, title 「あめふりロボ」 (global 0–6 s).

Beats (local t):
  0.0–4.0  crane down from the rainy sky (wires, poles, fluttering bunting) onto the high-angle street.
  1.6–3.2  title kana drop in one by one like raindrops: stretch while falling, squash + spring on
           landing, a ripple ring + sparkle under each letter. Gentle idle bob while holding.
  3.3–6.0  slow drift / push-in toward the left-hand shops (next scene: dagashi shopfront).
  4.6–5.4  the letters "wash away": slide down, stretch and fade like drips, shedding droplets.
Post: half-res bloom, cached vignette, grain.
"""
from __future__ import annotations

import math

import cairo
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from common import core
from common.core import (W, H, TAU, FPS, clamp, lerp, remap, smoothstep, ease_in, ease_in_out, noise1)
from lib import world

WEATHER = 0.0
TITLE = "あめふりロボ"

# ------------------------------------------------------------------ camera


def _smoother(u):
    u = clamp(u)
    return u * u * u * (u * (u * 6 - 15) + 10)


def camera(t):
    s0, s1 = world.STREET_CAM_START, world.STREET_CAM_HIGH
    # crane: brief float at the top, then a long, soft landing (zero vel/accel at the end)
    p = _smoother(remap(t, 0.15, 4.25, 0, 1) ** 0.82)
    cx, cy, z = (lerp(a, b, p) for a, b in zip(s0, s1))
    # drift toward the left shops + little push-in (still moving at the cut)
    d = ease_in_out(remap(t, 3.2, 7.4, 0, 1))
    cx -= 175 * d
    cy += 30 * d
    z += 0.075 * d
    # subtle handheld float (bigger while craning)
    hk = 1.0 + 1.2 * (1 - p)
    cx += 2.6 * hk * noise1(t * 0.9, 11)
    cy += 2.0 * hk * noise1(t * 0.8, 23)
    return (cx, cy, z)


# ------------------------------------------------------------------ fluttering bunting (near strings)
_BUNT_COLS = ["#EE6B6B", "#F7C948", "#6CC6C9", "#8FD18B", "#F29BB6", "#7B9BE0"]
_BUNTING = [  # (x0, x1, y, z0, z1, sag, seed)
    (-3.05, 3.05, 6.5, 9.5, 11.5, 0.55, 0),
    (-3.05, 3.05, 5.9, 26.0, 29.0, 0.5, 3),
]


def _wire(a, b, sag, n):
    return [(lerp(a[0], b[0], q), lerp(a[1], b[1], q) - sag * 4 * q * (1 - q), lerp(a[2], b[2], q))
            for q in (k / n for k in range(n + 1))]


def draw_bunting(ctx, t, cam):
    C = world.palette(WEATHER)
    S = lambda p: world.street_to_screen(*p, cam)  # noqa: E731
    for (x0, x1, y, z0, z1, sag, seed) in _BUNTING:
        sway = 0.05 * math.sin(t * 1.7 + seed)
        pts = _wire((x0, y, z0), (x1, y, z1), sag + sway, 36)
        sp = [S(p) for p in pts]
        zc = 0.5 * (z0 + z1)
        sc = world.street_to_screen(0, 1, zc, cam)[1] - world.street_to_screen(0, 0, zc, cam)[1]
        sc = abs(sc)  # px per metre at this depth
        ctx.move_to(*sp[0])
        for q in sp[1:]:
            ctx.line_to(*q)
        ctx.set_line_width(max(0.8, 0.022 * sc))
        ctx.set_source_rgba(*world._shade(C, "#E8E8E8", "F", zc), 0.75)
        ctx.stroke()
        for k in range(1, len(pts), 2):
            a, b = sp[k - 1], sp[k]
            ph = k * 1.37 + seed * 2.1
            # flutter: flag twists (width), tip swings in the gusty wind
            gust = 0.6 + 0.4 * noise1(t * 0.8 + seed, 5)
            sw = (math.sin(t * 7.3 + ph) * 0.55 + noise1(t * 3.1 + ph, 9) * 0.45) * gust
            twist = 0.65 + 0.35 * math.cos(t * 6.1 + ph * 1.7)
            mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            ax, ay = lerp(mx, a[0], twist), lerp(my, a[1], twist)
            bx, by = lerp(mx, b[0], twist), lerp(my, b[1], twist)
            L = 0.44 * sc
            tip = (mx + sw * 0.45 * L, my + L * (0.92 - 0.12 * abs(sw)))
            col = world._shade(C, _BUNT_COLS[(k // 2 + seed) % len(_BUNT_COLS)], "F", zc)
            ctx.move_to(ax, ay)
            ctx.line_to(bx, by)
            ctx.line_to(*tip)
            ctx.close_path()
            ctx.set_source_rgba(*col, 0.95)
            ctx.fill_preserve()
            # wet sheen on the flag
            ctx.set_source_rgba(1, 1, 1, 0.10 + 0.08 * twist)
            ctx.set_line_width(max(0.6, 0.012 * sc))
            ctx.stroke()


# ------------------------------------------------------------------ warm lamp halos (gentle flicker)


def draw_lamp_halos(ctx, t, cam):
    C = world.palette(WEATHER)
    _, lamps, _ = world._street_props()
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    warm = C["warm"]
    for k, (s, z) in enumerate(lamps):
        X = s * 3.25 - s * 0.85
        lx, ly = world.street_to_screen(X, 4.85, z, cam)
        r = 1450 / z * 0.9 * cam[2]
        fl = 0.85 + 0.1 * math.sin(t * 2.3 + k) + 0.05 * noise1(t * 9 + k * 3, 4)
        g = cairo.RadialGradient(lx, ly, 0, lx, ly, r * 2.2)
        g.add_color_stop_rgba(0, *warm, 0.30 * fl)
        g.add_color_stop_rgba(0.35, *warm, 0.10 * fl)
        g.add_color_stop_rgba(1, *warm, 0)
        ctx.set_source(g)
        ctx.arc(lx, ly, r * 2.2, 0, TAU)
        ctx.fill()
    ctx.restore()


# ------------------------------------------------------------------ title sprites
TITLE_SIZE = 138
TITLE_Y = 455           # baseline (screen)
TITLE_SPACING = 6
CREAM = "#FFF6E2"
MINT_CREAM = "#E9FFF7"
TEAL = "#3E9E9A"
TEAL_DARK = "#1F5E66"


def _make_sprites():
    f = core._font("maru", TITLE_SIZE)
    sprites = []
    pad = 40
    tmp = Image.new("RGBA", (10, 10))
    d0 = ImageDraw.Draw(tmp)
    advs = [d0.textlength(ch, font=f) + TITLE_SPACING for ch in TITLE]
    total = sum(advs) - TITLE_SPACING
    x = W / 2 - total / 2
    centers = []
    for ch, adv in zip(TITLE, advs):
        centers.append(x + (adv - TITLE_SPACING) / 2)
        x += adv
    for i, ch in enumerate(TITLE):
        sz = TITLE_SIZE + 2 * pad + 40
        ax, ay = sz / 2, sz - pad - 20     # anchor = baseline centre in sprite
        fill = core.rgb(MINT_CREAM if i >= 4 else CREAM)
        # text (fill + teal outline)
        im = Image.new("RGBA", (int(sz), int(sz)), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((ax, ay), ch, font=f, anchor="ms", fill=tuple(int(c * 255) for c in fill) + (255,),
               stroke_width=9, stroke_fill=tuple(int(c * 255) for c in core.rgb(TEAL)) + (255,))
        txt = np.asarray(im).astype(np.float32) / 255
        # inner top highlight: a lighter band in the upper glyph area
        ys = np.linspace(0, 1, txt.shape[0])[:, None]
        hl = np.clip(1.0 - (ys - (ay - TITLE_SIZE * 0.8) / sz) / 0.25, 0, 1) * 0.0
        txt[..., :3] = np.clip(txt[..., :3] + hl[..., None] * 0.1, 0, 1)
        txt[..., :3] *= txt[..., 3:4]
        # soft drop shadow (teal-dark), offset down
        a = txt[..., 3]
        sh = gaussian_filter(np.roll(a, 9, axis=0), 7) * 0.6
        shadow = np.zeros_like(txt)
        shadow[..., :3] = np.array(core.rgb(TEAL_DARK), np.float32) * sh[..., None]
        shadow[..., 3] = sh
        # glow halo (soft white) for luminous readability against the rain
        gl = gaussian_filter(a, 14) * 0.35
        glow = np.zeros_like(txt)
        glow[..., :3] = gl[..., None]
        glow[..., 3] = gl
        out = glow
        out = core.over(out, shadow)
        out = core.over(out, txt)
        sprites.append((core.rgba_surface(out), ax, ay))
    return sprites, centers


_SPRITES, _CENTERS = _make_sprites()

LAND0, LAND_STEP, FALL = 1.72, 0.15, 0.30
WASH0, WASH_STEP, WASH = 4.6, 0.075, 0.42


def _land_time(i):
    return LAND0 + i * LAND_STEP + 0.02 * (i % 2)


def letter_state(i, t):
    """-> (dx, dy, sx, sy, rot, alpha, phase) ; phase in {'none','fall','sit','wash'}"""
    Ti = _land_time(i)
    side = 1 if i % 2 else -1
    if t < Ti - FALL:
        return None
    if t < Ti:
        u = (t - (Ti - FALL)) / FALL
        dy = -360 * (1 - ease_in(u, 2))
        sy = 1 + 0.38 * u
        sx = 1 - 0.2 * u
        return (0, dy, sx, sy, side * 0.10 * (1 - u), smoothstep(0, 0.35, u), "fall")
    lt = t - Ti
    # landing squash + spring
    sq = 0.42 * math.exp(-6.5 * lt) * math.cos(TAU * 3.0 * lt)
    sy = 1 - sq
    sx = 1 + 0.75 * sq
    hop = -34 * math.exp(-5.5 * lt) * max(0.0, math.sin(TAU * 1.5 * (lt - 0.08))) if lt > 0.08 else 0.0
    rot = side * 0.07 * math.exp(-5 * lt) * math.sin(TAU * 2.2 * lt)
    idle_k = smoothstep(0.5, 1.1, lt)
    dy = hop - 6 * idle_k * math.sin(TAU * 0.75 * t + i * 0.95)
    rot += idle_k * 0.025 * math.sin(TAU * 0.5 * t + i * 1.3)
    Wi = WASH0 + i * WASH_STEP
    if t < Wi:
        return (0, dy, sx, sy, rot, 1.0, "sit")
    u = (t - Wi) / WASH
    if u >= 1:
        return None
    # anticipation: a tiny upward lift before sliding away
    ant = -10 * math.sin(math.pi * clamp(u / 0.18))
    fall = 300 * ease_in(clamp((u - 0.12) / 0.88), 2.2)
    dy = dy + ant + fall
    sy = 1 + 0.7 * ease_in(u, 1.6)
    sx = 1 - 0.32 * ease_in(u, 1.4)
    return (0, dy, sx, sy, rot * (1 - u), 1 - smoothstep(0.25, 1.0, u), "wash")


def _paint_sprite(ctx, i, x, y, sx, sy, rot, alpha):
    surf, ax, ay = _SPRITES[i]
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    ctx.scale(sx, sy)
    ctx.translate(-ax, -ay)
    ctx.set_source_surface(surf, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint_with_alpha(alpha)
    ctx.restore()


def _drop(ctx, x, y, r, stretch, a):
    """Teardrop pointing up (falling down)."""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(1, stretch)
    ctx.move_to(0, -r * 2.2)
    ctx.curve_to(r * 0.6, -r * 1.2, r, -r * 0.3, r, 0.2 * r)
    ctx.arc(0, 0.2 * r, r, 0, math.pi)
    ctx.curve_to(-r, -r * 0.3, -r * 0.6, -r * 1.2, 0, -r * 2.2)
    ctx.close_path()
    ctx.restore()
    ctx.set_source_rgba(0.95, 0.98, 1.0, 0.85 * a)
    ctx.fill_preserve()
    ctx.set_source_rgba(*core.rgb(TEAL), 0.6 * a)
    ctx.set_line_width(1.6)
    ctx.stroke()


def _land_fx(ctx, i, t):
    """Ripple rings + sparkles + little bounced droplets under a landing letter."""
    lt = t - _land_time(i)
    if lt < 0 or lt > 0.9:
        return
    x, y = _CENTERS[i], TITLE_Y + 10
    for k, (dl, rmax) in enumerate(((0.0, 95), (0.12, 62))):
        u = (lt - dl) / 0.62
        if 0 < u < 1:
            r = rmax * core.ease_out(u, 2.5)
            a = (1 - u) ** 1.5 * 0.8
            ctx.save()
            ctx.translate(x, y)
            ctx.scale(1, 0.24)
            ctx.arc(0, 0, r, 0, TAU)
            ctx.restore()
            ctx.set_line_width(3.2 * (1 - u) + 0.8)
            ctx.set_source_rgba(0.93, 1.0, 0.98, a)
            ctx.stroke()
    # splash droplets
    for k in range(5):
        ang = math.pi * (0.12 + 0.76 * k / 4)
        v = 260 + 90 * ((k * 37 + i * 11) % 7) / 7
        tt = lt
        if tt > 0.42:
            continue
        dx = math.cos(ang) * v * 0.55 * tt
        dy = -math.sin(ang) * v * tt + 0.5 * 1900 * tt * tt
        a = 1 - tt / 0.42
        ctx.arc(x + dx * (1.4 if k % 2 else 1.0), y - 6 + dy, 4.2 - 2 * tt / 0.42, 0, TAU)
        ctx.set_source_rgba(0.95, 1, 1, 0.85 * a)
        ctx.fill()
    # sparkles
    for k in range(3):
        st = 0.05 + 0.09 * k
        u = (lt - st) / 0.45
        if 0 < u < 1:
            ox = (-60, 58, 8)[k] + 6 * math.sin(i + k)
            oy = (-70, -100, -150)[k] + (i % 3) * 8
            world.draw_glint(ctx, x + ox, y + oy - 16 * u, 26 * math.sin(math.pi * u), alpha=0.95,
                             color="#FFFFFF", rot=0.3 * u)


def _wash_fx(ctx, i, t, st):
    Wi = WASH0 + i * WASH_STEP
    u = (t - Wi) / WASH
    if u <= 0:
        return
    x = _CENTERS[i]
    base = TITLE_Y + st[1] if st else TITLE_Y + 300
    # drips sliding off the bottom of the letter, falling faster than it
    for k in range(3):
        d0 = 0.05 + 0.18 * k
        uu = (t - Wi - d0 * WASH) / 0.55
        if 0 < uu < 1:
            ox = (-34, 18, 40)[k] * (0.8 + 0.1 * (i % 3))
            yy = base + 6 + 520 * uu * uu
            _drop(ctx, x + ox, yy, 7 - 2 * k, 1.0 + 1.2 * uu, 1 - uu)


def draw_title(ctx, t):
    if t < LAND0 - FALL or t > WASH0 + 6 * WASH_STEP + WASH + 0.6:
        return
    for i in range(len(TITLE)):
        _land_fx(ctx, i, t)
    for i in range(len(TITLE)):
        st = letter_state(i, t)
        if st is None:
            if t > WASH0:
                _wash_fx(ctx, i, t, None)
            continue
        dx, dy, sx, sy, rot, alpha, ph = st
        x, y = _CENTERS[i] + dx, TITLE_Y + dy
        if ph == "fall":
            # speed-line trail above the falling letter
            for k in range(1, 4):
                _paint_sprite(ctx, i, x, y - k * 34, sx * (1 - 0.05 * k), sy, rot, alpha * 0.16 / k)
        if ph == "wash":
            # melting smear: ghost copies trailing upward
            u = clamp((t - (WASH0 + i * WASH_STEP)) / WASH)
            for k in range(1, 4):
                _paint_sprite(ctx, i, x, y - k * 26 * u, sx * (1 - 0.06 * k), sy * (1 + 0.08 * k), rot,
                              alpha * 0.22 / k)
        _paint_sprite(ctx, i, x, y, sx, sy, rot, alpha)
        if ph == "wash":
            _wash_fx(ctx, i, t, st)


# ------------------------------------------------------------------ post
_VIG = None


def _vignette_mask():
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _VIG = (np.clip(d - 0.55, 0, 1) ** 1.5 * 0.34)[..., None].astype(np.float32)
    return _VIG


def _bloom(img, threshold=0.74, radius=34, strength=0.5):
    rgbp = img[..., :3]
    q = 4
    small = rgbp[::q, ::q]
    lum = small.max(axis=2, keepdims=True)
    bright = small * np.clip((lum - threshold) / (1 - threshold), 0, 1)
    g = np.empty_like(bright)
    for c in range(3):
        g[..., c] = (gaussian_filter(bright[..., c], radius / q, mode="nearest") * strength
                     + gaussian_filter(bright[..., c], radius * 0.3 / q, mode="nearest") * strength * 0.6)
    up = np.empty_like(rgbp)
    for c in range(3):
        up[..., c] = np.asarray(Image.fromarray(g[..., c], mode="F").resize((W, H), Image.BILINEAR))
    img[..., :3] = 1 - (1 - rgbp) * (1 - np.clip(up, 0, 1))
    return img


def post(s, frame):
    img = core.surface_rgba(s)
    _bloom(img)
    v = _vignette_mask()
    img[..., :3] = img[..., :3] * (1 - v) + np.array((0.02, 0.02, 0.08), np.float32) * v
    core.grain(img, frame)
    return core.finish(img)


# ------------------------------------------------------------------ render


def render(t: float) -> np.ndarray:
    cam = camera(t)
    s, ctx = core.new_surface()
    world.draw_street(ctx, t, WEATHER, cam)
    hz = world.to_screen(0, world.STREET_HORIZON_Y, cam)[1]
    poly = world.street_ground_poly(cam)
    draw_lamp_halos(ctx, t, cam)
    world.draw_rain(ctx, t, WEATHER, "bg", cam)
    draw_bunting(ctx, t, cam)
    world.draw_rain(ctx, t, WEATHER, "mid", cam)
    world.draw_splashes(ctx, t, WEATHER, poly, horizon_y=hz, density=1.15)
    draw_title(ctx, t)
    world.draw_rain(ctx, t, WEATHER, "fg", cam, density=1.0 + 0.6 * (1 - smoothstep(0.5, 3.0, t)))
    # fade up from black at the very start
    a = 1 - smoothstep(0.0, 0.45, t)
    if a > 0:
        ctx.set_source_rgba(0.02, 0.03, 0.05, a)
        ctx.paint()
    return post(s, int(round(t * FPS)))
