"""s01 — Opening: deep starry sky, tilt down to the sleeping town, a falling star, title.

Local time t in [0, 7).  Global = local (scene starts at 0).

Layer order per frame:
  A: sky, Milky Way, stars, moon, clouds, far town, hill        (camera + zoom)
  S: shooting star, embers, impact glow (behind the mid roofs)  (additive glow)
  B: mid + near town                                            (occludes the star)
  haze above the roofs after impact, title (screen space), post.
"""
from __future__ import annotations

import math

import cairo
import numpy as np

from common import core, world
from common.core import (W, H, FPS, TAU, clamp, lerp, smoothstep, ease_in_out, ease_in, ease_out,
                         noise1, fbm)

PHASE = 0.0
CAM0 = -1400.0
T_TILT_END = 5.0

# ----------------------------------------------------------------------------- camera
def _tilt_ease(u):
    # softer than a plain cosine: long, graceful start, feather-light landing
    u = clamp(u)
    return u * u * u * (u * (u * 6 - 15) + 10)


def camera(t):
    """-> (cam_x, cam_y, zoom, anchor)"""
    e = 0.5 * ease_in_out(t / T_TILT_END) + 0.5 * _tilt_ease(t / T_TILT_END)
    cam_y = CAM0 * (1 - e)
    # very small "floating" drift (camera on a crane, not a tripod)
    cam_y += 5.0 * fbm(t * 0.35, 3)
    cam_x = 8.0 * fbm(t * 0.25, 11)
    # settle: gentle lateral drift toward the left side of town
    cam_x -= 34.0 * ease_in_out((t - 4.6) / 2.6)
    # final push toward where the star fell
    zoom = 1.0 + 0.075 * ease_in((t - 5.7) / 1.3, 2)
    anchor = (560.0, 760.0)
    return cam_x, cam_y, zoom, anchor


def _apply_cam(ctx, zoom, anchor):
    if zoom != 1.0:
        ax, ay = anchor
        ctx.translate(ax, ay)
        ctx.scale(zoom, zoom)
        ctx.translate(-ax, -ay)


# ----------------------------------------------------------------------------- star path
# Key screen positions (t, x, y) as seen with the camera at that instant; converted to
# "mid-town world" coordinates (parallax 1.0) so the tail history reads as a real
# trajectory while the camera tilts.
_KEYS_SCREEN = [
    (2.20, 1870.0, 62.0),
    (2.55, 1640.0, 236.0),
    (2.95, 1330.0, 352.0),
    (3.35, 1010.0, 410.0),
    (3.75, 810.0, 432.0),
    (4.05, 738.0, 446.0),   # loses momentum, hangs...
    (4.30, 708.0, 466.0),
    (4.55, 676.0, 508.0),   # ...then dives, accelerating
    (4.80, 636.0, 592.0),
    (5.00, 601.0, 682.0),
    (5.15, 577.0, 762.0),
    (5.30, 562.0, 852.0),
    (5.50, 552.0, 960.0),
]
T_APPEAR = 2.20
T_SPUT0, T_SPUT1 = 3.30, 3.85
T_IMPACT = 5.12


def star_par(t):
    """Parallax of the star against the camera tilt: nearly screen-locked while it
    streaks across the sky (keeps the tail readable and off the title), fully
    attached to the town by the time it lands."""
    return lerp(0.2, 1.0, smoothstep(3.9, 4.7, t))


def _cam_only(t):
    cx, cy, _, _ = camera(t)
    return cx, cy


_KEYS = []
for (_t, _x, _y) in _KEYS_SCREEN:
    _cx, _cy = _cam_only(_t)
    _p = star_par(_t)
    _KEYS.append((_t, _x + _cx * _p, _y + _cy * _p))


def _catmull(p0, p1, p2, p3, u):
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


def star_world(t):
    K = _KEYS
    if t <= K[0][0]:
        # extrapolate backwards along the first segment (only used for tail sampling)
        (t0, x0, y0), (t1, x1, y1) = K[0], K[1]
        u = (t - t0) / (t1 - t0)
        return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    if t >= K[-1][0]:
        return K[-1][1], K[-1][2]
    i = 0
    while K[i + 1][0] < t:
        i += 1
    a = K[max(0, i - 1)]
    b = K[i]
    c = K[i + 1]
    d = K[min(len(K) - 1, i + 2)]
    u = (t - b[0]) / (c[0] - b[0])
    return _catmull(a[1], b[1], c[1], d[1], u), _catmull(a[2], b[2], c[2], d[2], u)


def _to_screen(p, cam, tt=None):
    k = 1.0 if tt is None else star_par(tt)
    return p[0] - cam[0] * k, p[1] - cam[1] * k


def star_screen(t, cam, tt=None):
    tt = t if tt is None else tt
    return _to_screen(star_world(tt), cam, tt)


def _flicker(t):
    """Sputter flicker 0..1 (steppy, like a failing bulb)."""
    k = math.floor(t * 26)
    h = noise1(k * 0.91, 41) * 0.5 + 0.5
    s = noise1(t * 9.0, 42) * 0.5 + 0.5
    v = 0.35 + 0.65 * (0.6 * h + 0.4 * s)
    if h < 0.28:
        v *= 0.25  # drop-outs
    return v


def star_intensity(t):
    if t < T_APPEAR:
        return 0.0
    lt = t - T_APPEAR
    I = ease_out(lt / 0.18, 2) * (1.0 + 0.35 * math.exp(-lt * 6))  # ignition flash
    I *= 1 + 0.06 * math.sin(t * 31) * math.sin(t * 7.3)
    s = smoothstep(T_SPUT0 - 0.08, T_SPUT0 + 0.06, t) * (1 - smoothstep(T_SPUT1 - 0.05, T_SPUT1 + 0.12, t))
    if s > 0:
        I = lerp(I, I * _flicker(t), s)
    # last gasp: flares up as it starts to dive
    I *= 1 + 0.45 * math.exp(-((t - 4.22) / 0.13) ** 2) + 0.18 * smoothstep(4.3, 4.6, t)
    return I


def _tail_len(t):
    lt = t - T_APPEAR
    L = 760 * ease_out(lt / 0.55, 2)
    # during the sputter the tail breaks off
    s = smoothstep(T_SPUT0, T_SPUT0 + 0.15, t) * (1 - smoothstep(T_SPUT1, T_SPUT1 + 0.35, t))
    L = lerp(L, 150, s)
    # dive: long, straight streak
    return L


def _history(t, cam, dur=0.75, n=36):
    pts = []
    for k in range(n, -1, -1):
        tt = t - dur * k / n
        if tt < T_APPEAR - 0.02:
            continue
        pts.append(star_screen(t, cam, tt))
    return pts


# ----------------------------------------------------------------------------- embers
_RNG = np.random.default_rng(101)
N_EMB = 70
_EMB_T = np.sort(np.concatenate([_RNG.uniform(T_SPUT0 - 0.05, T_SPUT1 + 0.05, 52),
                                 _RNG.uniform(T_SPUT1 + 0.05, 4.9, 18)]))
_EMB_OFF = _RNG.normal(0, 1, (N_EMB, 2))
_EMB_V = _RNG.normal(0, 1, (N_EMB, 2))
_EMB_LIFE = _RNG.uniform(0.6, 1.5, N_EMB)
_EMB_SZ = _RNG.uniform(0.5, 1.2, N_EMB)
_EMB_FR = _RNG.uniform(7, 16, N_EMB)
_EMB_PH = _RNG.uniform(0, TAU, N_EMB)
_EMB_COL = ["#FFF6D6", "#FFE9A8", "#FFC94D", "#FFD66B"]


def _draw_embers(ctx, t, cam):
    for i in range(N_EMB):
        te = _EMB_T[i]
        age = t - te
        if age < 0 or age > _EMB_LIFE[i]:
            continue
        p0 = star_world(te)
        p1 = star_world(te - 0.03)
        vx, vy = (p0[0] - p1[0]) / 0.03, (p0[1] - p1[1]) / 0.03
        # inherit a little of the star's motion, then drift and fall
        drag = 3.0
        k = (1 - math.exp(-drag * age)) / drag
        x = p0[0] + _EMB_OFF[i, 0] * 10 + (0.18 * vx + 40 * _EMB_V[i, 0]) * k
        y = p0[1] + _EMB_OFF[i, 1] * 10 + (0.18 * vy + 40 * _EMB_V[i, 1]) * k + 26 * age * age
        f = age / _EMB_LIFE[i]
        tw = 0.5 + 0.5 * math.sin(age * _EMB_FR[i] + _EMB_PH[i])
        a = (1 - f) ** 1.2 * (0.35 + 0.65 * tw) * (1 - 0.3 * (i >= 52))
        k2 = star_par(te)
        sx, sy = x - cam[0] * k2, y - cam[1] * k2
        world.draw_glint(ctx, sx, sy, (7 + 12 * _EMB_SZ[i]) * (1 - 0.5 * f), a,
                         _EMB_COL[i % 4], rot=_EMB_PH[i], diag=0.35)


def _draw_tail_fragments(ctx, t, cam):
    """Broken bits of tail left behind while the star sputters."""
    s = smoothstep(T_SPUT0 + 0.05, T_SPUT0 + 0.2, t) * (1 - smoothstep(T_SPUT1 + 0.1, T_SPUT1 + 0.45, t))
    if s <= 0.01:
        return
    col = core.rgb("#FFE9A8")
    for j, (t0, t1) in enumerate(((0.12, 0.19), (0.23, 0.29), (0.33, 0.41), (0.45, 0.50))):
        a0, a1 = t - t1, t - t0
        if a0 < T_APPEAR:
            continue
        n = 8
        pts = [star_screen(t, cam, a1 - (a1 - a0) * k / n) for k in range(n + 1)]
        fl = 0.55 + 0.45 * (noise1(t * 14 + j * 5, 60 + j) * 0.5 + 0.5)
        fade = s * fl * (1 - 0.18 * j)
        for wmax, cc, a in ((16, world.WINDOW_GLOW, 0.14), (6.0, col, 0.55), (2.0, (1, 1, 1), 0.8)):
            widths = [max(0.3, wmax * math.sin(math.pi * k / n) ** 0.7) for k in range(n + 1)]
            world._tail_polygon(ctx, pts, widths)
            ctx.set_source_rgba(*cc, a * fade)
            ctx.fill()


def _impact_amount(t):
    if t < T_IMPACT:
        return 0.0
    lt = t - T_IMPACT
    flash = ease_out(lt / 0.12, 2) * math.exp(-lt * 2.2)
    linger = smoothstep(0.0, 0.5, lt) * (0.52 + 0.05 * math.sin(lt * 5.3) + 0.03 * math.sin(lt * 11.7))
    return max(flash, linger) + 0.5 * flash


IMPACT = (568.0, 790.0)  # mid-town world point, just behind the roofline


def _draw_star_layer(ctx, t, cam):
    I = star_intensity(t)
    head_w = star_world(t)
    hx, hy = star_screen(t, cam)
    if I > 0.003 and t < 5.45:
        pts = _history(t, cam)
        # trim path to the tail length (draw_shooting_star uses `length` as a max)
        L = _tail_len(t)
        if len(pts) >= 2:
            dx = pts[-1][0] - pts[-2][0]
            dy = pts[-1][1] - pts[-2][1]
            ang = math.atan2(dy, dx)
        else:
            ang = math.pi * 0.85
        world.draw_shooting_star(ctx, (hx, hy), ang, L, t, I, path=pts if len(pts) >= 2 else None,
                                 width=1.25, seed=5)
    _draw_tail_fragments(ctx, t, cam)
    _draw_embers(ctx, t, cam)
    # impact glow behind the roofs
    A = _impact_amount(t)
    if A > 0.003:
        ix, iy = IMPACT[0] - cam[0], IMPACT[1] - cam[1]
        world.draw_light_pool(ctx, ix, iy - 30, 440, 230, "#FFB85C", 0.6 * A, op=cairo.OPERATOR_OVER)
        world.draw_light_pool(ctx, ix, iy - 20, 200, 100, "#FFE9A8", 0.8 * A, op=cairo.OPERATOR_OVER)
        world.draw_light_pool(ctx, ix, iy - 10, 70, 40, "#FFFFFF", 0.8 * A, op=cairo.OPERATOR_OVER)
        # a puff of glitter that pops up above the roofline
        world.draw_sparkles(ctx, t - T_IMPACT - 0.04, (ix, iy), n=30, speed=330, life=1.6, gravity=140,
                            seed=9, size=0.85, spread=1.3, direction=-math.pi / 2, flash=False)


# ----------------------------------------------------------------------------- title
TITLE = "ほしのおとしもの"
TITLE_SIZE = 120
TITLE_SPACING = 12
TITLE_Y = 232
TITLE_CX = 930
TITLE_COL = "#FFF3D0"
_GPAD = 70


def _build_title():
    from PIL import Image, ImageDraw
    f = core._font("klee", TITLE_SIZE)
    d = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    widths = [d.textlength(ch, font=f) + TITLE_SPACING for ch in TITLE]
    total = sum(widths) - TITLE_SPACING
    x = TITLE_CX - total / 2
    chars = []
    for ch, wdt in zip(TITLE, widths):
        lay = core.text_layer(ch, (x, TITLE_Y), TITLE_SIZE, "klee", TITLE_COL, anchor="lm")
        ys, xs = np.nonzero(lay[..., 3] > 0.002)
        x0, x1 = xs.min() - _GPAD, xs.max() + _GPAD + 1
        y0, y1 = ys.min() - _GPAD, ys.max() + _GPAD + 1
        crop = lay[y0:y1, x0:x1].copy()
        glow = core.blur(crop[..., :3] * np.array(core.rgb("#FFC45A"), np.float32), 14, 1)
        glow2 = core.blur(crop[..., :3] * np.array(core.rgb("#FF9A3D"), np.float32), 34, 2)
        chars.append(dict(crop=crop, glow=glow, glow2=glow2, x0=x0, y0=y0,
                          cx=x + (wdt - TITLE_SPACING) / 2))
        x += wdt
    return chars


_TITLE = None


def _title_state(i, t, n):
    """alpha, rise offset, extra brightness for character i."""
    # fade/rise in 2.6–3.4 with a soft left→right stagger
    d_in = 2.6 + 0.045 * i
    a_in = smoothstep(d_in, d_in + 0.62, t)
    rise = (1 - ease_out((t - d_in) / 0.9, 3)) * 16
    # dissolve out 5.6–6.4, also left→right, drifting up
    d_out = 5.6 + 0.04 * i
    a_out = 1 - smoothstep(d_out, d_out + 0.5, t)
    rise -= ease_in((t - d_out) / 0.6, 2) * 10
    return a_in * a_out, rise


def _draw_title(img, t, cam, glint_pts):
    global _TITLE
    if t < 2.55 or t > 6.5:
        return
    if _TITLE is None:
        _TITLE = _build_title()
    n = len(_TITLE)
    # star passing beneath lights the letters in turn
    sx = star_screen(t, cam)[0] if T_APPEAR <= t <= 4.2 else -9999
    breathe = 1 + 0.04 * math.sin(t * 2.1)
    for i, C in enumerate(_TITLE):
        a, rise = _title_state(i, t, n)
        if a <= 0.003:
            continue
        hi = math.exp(-((C["cx"] - sx) / 90.0) ** 2) * star_intensity(t)
        y0 = int(round(C["y0"] + rise))
        x0 = C["x0"]
        crop, glow, glow2 = C["crop"], C["glow"], C["glow2"]
        h, w = crop.shape[:2]
        ya, yb = max(0, y0), min(H, y0 + h)
        if yb <= ya:
            continue
        sl = img[ya:yb, x0:x0 + w]
        cy0 = ya - y0
        cr = crop[cy0:cy0 + (yb - ya)]
        g1 = glow[cy0:cy0 + (yb - ya)]
        g2 = glow2[cy0:cy0 + (yb - ya)]
        # glow (screen)
        gs = np.clip((g1 * (0.95 + 0.9 * hi) + g2 * (0.5 + 0.6 * hi)) * a * breathe, 0, 1)
        sl[..., :3] = 1 - (1 - sl[..., :3]) * (1 - gs)
        # glyph
        k = a * (1 + 0.0 * hi)
        sa = cr[..., 3:4] * k
        col = cr[..., :3] * k
        if hi > 0.01:
            col = col + (cr[..., 3:4] * k) * 0.25 * hi
        sl[..., :3] = col + sl[..., :3] * (1 - sa)
        if hi > 0.02:
            glint_pts.append((C["cx"], a, hi))
    # tiny sparkles shed as the title dissolves
    return


def _draw_title_sparkles(ctx, t):
    if not (5.55 < t < 6.9):
        return
    if _TITLE is None:
        return
    rng = np.random.default_rng(77)
    for i, C in enumerate(_TITLE):
        d_out = 5.6 + 0.04 * i
        for k in range(4):
            t0 = d_out + rng.uniform(0.0, 0.45)
            age = t - t0
            life = rng.uniform(0.5, 0.9)
            ox, oy = rng.uniform(-45, 45), rng.uniform(-45, 45)
            vy = rng.uniform(-60, -25)
            if age < 0 or age > life:
                continue
            f = age / life
            a = math.sin(math.pi * f) * 0.9
            world.draw_glint(ctx, C["cx"] + ox + 8 * math.sin(age * 3 + k), TITLE_Y + oy + vy * age,
                             9 + 6 * (1 - f), a, "#FFE9A8", rot=k * 0.4, diag=0.3)


# ----------------------------------------------------------------------------- moon (sky space)
MOON_SKY = (215.0, 128.0)  # screen position at cam_y = 0


# ----------------------------------------------------------------------------- render
_VIG = None


def _vignette(img, strength=0.34, color=(0.02, 0.02, 0.08)):
    """Same look as core.vignette, with the falloff mask cached."""
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _VIG = (np.clip(d - 0.55, 0, 1) ** 1.5)[..., None].astype(np.float32)
    v = _VIG * strength
    img[..., :3] = img[..., :3] * (1 - v) + np.array(color, np.float32) * v
    return img


def _bloom(img, threshold=0.72, radius=36.0, strength=0.55):
    """core.bloom, but the tight pass is blurred at half resolution (≈2x faster)."""
    rgbp = img[..., :3]
    lum = rgbp.max(axis=2, keepdims=True)
    bright = rgbp * np.clip((lum - threshold) / (1 - threshold + 1e-6), 0, 1)
    g = core.blur(bright, radius) * strength + core.blur(bright, radius * 0.3, 2) * strength * 0.6
    img[..., :3] = 1 - (1 - rgbp) * (1 - np.clip(g, 0, 1))
    return img


def _glow_crop(img, layer, passes):
    """core.add_glow restricted to the layer's bounding box (+ margin) — much cheaper
    than blurring the full frame when the glowing content is small."""
    a = layer[::8, ::8, 3]
    ys, xs = np.nonzero(a > 0.004)
    if len(ys) == 0:
        return
    m = int(max(r for r, _, _ in passes) * 2.6)
    y0, y1 = max(0, ys.min() * 8 - m), min(H, ys.max() * 8 + 8 + m)
    x0, x1 = max(0, xs.min() * 8 - m), min(W, xs.max() * 8 + 8 + m)
    sub = img[y0:y1, x0:x1]
    src = layer[y0:y1, x0:x1]
    for r, k, tint in passes:
        g = core.blur(src[..., :3], r)
        if tint is not None:
            g = g * np.array(core.rgb(tint), np.float32)
        sub[..., :3] = 1 - (1 - sub[..., :3]) * (1 - np.clip(g * k, 0, 1))


def render(t: float) -> np.ndarray:
    cam_x, cam_y, zoom, anchor = camera(t)
    cam = (cam_x, cam_y)
    ph = PHASE

    # A: sky + distant town
    sA, cA = core.new_surface()
    _apply_cam(cA, zoom, anchor)
    world.draw_sky(cA, ph, t, cam_y=cam_y, cam_x=cam_x)
    world.draw_stars(cA, ph, t, cam_y=cam_y, cam_x=cam_x)
    mx = MOON_SKY[0] - cam_x * world.SKY_PAR
    my = MOON_SKY[1] - cam_y * world.SKY_PAR
    world.draw_moon(cA, mx, my, 34, ph, t, illum=0.28)
    world.draw_clouds(cA, ph, t, cam_y=cam_y, cam_x=cam_x, speed=9.0)
    world.draw_town(cA, "far", ph, t, cam_x=cam_x, cam_y=cam_y)
    world.draw_town(cA, "hill", ph, t, cam_x=cam_x, cam_y=cam_y)
    img = core.surface_rgba(sA)

    # S: star + embers + impact glow (behind the mid roofs)
    if t >= T_APPEAR - 0.01:
        sS, cS = core.new_surface()
        _apply_cam(cS, zoom, anchor)
        _draw_star_layer(cS, t, cam)
        star = core.surface_rgba(sS)
        core.over(img, star)
        _glow_crop(img, star, ((22, 1.1, None), (70, 0.55, "#FFB85C")))

    # B: mid + near town (hides the star as it dives in)
    sB, cB = core.new_surface()
    _apply_cam(cB, zoom, anchor)
    world.draw_town(cB, "mid", ph, t, cam_x=cam_x, cam_y=cam_y)
    world.draw_town(cB, "near", ph, t, cam_x=cam_x, cam_y=cam_y)
    near = core.surface_rgba(sB)
    core.over(img, near)
    # warm light spilling up over the rooftops after the landing (screen blend)
    A = _impact_amount(t)
    if A > 0.003:
        sH, cH = core.new_surface()
        _apply_cam(cH, zoom, anchor)
        ix, iy = IMPACT[0] - cam[0], IMPACT[1] - cam[1]
        world.draw_light_pool(cH, ix, iy - 60, 440, 300, "#FFB85C", 0.22 * A, op=cairo.OPERATOR_OVER)
        world.draw_light_pool(cH, ix, iy - 30, 170, 95, "#FFE9A8", 0.20 * A, op=cairo.OPERATOR_OVER)
        hz = core.surface_rgba(sH)
        img[..., :3] = 1 - (1 - img[..., :3]) * (1 - hz[..., :3])

    # title (screen space) + passing glints
    glints = []
    _draw_title(img, t, cam, glints)
    if glints or 5.55 < t < 6.9:
        sG, cG = core.new_surface()
        if glints:
            # one tiny sparkle slides along the title, riding just above the passing star
            sx = star_screen(t, cam)[0]
            a = max(g[1] for g in glints)
            hi = max(g[2] for g in glints)
            gy = TITLE_Y + 46 + 6 * math.sin(t * 9)
            world.draw_glint(cG, sx + 20, gy, 30 * min(1, hi + 0.3), min(1, a * (0.6 + 0.5 * hi)),
                             "#FFF6D6", rot=t * 2.0, diag=0.4)
        _draw_title_sparkles(cG, t)
        gl = core.surface_rgba(sG)
        core.over(img, gl)
        core.add_glow(img, gl, 10, 0.8)

    # post
    _bloom(img, 0.72, 36, 0.55)
    _vignette(img, 0.34)
    core.grain(img, int(round(t * FPS)), 0.018)
    return core.finish(img)
