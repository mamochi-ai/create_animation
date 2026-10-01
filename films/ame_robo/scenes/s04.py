"""s04 (27.0-38.0 global): Robo's heroic mecha transformation into a giant umbrella -> the gag.

Local beats (126 bpm, beat = 0.475 s; SFX clacks on the beats):
  A 0.000-0.950  extreme low angle, 'scan' flash, rim light, radial speed lines, frozen rain
  B 0.950-1.425  close-up: back panel doors split (clack), steam + sparkles
  C 1.425-2.150  telescoping pole extends upward, camera tilts up with it, tip glints
  D 2.150-3.400  8 ribs unfold one by one (tip glints on each "tink"), circular camera move
  E 3.400-3.800  low-angle hero pose, treads clamp-lock (ka-chunk 3.55) impact frame
  F 3.800-4.400  WHUMP: wide shot, fabric blooms over Mina, rain drums, kitten scampers under
  G 4.400-6.500  low angle close on Mina under the canopy: 「すごーい！おっきい！」, cheer, twirl
  H 6.500-11.00  snap pull-back: Robo outside in the downpour, soaked & happy; 7.6 sparks + glitch;
                 push in: Mina worried, points 「ロボさん、ぬれちゃってるよ！」; hold.
"""
from __future__ import annotations

import math

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from common import core
from common.core import clamp, lerp, ease_in_out, ease_out, ease_in, ease_out_back, noise1, smoothstep, rgb
from lib import world
from lib.characters import (draw_mina, draw_robo, draw_kitten, draw_drops_on, draw_rim_drips, draw_sparkle,
                            draw_droplet)

T0 = 27.0
FPS = core.FPS
W, H = 1920, 1080
BT = 0.475
WEATHER = 0.0

# shared staging (world coords, CAM_ID screen space)
YS = 968.0
SC = world.shop_scale(YS)
MX = 720.0                       # Mina feet x
RX = MX + 300 * SC               # Robo x (holds the canopy over her from her right)
KX0, KY0 = 610.0, 885.0          # kitten under the awning
KX1, KY1 = 590.0, 990.0          # kitten under the canopy

L07, L07_DUR = 32.0 - T0, 1.43   # local 5.0
L08, L08_DUR = 35.4 - T0, 2.23   # local 8.4
SPARK_T = 7.6


def _h(*a):
    x = math.sin(sum((i + 1) * 12.9898 * v for i, v in enumerate(a)) + 78.233) * 43758.5453
    return x - math.floor(x)


# ------------------------------------------------------------------------------------------ post
_VIG = None


def _vig_mask(strength):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _VIG = (np.clip(d - 0.55, 0, 1) ** 1.5)[..., None]
    return _VIG * strength


def _resize(a, size):
    out = np.empty((size[1], size[0], a.shape[2]), np.float32)
    for c in range(a.shape[2]):
        out[..., c] = np.asarray(Image.fromarray(a[..., c], mode="F").resize(size, Image.BILINEAR))
    return out


def post(img, frame, thr=0.74, strength=0.5, vig=0.32, grain=0.016, extra=None):
    """Half/quarter-res bloom + cached vignette + grain. img float RGB(A) (H,W,>=3)."""
    rgbp = img[..., :3]
    half = rgbp[::2, ::2]
    lum = half.max(axis=2, keepdims=True)
    bright = half * np.clip((lum - thr) / (1 - thr + 1e-6), 0, 1)
    if extra is not None:
        bright = bright + extra[::2, ::2, :3]
    q = _resize(bright, (W // 8, H // 8))
    big = np.stack([gaussian_filter(q[..., c], 4.0, mode="nearest") for c in range(3)], -1)
    small = np.stack([gaussian_filter(bright[..., c], 5.0, mode="nearest", truncate=2.5) for c in range(3)], -1)
    g = _resize(big, (W // 2, H // 2)) * strength + small * strength * 0.6
    g = _resize(g, (W, H))
    img[..., :3] = 1 - (1 - rgbp) * (1 - np.clip(g, 0, 1))
    v = _vig_mask(vig)
    img[..., :3] = img[..., :3] * (1 - v) + np.array((0.02, 0.02, 0.08), np.float32) * v
    core.grain(img, frame, grain)
    return img


# ------------------------------------------------------------------------------------------ helpers
def _sc(ctx, c, a=1.0):
    r, g, b = rgb(c) if isinstance(c, str) else c
    ctx.set_source_rgba(r, g, b, a)


def _robo_anchors(**kw):
    """Anchor pass on a tiny surface (identity space) to plan camera moves."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 4, 4)
    c = cairo.Context(s)
    return draw_robo(c, **kw)


def speed_lines(ctx, t, focus, c0, c1, n=80, seed=1, inner=260, alpha=0.55, col="#FFFFFF", rot=0.0):
    """Flat anime BG: radial gradient + radial wedge speed lines that boil at 12 fps."""
    fx, fy = focus
    g = cairo.RadialGradient(fx, fy, 40, fx, fy, 1500)
    g.add_color_stop_rgb(0, *rgb(c0))
    g.add_color_stop_rgb(1, *rgb(c1))
    ctx.set_source(g)
    ctx.paint()
    k = int(t * 12)
    for i in range(n):
        a = rot + (i + _h(i, seed, k) * 0.8) / n * math.tau
        w = (0.004 + 0.012 * _h(i, seed, k, 3))
        r0 = inner * (0.8 + 0.9 * _h(i, seed, k, 5))
        ctx.move_to(fx + math.cos(a) * r0, fy + math.sin(a) * r0)
        ctx.line_to(fx + math.cos(a - w) * 2600, fy + math.sin(a - w) * 2600)
        ctx.line_to(fx + math.cos(a + w) * 2600, fy + math.sin(a + w) * 2600)
        ctx.close_path()
        _sc(ctx, col, alpha * (0.35 + 0.65 * _h(i, seed, k, 7)))
        ctx.fill()


def parallel_lines(ctx, t, angle, c0, c1, n=60, seed=3, speed=4000, alpha=0.5, col="#FFFFFF"):
    """Flat BG with parallel motion streaks travelling along `angle` (rad, screen)."""
    g = cairo.LinearGradient(0, 0, W, H)
    g.add_color_stop_rgb(0, *rgb(c0))
    g.add_color_stop_rgb(1, *rgb(c1))
    ctx.set_source(g)
    ctx.paint()
    dx, dy = math.cos(angle), math.sin(angle)
    px, py = -dy, dx
    for i in range(n):
        off = (_h(i, seed) - 0.5) * 2600
        L = 300 + 900 * _h(i, seed, 2)
        pos = ((t * speed * (0.6 + 0.8 * _h(i, seed, 4)) + _h(i, seed, 5) * 3000) % 3600) - 1800
        cx, cy = 960 + px * off + dx * pos, 540 + py * off + dy * pos
        ctx.move_to(cx - dx * L / 2, cy - dy * L / 2)
        ctx.line_to(cx + dx * L / 2, cy + dy * L / 2)
        ctx.set_line_width(2 + 7 * _h(i, seed, 6))
        _sc(ctx, col, alpha * (0.3 + 0.7 * _h(i, seed, 7)))
        ctx.stroke()


def frozen_rain(ctx, t, n=140, seed=9, alpha=0.5, slow=60.0, length=(30, 90), slant=0.2):
    """Slow-motion rain streaks for the montage (almost frozen in time)."""
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i in range(n):
        z = 0.4 + 0.6 * _h(i, seed)
        x = _h(i, seed, 1) * 2200 - 140
        y = ((_h(i, seed, 2) * 1300 + t * slow * z) % 1300) - 110
        L = lerp(*length, _h(i, seed, 3)) * z
        ctx.move_to(x, y)
        ctx.line_to(x - slant * L, y - L)
        ctx.set_line_width(1.2 + 2.4 * z)
        _sc(ctx, "#E6F2FA", alpha * z)
        ctx.stroke()
        if z > 0.85:
            draw_droplet(ctx, x, y + 4, 3.5 * z, alpha)


def rim_light_draw(ctx, fn, offset=(-7, -7), color="#8FF6FF", a=0.95):
    """Draw fn(ctx) with a strong coloured rim: mask copy offset behind the real drawing."""
    ctx.push_group()
    fn(ctx)
    pat = ctx.pop_group()
    ctx.save()
    ctx.translate(*offset)
    _sc(ctx, color, a)
    ctx.mask(pat)
    ctx.restore()
    ctx.set_source(pat)
    ctx.paint()


def flash(ctx, a, col="#FFFFFF"):
    if a > 0.003:
        _sc(ctx, col, clamp(a))
        ctx.paint()


def steam(ctx, t, x, y, s, t0, seed=0, n=10, dirx=1.0):
    for i in range(n):
        st = t0 + 0.03 * i
        u = (t - st) / 0.55
        if not 0 <= u <= 1:
            continue
        ang = -math.pi / 2 + (_h(i, seed) - 0.5) * 1.8 + 0.5 * dirx
        d = 160 * s * ease_out(u) * (0.5 + _h(i, seed, 1))
        r = (18 + 50 * u) * s * (0.6 + 0.6 * _h(i, seed, 2))
        cx, cy = x + math.cos(ang) * d, y + math.sin(ang) * d - 40 * s * u
        g = cairo.RadialGradient(cx, cy, 0, cx, cy, r)
        g.add_color_stop_rgba(0, 1, 1, 1, 0.55 * (1 - u))
        g.add_color_stop_rgba(1, 1, 1, 1, 0)
        ctx.set_source(g)
        ctx.arc(cx, cy, r, 0, math.tau)
        ctx.fill()


def spark_burst(ctx, x, y, s, k, seed=0, n=7, col="#FFF6A0"):
    """Small radiating clamp sparks; k 0..1 strength."""
    if k <= 0.01:
        return
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i in range(n):
        a = -math.pi * (0.1 + 0.8 * _h(i, seed))
        L = (30 + 70 * _h(i, seed, 1)) * s * (0.4 + 0.6 * k)
        ctx.move_to(x + math.cos(a) * L * 0.3, y + math.sin(a) * L * 0.3)
        ctx.line_to(x + math.cos(a) * L, y + math.sin(a) * L)
        ctx.set_line_width(3.0 * s)
        _sc(ctx, col, k)
        ctx.stroke()
    draw_sparkle(ctx, x, y, 40 * s * k, k, 0.3, "#FFFFFF")


def shake(t, t_hit, amp, dur=0.3, seed=0):
    u = t - t_hit
    if u < 0 or u > dur:
        return 0.0, 0.0
    d = amp * (1 - u / dur) ** 2
    return d * math.sin(u * 95 + seed), d * math.cos(u * 77 + seed * 2)


# ------------------------------------------------------------------------------------------ montage
def shot_a(t):
    """Extreme low angle hero intro, scan flash."""
    s, ctx = core.new_surface()
    u = t / 0.95
    sc = lerp(2.55, 2.95, ease_in_out(u))
    ctx.save()
    ctx.translate(960, 540)
    ctx.rotate(-0.13 + 0.03 * u)
    ctx.translate(-960, -540)
    speed_lines(ctx, t, (980, 260), "#2C7F8F", "#0B1B2E", n=90, seed=1, alpha=0.5)
    frozen_rain(ctx, t, n=90, seed=4, alpha=0.35)
    rx, ry = 1010, 1080 + 330 * sc * 0.55
    face = "scan" if t > 0.08 else "surprised"
    pose = dict(view="three_quarter_left", face=face, screen_glow=1.0, t=T0 + t, antenna_blink=1.0,
                arms="rest", wet=0.25, bob=-6 * math.sin(min(1, t / 0.2) * math.pi), rim="#9FF7FF")
    rim_light_draw(ctx, lambda c: draw_robo(c, rx, ry, sc, **pose), (-9, -6), "#8FF6FF")
    rim_light_draw(ctx, lambda c: None, (0, 0))
    # warm kicker from the other side
    ctx.restore()
    # scan beam sweeping down the frame
    if 0.15 < t < 0.75:
        k = (t - 0.15) / 0.6
        yb = lerp(-60, 1140, ease_in_out(k))
        g = cairo.LinearGradient(0, yb - 90, 0, yb + 10)
        g.add_color_stop_rgba(0, 0.45, 0.95, 1, 0)
        g.add_color_stop_rgba(0.85, 0.45, 0.95, 1, 0.35)
        g.add_color_stop_rgba(1, 0.85, 1, 1, 0.9)
        ctx.set_source(g)
        ctx.rectangle(0, yb - 90, W, 100)
        ctx.fill()
        for i in range(0, H, 6):
            pass
    # scanlines texture during the scan
    if t < 0.8:
        ctx.set_line_width(1.0)
        for yy in range(0, H, 6):
            ctx.move_to(0, yy + 0.5)
            ctx.line_to(W, yy + 0.5)
        _sc(ctx, "#0B1B2E", 0.12)
        ctx.stroke()
    frozen_rain(ctx, t, n=30, seed=11, alpha=0.6, length=(80, 160))
    flash(ctx, 1 - t / 0.12, "#E9FFFF")
    return s, 0.62, 0.8


def shot_b(t):
    """Close-up: back panel doors split open with a clack; steam + sparkles."""
    u = t - 0.95
    s, ctx = core.new_surface()
    sc = 4.4
    dx, dy = shake(u, 0.0, 26)
    parallel_lines(ctx, u, -0.25, "#1F6577", "#0D2135", n=55, seed=5, speed=2600, alpha=0.35)
    frozen_rain(ctx, t, n=60, seed=6, alpha=0.3, length=(50, 120))
    can = 0.1 * ease_out(clamp(u / 0.22), 4) + 0.035 * smoothstep(0.22, 0.475, u)
    # panel base is at local (f=-64, v=-176): with side_left view, the back is to screen-right
    rx = 720 + dx - 60 * u
    ry = 1430 + dy
    pose = dict(view="side_left", face="dots", t=T0 + t, canopy=can, wet=0.3, antenna_blink=1.0,
                rim="#9FF7FF", bob=4 * math.sin(u * 30) * (1 - clamp(u / 0.3)))
    a = draw_robo(ctx, rx, ry, sc, **pose)
    bx, by = rx + 64 * sc, ry - 176 * sc
    steam(ctx, u, bx, by, 1.6, 0.02, seed=3, n=12, dirx=1)
    for i in range(5):
        st = 0.04 + 0.06 * i
        k = clamp(1 - abs(u - st - 0.08) / 0.12)
        if k > 0:
            draw_sparkle(ctx, bx + (_h(i, 2) - 0.3) * 260, by + (_h(i, 3) - 0.5) * 360, 60 * k, k, 0.4 * i, "#FFFFFF")
    frozen_rain(ctx, t, n=20, seed=12, alpha=0.5, length=(90, 170))
    flash(ctx, 0.85 - u / 0.08)
    return s, 0.6, 0.9


def shot_c(t):
    """Telescoping pole extends upward; camera tilts up following the tip."""
    u = (t - 1.425) / (2.15 - 1.425)
    s, ctx = core.new_surface()
    can = lerp(0.135, 0.3, ease_in_out(clamp(u * 1.05)))
    # two "ka-chunk" segment locks on the beat (1.425, 1.9)
    sx, sy = shake(t, 1.425, 14, seed=1)
    sx2, sy2 = shake(t, 1.9, 12, seed=2)
    sc = 1.75
    tilt = lerp(0, 520, ease_in_out(u))
    ctx.save()
    ctx.translate(960 + sx + sx2, 540 + sy + sy2)
    ctx.rotate(0.22)
    ctx.translate(-960, -540)
    parallel_lines(ctx, t, math.pi / 2, "#2A8C93", "#0C2236", n=70, seed=7, speed=3800, alpha=0.4)
    frozen_rain(ctx, t, n=70, seed=8, alpha=0.3)
    rx, ry = 1330, 1180 + tilt
    pose = dict(view="three_quarter_left", face="happy", t=T0 + t, canopy=can, wet=0.3, antenna_blink=1.0,
                rim="#9FF7FF")
    holder = {}

    def dr(c):
        holder.update(draw_robo(c, rx, ry, sc, **pose))
    rim_light_draw(ctx, dr, (-7, -6), "#9FF6FF")
    ap = holder.get("canopy_apex")
    if ap:
        for i, tb in enumerate((1.425, 1.9, 2.1)):
            k = clamp(1 - abs(t - tb - 0.06) / 0.14)
            if k > 0:
                draw_sparkle(ctx, ap[0], ap[1], 110 * k, k, 0.5 * i, "#FFFFFF")
        draw_sparkle(ctx, ap[0], ap[1], 36 + 10 * math.sin(t * 40), 0.9, t * 3, "#DFFFFF")
    ctx.restore()
    frozen_rain(ctx, t, n=20, seed=13, alpha=0.5, length=(90, 170))
    flash(ctx, 0.75 - (t - 1.425) / 0.08)
    return s, 0.6, 1.0


RIB_T = [2.15 + 0.09 + 0.19 * i for i in range(8)]   # "shk-tink" per rib (sfx)


def shot_d(t):
    """Ribs unfold one by one; camera circles under the canopy."""
    u = (t - 2.15) / (3.4 - 2.15)
    s, ctx = core.new_surface()
    can = lerp(0.3, 0.665, clamp((t - 2.15) / 1.4))
    sc = 1.12
    ax, ay = 960, 330
    rot = lerp(0.42, -0.12, ease_in_out(u))
    zoom = lerp(1.0, 0.92, ease_in_out(u))
    ctx.save()
    ctx.translate(ax, ay)
    ctx.rotate(rot)
    ctx.scale(zoom, zoom)
    ctx.translate(-ax, -ay)
    speed_lines(ctx, t, (ax, ay), "#3FA9A3", "#0E2A3A", n=96, seed=2, alpha=0.45, inner=150, rot=t * 0.4)
    frozen_rain(ctx, t, n=80, seed=10, alpha=0.3)
    rx, ry = ax + 470 * sc, ay + 765 * sc
    pose = dict(view="three_quarter_left", face="happy", t=T0 + t, canopy=can, wet=0.35, antenna_blink=1.0,
                rim="#9FF7FF", bob=3 * math.sin(t * 13))
    holder = {}

    def dr(c):
        holder.update(draw_robo(c, rx, ry, sc, **pose))
    rim_light_draw(ctx, dr, (-6, -6), "#9FF6FF")
    tips = holder.get("canopy_rim_points") or []
    for i, tb in enumerate(RIB_T):
        k = clamp(1 - abs(t - tb) / 0.13)
        if k > 0 and i < len(tips):
            q = tips[(i * 3) % len(tips)]
            draw_sparkle(ctx, q[0], q[1], 95 * k, k, 0.3 * i, "#FFFFFF")
            draw_sparkle(ctx, q[0], q[1], 40 * k, k, 0.3 * i + 0.8, "#BFFBFF")
    ap = holder.get("canopy_apex")
    if ap:
        draw_sparkle(ctx, ap[0], ap[1], 34, 0.8, t * 2, "#FFFFFF")
    ctx.restore()
    frozen_rain(ctx, t, n=20, seed=14, alpha=0.45, length=(90, 170))
    flash(ctx, 0.7 - (t - 2.15) / 0.08)
    return s, 0.6, 1.0


def shot_e(t):
    """Low-angle hero pose; treads clamp-lock (ka-chunk at 3.55) with an impact frame."""
    u = t - 3.4
    s, ctx = core.new_surface()
    hit = 3.55
    sx, sy = shake(t, hit, 30, 0.25, 3)
    can = lerp(0.665, 0.7, clamp((t - 3.4) / 0.2))
    sq = 0.0
    if t >= hit:
        sq = 0.16 * math.exp(-(t - hit) * 14) * math.cos((t - hit) * 40)
    else:
        sq = -0.05 * smoothstep(3.4, hit, t)        # anticipation stretch
    ctx.save()
    ctx.translate(960 + sx, 540 + sy)
    ctx.rotate(0.08)
    ctx.translate(-960, -540)
    impact = t >= hit and t < hit + 0.05
    if impact:      # inverted anime impact frame
        _sc(ctx, "#F8FFFF")
        ctx.paint()
        speed_lines(ctx, t, (1000, 560), "#FFFFFF", "#CFEFF2", n=70, seed=8, alpha=0.9, col="#0E2A3A", inner=320)
    else:
        speed_lines(ctx, t, (1000, 520), "#46B3AA", "#0B1F30", n=100, seed=6, alpha=0.5, inner=330)
    frozen_rain(ctx, t, n=60, seed=15, alpha=0.3)
    sc = 1.35
    rx, ry = 1180, 1050
    pose = dict(view="three_quarter_left", face="happy" if t < hit else "heart" if False else "happy", t=T0 + t,
                canopy=can, wet=0.4, antenna_blink=1.0, rim="#9FF7FF", squash=sq, arms="salute",
                hand="screen_right", screen_glow=1.0)
    holder = {}

    def dr(c):
        holder.update(draw_robo(c, rx, ry, sc, **pose))
    rim_light_draw(ctx, dr, (-8, -7), "#9FF6FF")
    k = clamp(1 - (t - hit) / 0.2) if t >= hit else 0
    for j, key in enumerate(("tread_l", "tread_r")):
        p = holder.get(key)
        if p:
            spark_burst(ctx, p[0], p[1] - 10, 1.6, k, seed=j + 4)
    # water ring kicked up by the clamp
    if t >= hit:
        v = clamp((t - hit) / 0.25)
        ctx.save()
        ctx.translate(rx, ry + 10)
        ctx.scale(1, 0.22)
        ctx.arc(0, 0, 120 + 380 * ease_out(v), 0, math.tau)
        ctx.restore()
        ctx.set_line_width(6 * (1 - v))
        _sc(ctx, "#E8F6FF", 0.8 * (1 - v))
        ctx.stroke()
    ctx.restore()
    frozen_rain(ctx, t, n=20, seed=16, alpha=0.5, length=(90, 170))
    flash(ctx, 0.6 - u / 0.08)
    return s, 0.6, 1.0


# ------------------------------------------------------------------------------------------ live shots
def canopy_p(t):
    if t < 3.8:
        return 0.7
    return lerp(0.7, 1.0, clamp((t - 3.8) / 0.42))


def robo_wet(t):
    return lerp(0.4, 0.65, smoothstep(3.8, 6.5, t)) if t < 6.5 else lerp(0.65, 1.0, smoothstep(6.5, 7.5, t))


def sparks_amt(t):
    if t < SPARK_T:
        return 0.0
    u = t - SPARK_T
    burst = math.exp(-u * 3.5)
    flick = 0.35 + 0.3 * (noise1(t * 9, 3) * 0.5 + 0.5)
    pop = 0.6 if _h(int(t * 8), 77) > 0.82 else 0.0
    return clamp(max(burst, flick + pop))


def robo_face(t):
    if t < SPARK_T:
        return "happy"
    if t < SPARK_T + 0.12:
        return "glitch"
    # flicker between happy and glitch, glitchier over time
    k = int(t * 10)
    p = 0.25 + 0.2 * smoothstep(8.4, 10.6, t)
    return "glitch" if _h(k, 91) < p else "happy"


def robo_pose(t, tg):
    jit = 0.0
    jx = 0.0
    if t >= SPARK_T:
        amp = 1.0 * math.exp(-(t - SPARK_T) * 2.5) + 0.25
        jx = (_h(int(t * FPS), 5) - 0.5) * 10 * amp
        jit = (_h(int(t * FPS), 6) - 0.5) * 0.06 * amp
    bulb = True
    if t >= SPARK_T:
        bulb = 1.0 if _h(int(t * 16), 12) > 0.45 else 0.1
    # whump recoil at 3.8, then happy little bob (he's proud)
    sq = 0.0
    if 3.8 <= t < 4.6:
        sq = 0.12 * math.exp(-(t - 3.8) * 6) * math.cos((t - 3.8) * 22)
    bob = 3 * math.sin(t * 4.5) if t < SPARK_T else 0
    return dict(view="three_quarter_left", face=robo_face(t), t=tg, canopy=canopy_p(t), wet=robo_wet(t),
                sparks=sparks_amt(t), antenna_blink=bulb, arms="hold_handle", squash=sq, bob=bob,
                body_tilt=jit + 0.03 * math.sin(t * 2.2), head_tilt=0.05 * math.sin(t * 1.7) - (0.1 if t > 8.4 else 0),
                look=(0.6, -0.2), screen_glow=1.0), jx


def mina_pose(t, tg):
    o, v, talking = core.mouth("mina", tg)
    pose = dict(view="three_quarter_right", t=tg, wet=0.3, mouth_open=o, vowel=v,
                blink=core.blink(tg, 3), wind=0.6)
    if t < 4.4:      # whump: startled pop then amazed look up
        u = t - 3.8
        pose.update(expr="surprised" if u < 0.15 else "amazed", head_pitch=lerp(0.2, 0.75, ease_out(clamp(u / 0.3))),
                    bounce=0.18 * math.exp(-u * 7) * math.cos(u * 25), arms="hands_clasped" if u > 0.2 else "rest",
                    mouth_open=0.5 if u > 0.15 else 0.25, vowel="o", look=(0.2, -1))
    elif t < 6.5:
        u = t - 4.4
        line = L07 - 0.05 <= t <= L07 + L07_DUR + 0.05
        expr = "amazed" if t < L07 + 0.75 else "happy"
        yaw = None
        if t > 6.02:     # little twirl after the line
            k = ease_in_out(clamp((t - 6.02) / 0.45))
            yaw = 0.6 + k * math.tau
            pose["view"] = None
        pose.update(expr=expr, head_pitch=0.6 if t < L07 else 0.35, look=(0.1, -1) if t < L07 + 0.7 else (0.3, -0.4),
                    arms="arms_up" if t > L07 - 0.15 else "hands_clasped",
                    arm_phase=tg * 2.4, bounce=0.06 * math.sin(tg * 9) if t > L07 - 0.15 else 0.02 * math.sin(tg * 3),
                    tiptoe=0.5 if t > L07 - 0.15 else 0.2)
        if not line:
            pose.update(mouth_open=0.35 if t < L07 else 0.15, vowel="o" if t < L07 else "a")
        if yaw is not None:
            pose.pop("view")
            pose["yaw"] = yaw
            pose["wind"] = 1.6
    else:
        if t < 7.85:     # still beaming at the canopy, oblivious
            pose.update(expr="happy", head_pitch=0.45, look=(0, -0.8), arms="hands_clasped",
                        bounce=0.03 * math.sin(tg * 4))
        elif t < 8.25:   # take: head snaps toward Robo
            u = t - 7.85
            pose.update(expr="surprised", head_turn=0.25, head_pitch=0.1, look=(1, 0),
                        bounce=-0.12 * math.exp(-u * 10), mouth_open=0.3, vowel="o")
        else:
            line = L08 - 0.05 <= t <= L08 + L08_DUR + 0.05
            pose.update(expr="worried", head_turn=0.25, head_pitch=0.15, look=(1, -0.1),
                        arms="point" if t < 10.4 else "hands_clasped", hand="screen_right",
                        body_lean=0.08 * smoothstep(8.25, 8.7, t), tiptoe=0.3 * smoothstep(8.25, 8.6, t),
                        head_tilt=0.08 * math.sin(tg * 2.0), bounce=0.02 * math.sin(tg * 6))
            if not line:
                pose.update(mouth_open=0.08)
    return pose


def kitten_state(t):
    if t < 3.95:
        return KX0, KY0, "sit", 1, 0.0
    u = clamp((t - 3.95) / 0.5)
    e = ease_in_out(u)
    x = lerp(KX0, KX1, e)
    y = lerp(KY0, KY1, e) - 14 * abs(math.sin(u * math.pi * 3)) * (1 - u)
    pose = "stand" if u < 1 else "sit"
    return x, y, pose, 1, u


def cam_for(t):
    """(cx, cy, zoom) for the live shots."""
    if t < 4.4:      # F: wide on the whump
        u = t - 3.8
        cx, cy, z = 900, 470, lerp(0.86, 0.83, ease_out(clamp(u / 0.6)))
        sx, sy = shake(t, 3.8, 18, 0.35, 4)
        return cx + sx, cy + sy, z
    if t < 6.5:      # G: low angle close on Mina, slow push
        u = (t - 4.4) / 2.1
        return lerp(700, 690, u), lerp(500, 480, u), lerp(2.05, 2.25, ease_in_out(u))
    # H: snap pull back -> hold -> push into the two-shot
    u = clamp((t - 6.5) / 0.45)
    z = lerp(1.25, 0.84, ease_out(u, 4))
    cx = lerp(760, 905, ease_out(u, 4))
    cy = lerp(560, 500, ease_out(u, 4))
    p = ease_in_out(clamp((t - 8.0) / 1.3))
    z = lerp(z, 1.3, p) + 0.06 * smoothstep(9.3, 11, t)
    cx = lerp(cx, 905, p)
    cy = lerp(cy, 640, p)
    sx, sy = shake(t, SPARK_T, 10, 0.3, 6)
    dx = 6 * noise1(t * 0.6, 1)
    dy = 4 * noise1(t * 0.5, 2)
    return cx + sx + dx, cy + sy + dy, z


def shot_live(t, frame):
    tg = T0 + t
    G = 4.4 <= t < 6.5
    cam = cam_for(t)
    s, ctx = core.new_surface()
    dof = 12.0 if G else 0.0
    world.draw_shopfront(ctx, tg, WEATHER, cam, dof=dof)

    rpose, jx = robo_pose(t, tg)
    mpose = mina_pose(t, tg)
    kx, ky, kp, kf, ku = kitten_state(t)
    rx = RX + jx

    def chars(c, reflect=False):
        c.save()
        world._apply_cam(c, cam)
        if not G:
            draw_robo(c, rx, YS, SC, **dict(rpose, layer="body"))
        km = "mew" if (SPARK_T < t < SPARK_T + 0.5) else ("happy" if 4.6 < t < 7.5 else "neutral")
        draw_kitten(c, kx, ky, world.shop_scale(ky), pose=kp, expr=km, t=tg, wet=0.35, facing=kf,
                    look=(0.3, -0.6) if t < 7.6 else (1, -0.2), blink=core.blink(tg, 9))
        draw_mina(c, MX, YS, SC, **mpose)
        c.restore()

    if not G:
        world.draw_ground_reflection(ctx, tg, WEATHER, lambda c: chars(c, True), cam)
    world.draw_rain(ctx, tg, WEATHER, "bg", cam)
    # canopy (and pole) behind the characters
    ctx.save()
    world._apply_cam(ctx, cam)
    ca = draw_robo(ctx, rx, YS, SC, **dict(rpose, layer="canopy"))
    ctx.restore()
    chars(ctx)
    # canopy span in screen coords -> keep rain out from under it
    span = ca.get("canopy_span", (0, 0))
    center = ca.get("canopy_center", (MX, 300))
    rad = ca.get("canopy_radius", 0)
    sx0 = world.to_screen(span[0], 0, cam)[0]
    sx1 = world.to_screen(span[1], 0, cam)[0]
    rim_y = world.to_screen(0, center[1], cam)[1]
    regions = [(-200, -300, sx0, H + 300), (sx1, -300, W + 200, H + 300), (sx0, -300, sx1, rim_y - 40)]
    for lay in ("mid",):
        for r in regions:
            if r[2] > r[0] + 2:
                world.draw_rain(ctx, tg, WEATHER, lay, cam, region=r)
    # extra downpour hammering Robo (the gag) once revealed
    if t >= 6.5:
        rs = world.to_screen(rx, YS, cam)
        z = cam[2]
        reg = (max(sx1, rs[0] - 190 * z), rs[1] - 520 * z, rs[0] + 190 * z, rs[1])
        world.draw_rain(ctx, tg, WEATHER, "mid", cam, region=reg, density=1.4, seed=5)
    ctx.save()
    world._apply_cam(ctx, cam)
    if rad > 0:
        k = clamp((t - 3.85) / 0.3)
        world.draw_canopy_rain(ctx, tg, center, rad, WEATHER, ground_y=YS + 20, intensity=k, seed=21)
        draw_rim_drips(ctx, ca.get("canopy_rim_points", []), tg, SC, seed=2, amount=k,
                       fall=(YS - center[1]) / SC)
    if not G:
        ra = draw_robo(_NULLCTX(), rx, YS, SC, **dict(rpose, layer="body"))
        draw_drops_on(ctx, ra.get("drop_points", []), tg, SC, seed=4, amount=rpose["wet"])
        if t >= 6.5:
            ht = ra["head_top"]
            world.draw_drips_from(ctx, tg, [(ht[0] + (i - 2) * 30 * SC, ht[1] + 4) for i in range(5)],
                                  ht[1] + 4, period=(0.25, 0.5), size=4, weather=WEATHER, seed=8, rate=2.0) \
                if hasattr(world, "draw_drips_from") else None
            _head_splashes(ctx, tg, ht, SC)
    ctx.restore()
    world.draw_shopfront_fg(ctx, tg, WEATHER, cam)
    for r in regions:
        if r[2] > r[0] + 2:
            world.draw_rain(ctx, tg, WEATHER, "fg", cam, region=r)
    # whump: air shockwave from the canopy + flash
    if 3.8 <= t < 4.2:
        u = (t - 3.8) / 0.4
        cs = world.to_screen(*center, cam)
        ctx.save()
        ctx.translate(*cs)
        ctx.scale(1, 0.35)
        ctx.arc(0, 0, (rad * cam[2]) * (1.0 + 0.7 * ease_out(u)), 0, math.tau)
        ctx.restore()
        ctx.set_line_width(14 * (1 - u))
        _sc(ctx, "#FFFFFF", 0.5 * (1 - u))
        ctx.stroke()
        flash(ctx, 0.55 - (t - 3.8) / 0.1)
    if G:
        flash(ctx, 0.3 - (t - 4.4) / 0.08)
    if 6.5 <= t < 6.6:
        flash(ctx, 0.25)
    if SPARK_T <= t < SPARK_T + 0.09:
        flash(ctx, 0.35, "#FFF6C0")
    return s, cam, rx, rpose


class _NULLCTX:
    """Anchor-only cairo context for re-querying Robo anchors in the current cam space."""
    def __new__(cls):
        s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
        return cairo.Context(s)


def _head_splashes(ctx, tg, ht, s):
    """Rain bouncing off Robo's head: little crowns + droplets."""
    for i in range(7):
        per = 0.22 + 0.15 * _h(i, 31)
        u = ((tg + _h(i, 32) * per) % per) / per
        x = ht[0] + (_h(i, 33, int(tg / per)) - 0.5) * 120 * s
        y = ht[1] + 6 * s
        if u < 0.5:
            k = u / 0.5
            for j in (-1, 1):
                ctx.move_to(x, y)
                ctx.line_to(x + j * 16 * s * k, y - 14 * s * math.sin(k * math.pi))
            ctx.set_line_width(2.0 * s)
            _sc(ctx, "#EAF6FF", 0.8 * (1 - k))
            ctx.stroke()
            draw_droplet(ctx, x + 10 * s * k, y - 20 * s * math.sin(k * math.pi), 2.4 * s, 0.8 * (1 - k))


def glitch_frame(img, t):
    """One-frame digital glitch: band shifts + RGB split."""
    k = int(t * FPS)
    out = img.copy()
    for b in range(9):
        y0 = int(_h(b, k, 1) * H)
        hh = int(20 + 90 * _h(b, k, 2))
        sh = int((_h(b, k, 3) - 0.5) * 160)
        out[y0:y0 + hh] = np.roll(img[y0:y0 + hh], sh, axis=1)
    out[..., 0] = np.roll(out[..., 0], 9, axis=1)
    out[..., 2] = np.roll(out[..., 2], -9, axis=1)
    return out


def render(t: float) -> np.ndarray:
    frame = int(round(t * FPS))
    if t < 0.95:
        s, thr, st = shot_a(t)
    elif t < 1.425:
        s, thr, st = shot_b(t)
    elif t < 2.15:
        s, thr, st = shot_c(t)
    elif t < 3.4:
        s, thr, st = shot_d(t)
    elif t < 3.8:
        s, thr, st = shot_e(t)
    else:
        s, cam, rx, rpose = shot_live(t, frame)
        thr, st = 0.74, 0.5
        if t >= SPARK_T:   # spark flashes bloom harder
            thr, st = 0.7, 0.5 + 0.4 * sparks_amt(t)
    img = core.surface_rgba(s)
    img = post(img, frame, thr, st, vig=0.36 if t < 3.8 else 0.32)
    if SPARK_T + 0.02 <= t < SPARK_T + 0.09 or (9.1 <= t < 9.14):
        img = glitch_frame(img, t)
    return core.finish(img)
