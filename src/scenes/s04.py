"""s04 — The journey up the shrine steps (global 25.0–37.0 s, 288 frames).

Shots (local seconds):
  A 0.00–4.00  MCU Haru in the night garden cupping Kira, says l05; anticipation,
               then bolts screen-right (whip pan).
  B 4.00–6.00  Side-scrolling run up the stone steps (whip-in), torii sweep past.
  C 6.00–9.10  Close-up: Kira riding in Haru's cupped hands, streaking background, l06.
  D 9.10–10.00 Back to the side run; Kira dim.
  E 10.00–11.40 Haru face close-up, sweat, glance down, 「もうすこし！」 (l07 @10.6).
  F 11.40–12.00 Wide: she bursts through the last torii, violet sky ahead.
"""
from __future__ import annotations

import math
import sys
import pathlib

import cairo
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import core, world  # noqa: E402
from common.core import (clamp, lerp, smoothstep, ease_in_out, ease_out, ease_in, noise1, invlerp,  # noqa: E402
                         W, H, FPS, TAU)
from common.characters import draw_haru, draw_kira, draw_kira_glow  # noqa: E402

T0 = 25.0
STEP_T0 = 0.12          # first footstep SFX (local)
STEP_RATE = 3.0         # footsteps per second (matches sfx.py)
RUN_S = 0.8             # Haru scale in the side-run shots
RUN_ZOOM = 1.18        # camera zoom on the side-run world

# ============================================================== global timing helpers


def phase_at(t):
    """Time-of-day phase: deep night -> first faint violet at the end."""
    return lerp(0.03, 0.2, smoothstep(5.0, 12.0, t))


def kira_glow(t):
    """Kira's life: 0.6 -> 0.3 with flickers / dips."""
    base = lerp(0.6, 0.3, smoothstep(4.0, 10.0, t))
    if t < 4.0:
        base = 0.6 + 0.03 * math.sin(t * 5.0)
    fl = 0.06 * noise1(t * 9.0, 11) + 0.04 * noise1(t * 23.0, 12)
    dip = 0.0
    for td, amp, w in ((2.9, 0.12, 0.10), (5.3, 0.18, 0.09), (7.2, 0.14, 0.12), (8.6, 0.2, 0.08),
                       (9.6, 0.12, 0.08), (11.0, 0.1, 0.1)):
        dip = max(dip, amp * math.exp(-((t - td) / w) ** 2))
    return clamp(base + fl * smoothstep(3.5, 5.0, t) - dip, 0.12, 1.0)


# ------------------------------------------------------------- hero world x (continuous run)
V0, V1 = 320.0, 360.0       # px/s along the stairs
X_START = 880.0             # world x at t = 4.0 (set so a torii is passed near the end)


def _speed(t):
    return lerp(V0, V1, smoothstep(4.0, 11.5, t))


def hero_x(t):
    """World x of Haru (integral of speed, analytic enough via fine sum)."""
    if t <= 4.0:
        return X_START - 150.0 * (4.0 - t)
    n = max(1, int((t - 4.0) * 24))
    dt = (t - 4.0) / n
    s = 0.0
    for i in range(n):
        s += _speed(4.0 + (i + 0.5) * dt) * dt
    return X_START + s


def run_phase(t):
    # contacts at run_phase 0.25 / 0.75 -> line them up with footstep SFX times
    return (t - STEP_T0) * STEP_RATE * 0.5 + 0.25


def _tread_center(wx):
    k = math.floor(wx / world.TREAD)
    return (k + 0.5) * world.TREAD


def hero_ground(t):
    """World y of the ground point under Haru: each footfall lands on a tread; between
    footfalls interpolate (the rig does the hip bob)."""
    k = math.floor((t - STEP_T0) * STEP_RATE)
    ta = STEP_T0 + k / STEP_RATE
    tb = ta + 1 / STEP_RATE
    reach = 34.0 * RUN_S
    ya = world.steps_height_at(hero_x(ta) + reach)
    yb = world.steps_height_at(hero_x(tb) + reach)
    u = (t - ta) * STEP_RATE
    return lerp(ya, yb, ease_in_out(clamp(u / 0.8)))


def cam_x_at(t):
    """Camera: lags at the start (she runs ahead of centre) then leads a bit; handheld drift."""
    hx = hero_x(t)
    off = lerp(-260.0, 60.0, ease_out(clamp((t - 4.0) / 2.2), 2))  # hero screen x offset from 960
    if t > 9.0:
        off = lerp(off, -40.0, smoothstep(9.1, 10.0, t))
    if t > 11.4:
        off = lerp(-40.0, 180.0, ease_in(clamp((t - 11.4) / 0.6), 2))
    off += 24 * noise1(t * 0.9, 4)
    return hx - 960.0 - off


# ============================================================== utilities


def motion_blur_h(img, dist, taps=10, ds=4):
    """Horizontal motion blur of an (H,W,C) float array by `dist` px (cheap: low-res taps)."""
    if abs(dist) < 1.5:
        return img
    h, w, c = img.shape
    small = img[::ds, ::ds]
    acc = np.zeros_like(small)
    sd = dist / ds
    for i in range(taps):
        off = int(round(sd * (i / (taps - 1) - 0.5)))
        acc += np.roll(small, off, axis=1)
    acc /= taps
    up = np.repeat(np.repeat(acc, ds, axis=0), ds, axis=1)[:h, :w]
    return up


def soft_blur_h(img, dist, taps=12):
    """Full-res horizontal blur (for the whip pan)."""
    if abs(dist) < 1.5:
        return img
    acc = np.zeros_like(img)
    for i in range(taps):
        off = int(round(dist * (i / (taps - 1) - 0.5)))
        acc += np.roll(img, off, axis=1)
    return acc / taps


def surf_from_rgba(a):
    return core.rgba_surface(a)


def kira_held(kpose, glow, t, store, scale_mul=1.0, pool=True):
    """held-callback factory: draws Kira's halo + body at the cup, records screen position."""
    def cb(ctx, x, y, s):
        ks = 0.4 * s * scale_mul
        store["pos"] = ctx.user_to_device(x, y - 6 * s)
        store["s"] = ks
        draw_kira_glow(ctx, x, y - 6 * s, ks, glow, t)
        draw_kira(ctx, x, y - 6 * s, ks, glow=glow, t=t, **kpose)
    return cb


def glow_blob(pos, r, strength, color="#FFE3A0"):
    """Small bright layer used as bloom seed for Kira's light."""
    s, ctx = core.new_surface()
    x, y = pos
    c = core.rgb(color)
    g = core.radial_gradient(x, y, 0, r, [(0, c, strength), (0.4, c, strength * 0.5), (1, c, 0)])
    ctx.set_source(g)
    ctx.arc(x, y, r, 0, TAU)
    ctx.fill()
    return core.surface_rgba(s)


def haru_anchor(pose, x=0.0, y=0.0, scale=1.0):
    """Dry-run draw_haru on a tiny surface to get its anchors (for framing)."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 4, 4)
    ctx = cairo.Context(s)
    return draw_haru(ctx, x, y, scale, **pose)


def draw_sweat(ctx, x, y, size, alpha=1.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(size, size)
    ctx.move_to(0, -1.3)
    ctx.curve_to(0.55, -0.35, 0.8, 0.25, 0.55, 0.75)
    ctx.curve_to(0.3, 1.1, -0.3, 1.1, -0.55, 0.75)
    ctx.curve_to(-0.8, 0.25, -0.55, -0.35, 0, -1.3)
    ctx.close_path()
    ctx.set_source_rgba(0.78, 0.9, 1.0, 0.85 * alpha)
    ctx.fill_preserve()
    ctx.set_line_width(0.12)
    ctx.set_source_rgba(0.25, 0.35, 0.6, 0.9 * alpha)
    ctx.stroke()
    ctx.move_to(-0.25, 0.1)
    ctx.curve_to(-0.35, 0.4, -0.2, 0.6, 0.0, 0.7)
    ctx.set_line_width(0.14)
    ctx.set_source_rgba(1, 1, 1, 0.9 * alpha)
    ctx.stroke()
    ctx.restore()


def speed_lines(ctx, t, n, region, seed, alpha, length=(160, 420), color=(0.85, 0.9, 1.0), vx=-2600):
    x0, y0, x1, y1 = region
    rng = np.random.default_rng(seed)
    ys = rng.uniform(y0, y1, n)
    xs = rng.uniform(0, 1, n)
    ls = rng.uniform(*length, n)
    sp = rng.uniform(0.7, 1.3, n)
    ww = x1 - x0 + 600
    ctx.save()
    for i in range(n):
        x = x0 - 300 + (xs[i] * ww + vx * sp[i] * t) % ww
        a = alpha * (0.4 + 0.6 * rng.random())
        g = cairo.LinearGradient(x, 0, x + ls[i], 0)
        g.add_color_stop_rgba(0, *color, 0)
        g.add_color_stop_rgba(0.7, *color, a)
        g.add_color_stop_rgba(1, *color, 0)
        ctx.set_source(g)
        ctx.set_line_width(rng.uniform(1.5, 3.5))
        ctx.move_to(x, ys[i])
        ctx.line_to(x + ls[i], ys[i])
        ctx.stroke()
    ctx.restore()


# ============================================================== shot A: garden MCU
_GARDEN = {}


def garden_backdrop():
    if "bg" not in _GARDEN:
        s, ctx = core.new_surface()
        world.draw_garden(ctx, 2.0, 0.03, cam_x=260, light=(1000, 820, 260, "#FFE9A8", 0.5))
        a = core.surface_rgba(s)
        a = core.blur(a, 10)
        a[..., :3] *= 0.72
        _GARDEN["bg"] = core.rgba_surface(a)
    return _GARDEN["bg"]


def shot_a(t, tg):
    s, ctx = core.new_surface()
    # push-in on the backdrop + slight handheld
    zoom = 1.08 + 0.05 * ease_in_out(t / 3.4)
    hx = 6 * noise1(t * 0.7, 1)
    hy = 4 * noise1(t * 0.6, 2)
    # whip: camera pans right fast at the end
    whip = ease_in(clamp((t - 3.62) / 0.38), 2.2)
    pan = 1400 * whip
    ctx.save()
    ctx.translate(960 + hx - pan * 0.5, 540 + hy)
    ctx.scale(zoom, zoom)
    ctx.translate(-960, -540)
    ctx.set_source_surface(garden_backdrop(), 0, 0)
    ctx.paint()
    ctx.restore()
    # bokeh fireflies (out of focus: big, soft)
    world.draw_fireflies(ctx, 2.0 + t, 16, (-100 - pan * 0.6, 0, 2020 - pan * 0.6, 1000), seed=21, size=2.6,
                         alpha=0.55, wander=90)

    # ---- Haru
    g = kira_glow(t)
    mo, vo, talking = core.mouth("haru", tg)
    # expression beats: determined -> happy smile on "いっしょにいこう"
    expr = "determined" if t < 1.55 else "happy"
    # anticipation: 3.25-3.62 dip + turn to screen-right, then bolt
    ant = smoothstep(3.2, 3.6, t)
    yaw = lerp(math.radians(38), math.radians(90), smoothstep(3.3, 3.7, t))
    bolt = ease_in(clamp((t - 3.6) / 0.4), 2.0)
    base_x = 900 + 20 * math.sin(t * 0.8)
    x = base_x - 30 * ant + 1700 * bolt - pan * 0.25
    scale = 3.2
    breath = 5 * math.sin(t * TAU / 3.0)
    y = 1760 + breath + 40 * ant * (1 - bolt) - 30 * bolt
    # head: looks down at Kira while talking, lifts with the smile
    pitch = lerp(-0.25, 0.05, smoothstep(1.2, 1.9, t)) + 0.1 * ant
    look = (lerp(0.1, 0.35, smoothstep(1.2, 1.9, t)), lerp(0.45, 0.0, smoothstep(1.2, 1.9, t)))
    if 2.4 < t < 3.2:
        look = (lerp(look[0], 0.0, smoothstep(2.4, 2.7, t)), lerp(0.0, 0.5, smoothstep(2.4, 2.7, t)))
        pitch = lerp(pitch, -0.2, smoothstep(2.4, 2.7, t))
    if t >= 3.2:
        look = (0.6, 0.0)
        pitch = 0.0
    tilt = 0.06 * math.sin(t * 1.3) + (0.1 if 1.5 < t < 2.8 else 0.0) * smoothstep(1.5, 1.8, t) * (1 - smoothstep(2.6, 2.8, t))
    kstore = {}
    kpose = dict(expr="happy" if 1.8 < t < 3.0 else "neutral", look=(-0.2, -0.8), blink=core.blink(tg, 7),
                 squash=0.12 * math.sin(t * 3.0) - 0.25 * ant * (1 - bolt), rot=0.05 * math.sin(t * 1.7))
    held = kira_held(kpose, g, t, kstore)
    rim = "#FFC27A"
    anc = draw_haru(ctx, x, y, scale, yaw=yaw, expr=expr, mouth_open=mo, vowel=vo,
                    blink=core.blink(tg, 3), look=look, head_pitch=pitch, head_tilt=tilt,
                    body_lean=-0.05 * ant + 0.22 * bolt, crouch=0.0, arms="cup_hands", t=tg, held=held,
                    crop="bust", rim=rim, light="right", wind=0.6 + 1.5 * bolt, lantern=False,
                    bounce=0.08 * ant * (1 - bolt) - 0.1 * bolt)
    # Kira's warm light on her hands / chin
    if "pos" in kstore:
        kx, ky = kstore["pos"]
        world.draw_light_pool(ctx, kx, ky - 60, 380, 330, "#FFD58A", 0.32 * g / 0.6)
    img = core.surface_rgba(s)
    if "pos" in kstore:
        core.add_glow(img, glow_blob(kstore["pos"], 110, 0.9 * g), 40, 1.0)
    if whip > 0:
        img = soft_blur_h(img, 30 + 520 * whip, taps=14)
    return img


# ============================================================== side-run shots (B, D, F)
FG_TORII_EVERY = 2300.0
FG_PAR = 1.6


def _fg_torii_layer(t, cam_x, phase, burst=0.0):
    """Big, fast foreground torii pillars (parallax > 1), motion-blurred."""
    s, ctx = core.new_surface()
    shift = cam_x * FG_PAR
    drawn = False
    k0 = math.floor((shift - 1200) / FG_TORII_EVERY)
    for k in range(k0, k0 + 3):
        sx = 400 + k * FG_TORII_EVERY - shift
        if -1400 < sx < W + 1400:
            base = 1320 - 0.25 * (sx - 960) * world.SLOPE
            world.draw_torii(ctx, sx, base, 900, 1850, phase=phase, base_r=base - 980 * world.SLOPE * 0.25)
            drawn = True
    if not drawn:
        return None
    a = core.surface_rgba(s)
    a[..., :3] *= 0.55  # foreground silhouette, darker
    return a


def side_run(t, tg, shot):
    phase = phase_at(t)
    cam_x = cam_x_at(t)
    hx = hero_x(t)
    gy = hero_ground(t)
    sx, sy = world.steps_to_screen(hx, gy, cam_x)
    s, ctx = core.new_surface()
    # camera shake from footfalls
    ft = ((t - STEP_T0) * STEP_RATE) % 1.0
    shake = 3.0 * math.exp(-ft * 6)
    ctx.save()
    ctx.translate(960, 640 + shake)
    ctx.scale(RUN_ZOOM, RUN_ZOOM)
    ctx.translate(-960, -640)
    world.draw_steps_path(ctx, cam_x, t, phase)
    # violet glow of the coming dawn up ahead (upper right)
    va = smoothstep(9.0, 12.0, t)
    if va > 0:
        world.draw_light_pool(ctx, 2000, 0, 1400, 750, "#9C86C8", 0.55 * va)
        world.draw_light_pool(ctx, 2050, -40, 800, 420, "#E7A6A0", 0.3 * va)
    burst = smoothstep(11.5, 12.0, t)
    # fireflies streaming past
    fly = core.draw_layer(world.draw_fireflies, t, 30, (-40, 60, 1960, 1040), seed=33, size=1.1,
                          flow=(-_speed(t) * 0.7, 18), wander=50)
    g = kira_glow(t)
    rp = run_phase(t)
    kstore = {}
    kpose = dict(expr="sad" if t < 9.0 else "sleepy", look=(-0.3, -0.7),
                 squash=0.2 * math.cos(TAU * 2 * rp), rot=0.1 * math.sin(TAU * 2 * rp))
    held = kira_held(kpose, g, t, kstore, scale_mul=1.25)
    lean = 0.08 + 0.05 * smoothstep(10.8, 11.6, t)
    expr = "determined"
    anc = draw_haru(ctx, sx, sy, RUN_S, view="side_right", expr=expr, run_phase=rp, arms="cup_hands",
                    held=held, lantern="back", lantern_lit=0.0, t=tg, wind=1.4, wind_dir=-1,
                    blink=core.blink(tg, 3), body_lean=lean, head_pitch=-0.15 if 9.2 < t < 9.8 else 0.05,
                    look=(0.3, 0.5) if 9.2 < t < 9.8 else (0.6, 0.0), rim="#FFC27A", light="left")
    if "pos" in kstore:
        kx, ky = kstore["pos"]
        world.draw_light_pool(ctx, kx, ky, 230, 200, "#FFD58A", 0.35 * g / 0.6)
    world.draw_parallax_forest(ctx, cam_x, "fg", phase)
    ctx.restore()
    img = core.surface_rgba(s)
    core.over(img, fly)
    if "pos" in kstore:
        kx, ky = kstore["pos"]
        c = core.rgb("#FFE3A0")
        seed = fly.copy()
        yy, xx = np.ogrid[0:H, 0:W]
        blob = np.exp(-(((xx - kx) ** 2 + (yy - ky) ** 2) / (2 * 22.0 ** 2))).astype(np.float32) * 0.9 * g
        seed[..., 0] += blob * c[0]
        seed[..., 1] += blob * c[1]
        seed[..., 2] += blob * c[2]
        core.add_glow(img, seed, 20, 1.2)
    else:
        core.add_glow(img, fly, 16, 1.2)
    # big foreground torii sweeping past
    fg = _fg_torii_layer(t, cam_x, phase)
    if fg is not None:
        fg = motion_blur_h(fg, _speed(t) * FG_PAR / FPS * 2.2, taps=8, ds=2)
        core.over(img, fg)
    # bursting out toward the hilltop: violet dawn light floods in from the upper right
    if t > 11.3:
        L = core.draw_layer(world.draw_light_pool, 1900, 120, 1500 + 600 * burst, 1000, "#B7A2E0",
                            0.25 + 0.55 * burst)
        img[..., :3] = 1 - (1 - img[..., :3]) * (1 - L[..., :3])
    # whip-in at the start of B
    if shot == "B" and t < 4.2:
        img = soft_blur_h(img, 560 * (1 - ease_out((t - 4.0) / 0.2, 2)), taps=14)
    return img


# ============================================================== shot C: Kira close-up
def _streak_bg(t, cam_x, phase, zoom=1.35, dark=0.62, blur=260):
    """Blurred, streaking stairs background, rendered at quarter resolution."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    q = 4
    s, ctx = core.new_surface(W // q, H // q)
    ctx.scale(1 / q, 1 / q)
    ctx.save()
    ctx.translate(960, 540)
    ctx.scale(zoom, zoom)
    ctx.translate(-960, -540 + 140)
    world.draw_steps_path(ctx, cam_x, t, phase)
    ctx.restore()
    va = smoothstep(9.0, 12.0, t)
    if va > 0:
        world.draw_light_pool(ctx, 1900, 60, 1200, 650, "#9C86C8", 0.4 * va)
    a = core.surface_rgba(s)
    acc = np.zeros_like(a)
    taps = 12
    for i in range(taps):
        acc += np.roll(a, int(round(blur / q * (i / (taps - 1) - 0.5))), axis=1)
    a = gaussian_filter(acc / taps, (1.5, 1.5, 0))
    a[..., :3] *= dark
    out = np.empty((H, W, 4), np.float32)
    for c in range(4):
        out[..., c] = np.asarray(Image.fromarray(a[..., c], mode="F").resize((W, H), Image.BILINEAR))
    return out


def shot_c(t, tg):
    phase = phase_at(t)
    cam_x = cam_x_at(t)
    rp = run_phase(t)
    bg = _streak_bg(t, cam_x, phase)
    s = core.rgba_surface(bg)
    ctx = cairo.Context(s)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    speed_lines(ctx, t, 26, (0, 0, W, H), 5, 0.22)
    g = kira_glow(t)
    mo, vo, talking = core.mouth("kira", tg)
    lt = t - 6.1
    # Kira: sad, drooping; sleepy toward "よわくなってきた"
    kexpr = "sad" if lt < 2.2 else "sleepy"
    bob = math.sin(TAU * 2 * rp)
    kpose = dict(expr=kexpr, mouth_open=mo, vowel=vo, blink=core.blink(tg, 9, 2.6), look=(0.35, -0.75),
                 squash=0.18 * max(0.0, -bob) - 0.08 * max(0.0, bob), rot=0.08 * math.sin(TAU * rp) - 0.05,
                 pose="droop" if lt > 1.0 else "none")
    kstore = {}
    held = kira_held(kpose, g, t, kstore, scale_mul=1.35)
    scale = 4.5
    dn = smoothstep(6.5, 6.9, t)
    pose = dict(view="side_right", expr="worried", run_phase=rp, arms="cup_hands", crop="bust", t=tg,
                wind=1.6, wind_dir=-1, rim="#FFC27A", light="left", lantern="back", body_lean=0.02,
                head_pitch=-0.6 * dn, look=(lerp(0.6, 0.1, dn), lerp(0.0, 1.0, dn)))
    # frame so that the cup sits right of centre, with handheld follow lag
    anc0 = haru_anchor(pose, 0.0, 0.0, scale)
    cx, cy = anc0["cup"]
    push = 1.0 + 0.0 * t
    tx = 1000 + 18 * noise1(t * 1.3, 6) + 30 * ease_in_out(invlerp(6.0, 9.1, t))
    ty = 770 + 16 * noise1(t * 1.1, 8) - 22 * math.sin(TAU * 2 * rp - 0.6) * 0.35
    # let the body bob partly through (camera doesn't fully stabilise)
    x = tx - cx
    y = ty - cy - 10 * bob
    draw_haru(ctx, x, y, scale, held=held, blink=core.blink(tg, 3), **pose)
    if "pos" in kstore:
        kx, ky = kstore["pos"]
        world.draw_light_pool(ctx, kx, ky + 20, 520, 420, "#FFD58A", 0.4 * g / 0.6)
    s.flush()
    img = core.surface_rgba(s)
    if "pos" in kstore:
        core.add_glow(img, glow_blob(kstore["pos"], 190, 0.8 * g), 60, 1.0)
    # sparkle trail left by Kira streaking backwards (dim)
    return img


# ============================================================== shot E: Haru face close-up
def shot_e(t, tg):
    phase = phase_at(t)
    cam_x = cam_x_at(t)
    rp = run_phase(t)
    bg = _streak_bg(t, cam_x, phase, zoom=1.2, dark=0.7, blur=220)
    s = core.rgba_surface(bg)
    ctx = cairo.Context(s)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    va = smoothstep(10.0, 12.0, t)
    world.draw_light_pool(ctx, 1850, 180, 900, 700, "#B39BD6", 0.25 + 0.2 * va)
    speed_lines(ctx, t, 22, (0, 0, W, H), 8, 0.18)
    mo, vo, talking = core.mouth("haru", tg)
    g = kira_glow(t)
    # glance down at Kira 10.05-10.5, then up and shout, then push
    down = smoothstep(10.05, 10.25, t) * (1 - smoothstep(10.4, 10.6, t))
    push = ease_in(clamp((t - 11.15) / 0.25), 2)
    bob = math.sin(TAU * 2 * rp)
    scale = 3.4 + 0.4 * push
    kstore = {}
    kpose = dict(expr="sleepy", look=(0.3, -0.8), squash=0.15 * max(0, -bob))
    held = kira_held(kpose, g, t, kstore)
    pose = dict(yaw=math.radians(62), expr="determined", mouth_open=mo, vowel=vo,
                blink=max(core.blink(tg, 3), 0.35 * down), look=(lerp(0.5, 0.1, down), lerp(-0.05, 1.0, down)),
                head_pitch=lerp(0.05, -0.65, down) + 0.08 * push, head_tilt=0.04 * bob, run_phase=rp,
                arms="cup_hands", crop="bust", t=tg, wind=1.8, wind_dir=-1, rim="#C8B4FF", light="left",
                lantern="back", body_lean=0.08 + 0.1 * push, brow_tilt=0.5 if t > 10.55 else None)
    anc0 = haru_anchor(pose, 0.0, 0.0, scale)
    fx, fy = anc0["face"]
    tx = 820 + 14 * noise1(t * 1.4, 3) + 260 * push
    ty = 470 + 10 * noise1(t * 1.2, 5) + 10 * bob
    x, y = tx - fx, ty - fy
    anc = draw_haru(ctx, x, y, scale, held=held, **pose)
    # Kira's light from below (she holds it at chest height)
    if "pos" in kstore:
        kx, ky = kstore["pos"]
        world.draw_light_pool(ctx, kx, ky, 520, 520, "#FFD58A", 0.3 * g / 0.6)
    # sweat: one drop on the temple sliding down, small drops flicked off backwards
    fx2, fy2 = anc["face"]
    k = scale / 2.7
    sw = (t - 10.0) / 1.4
    hx2, hy2 = anc["head"]
    draw_sweat(ctx, hx2 + 16 * scale, hy2 - 4 * scale + 30 * k * sw, 17 * k, alpha=1 - smoothstep(0.85, 1.0, sw))
    for i, t0 in enumerate((10.25, 10.75, 11.05)):
        u = t - t0
        if 0 < u < 0.45:
            draw_sweat(ctx, hx2 - 20 * k - 700 * u, hy2 - 30 * k - 180 * u + 500 * u * u, 9 * k,
                       alpha=1 - u / 0.45)
    s.flush()
    img = core.surface_rgba(s)
    if "pos" in kstore:
        core.add_glow(img, glow_blob(kstore["pos"], 110, 0.8 * g), 40, 1.0)
    if push > 0:
        img = soft_blur_h(img, 90 * push, taps=8)
    return img


# ============================================================== render
def fast_bloom(img, threshold=0.72, radius=36.0, strength=0.55):
    """core.bloom with the small-radius pass at half resolution (same look, ~3x faster)."""
    rgbp = img[..., :3]
    lum = rgbp.max(axis=2, keepdims=True)
    bright = rgbp * np.clip((lum - threshold) / (1 - threshold + 1e-6), 0, 1)
    g = core.blur(bright, radius) * strength + core.blur(bright, radius * 0.3, downscale=2) * strength * 0.6
    img[..., :3] = 1 - (1 - rgbp) * (1 - np.clip(g, 0, 1))
    return img


def render(t: float) -> np.ndarray:
    tg = T0 + t
    if t < 4.0:
        img = shot_a(t, tg)
    elif t < 6.0:
        img = side_run(t, tg, "B")
    elif t < 9.1:
        img = shot_c(t, tg)
    elif t < 10.0:
        img = side_run(t, tg, "D")
    elif t < 11.4:
        img = shot_e(t, tg)
    else:
        img = side_run(t, tg, "F")
    fast_bloom(img, 0.72, 36, 0.55)
    core.vignette(img, 0.32)
    core.grain(img, int(round(t * FPS)))
    return core.finish(img)
