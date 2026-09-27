"""s06 — Ascension (global 48.0–55.0, 168 frames).

The lit lantern with Kira rises from tiny Haru's hands on the summit into the
vast pre-dawn sky while the camera cranes up to follow it.  During Kira's line
(l11, local 1.4–4.46) we cut to a close-up of Kira peeking out of the top of the
glowing lantern, then back to the wide.  4.6–4.8 the lantern trembles and
gathers light; at 4.8 it bursts into golden sparks, a streak of light shoots to
the zenith and settles as one new bright star while the sky warms.
"""
from __future__ import annotations

import math

import cairo
import numpy as np

from common import core, world
from common.core import clamp, lerp, smoothstep, ease_in_out, ease_out, noise1, TAU
from common.characters import draw_haru, draw_lantern, draw_lantern_glow, draw_kira, draw_kira_glow

S0 = 48.0
T_END = 7.0
T_BURST = 4.8
T_ARRIVE = 5.32          # streak reaches the zenith and becomes the star
CU_IN, CU_OUT = 1.30, 3.92   # close-up cut window (local)

HARU_X = 1120.0
HARU_S = 0.6
LANT_S = 1.4            # lantern scale in hilltop world units (wide shot)
G = dict(loop=18.0, body_cy=26 + 46 * 0.88, body_bot=26 + 46 * 1.76)

GOLD = (1.0, 0.84, 0.42)
WARM = (1.0, 0.72, 0.36)


# ------------------------------------------------------------------ fast compositing (bbox-limited)
def _bbox(a, pad=0):
    al = a[::4, ::4, 3]
    rows = np.flatnonzero(al.max(1) > 0.002)
    if rows.size == 0:
        return None
    cols = np.flatnonzero(al.max(0) > 0.002)
    y0 = max(0, rows[0] * 4 - 4 - pad); y1 = min(core.H, rows[-1] * 4 + 8 + pad)
    x0 = max(0, cols[0] * 4 - 4 - pad); x1 = min(core.W, cols[-1] * 4 + 8 + pad)
    return slice(int(y0), int(y1)), slice(int(x0), int(x1))


def over_fast(dst, src):
    b = _bbox(src)
    if b is not None:
        core.over(dst[b], src[b])


def glow_fast(dst, src, radius, strength, tint=None):
    b = _bbox(src, int(radius * 3))
    if b is not None:
        core.add_glow(dst[b], src[b], radius=radius, strength=strength, tint=tint)


def screen_rows(img, y0, y1, stops):
    """Screen-blend a vertical gradient (list of (offset, colour, alpha)) between rows y0..y1."""
    ys = np.arange(core.H, dtype=np.float32)
    u = np.clip((ys - y0) / max(1.0, y1 - y0), 0, 1)
    offs = [st[0] for st in stops]
    cols = np.array([core.rgb(st[1]) if isinstance(st[1], str) else st[1] for st in stops], np.float32)
    als = np.array([st[2] for st in stops], np.float32)
    a = np.interp(u, offs, als).astype(np.float32)
    c = np.stack([np.interp(u, offs, cols[:, k]) for k in range(3)], 1).astype(np.float32) * a[:, None]
    img[..., :3] = 1 - (1 - img[..., :3]) * (1 - c[:, None, :])


def bloom_fast(img, threshold=0.75, radius=40.0, strength=0.6):
    """core.bloom with the small-radius pass blurred at half resolution (much faster)."""
    rgbp = img[..., :3]
    lum = rgbp.max(axis=2, keepdims=True)
    bright = rgbp * np.clip((lum - threshold) / (1 - threshold + 1e-6), 0, 1)
    g = core.blur(bright, radius) * strength + core.blur(bright, radius * 0.3, downscale=2) * strength * 0.6
    img[..., :3] = 1 - (1 - rgbp) * (1 - np.clip(g, 0, 1))
    return img


_VIG = None


def vignette_fast(img, strength=0.38, color=(0.02, 0.02, 0.08)):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:core.H, 0:core.W].astype(np.float32)
        d = np.sqrt(((xx - core.W / 2) / (core.W * 0.62)) ** 2 + ((yy - core.H / 2) / (core.H * 0.62)) ** 2)
        _VIG = (np.clip(d - 0.55, 0, 1) ** 1.5)[..., None].astype(np.float32)
    v = _VIG * strength
    img[..., :3] = img[..., :3] * (1 - v) + np.array(color, np.float32) * v
    return img


# ------------------------------------------------------------------ timing curves
def phase_at(t):
    p = 0.45 + 0.17 * smoothstep(0.0, 7.0, t)
    p += 0.08 * ease_out(clamp((t - T_BURST) / 1.6), 2)   # the burst warms the sky
    return p


def cam_at(t):
    """Hilltop camera (cx, cy, zoom)."""
    u = ease_in_out(clamp((t + 0.6) / (T_BURST + 0.6)))
    cx = lerp(1150, 1100, u)
    cy = lerp(-700, -1240, u)
    z = lerp(0.27, 0.252, u)
    # after the burst: slow drift, still rising a touch
    a = ease_out(clamp((t - T_BURST) / 2.4), 2)
    cy -= 90 * a
    cx -= 25 * a
    z -= 0.004 * a
    # gentle handheld float
    cx += 10 * noise1(t * 0.5, 3)
    cy += 8 * noise1(t * 0.45, 8)
    return cx, cy, z


_HANDS = None


def _haru_hands():
    """World position of Haru's raised hands (reach_up) — lantern start point."""
    global _HANDS
    if _HANDS is None:
        s, c = core.new_surface(8, 8)
        a = draw_haru(c, HARU_X, world.hilltop_ground_y(HARU_X), HARU_S, view="back", arms="reach_up", t=S0)
        _HANDS = a["hands"] if "hands" in a else a["hand_l"]
    return _HANDS


_RISE_T = np.linspace(0, T_BURST, 481)


def _rise_table():
    # velocity profile: lift-off ease-in, cruise, hover/tremble before the burst
    v = 640 * np.array([smoothstep(0.05, 1.7, x) for x in _RISE_T])
    v *= np.array([1 - 0.75 * smoothstep(4.35, 4.75, x) for x in _RISE_T])
    d = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) * 0.5 * np.diff(_RISE_T))])
    return d


_RISE = _rise_table()


def lantern_world(t):
    """(hang_x, hang_y, sway, tremble) of the lantern in hilltop world coords."""
    t = min(t, T_BURST)
    hx, hy = _haru_hands()
    rise = float(np.interp(t, _RISE_T, _RISE))
    y0 = hy - G["body_bot"] * LANT_S + 6
    y = y0 - rise
    k = smoothstep(0.1, 2.5, t)
    x = hx + 110 * ease_in_out(clamp(t / T_BURST)) + k * 75 * math.sin(t * 0.95 + 0.3)
    sway = k * (0.14 * math.sin(t * 1.9 + 0.8) + 0.05 * math.sin(t * 3.3))
    trem = smoothstep(4.55, 4.8, t)
    x += trem * 9 * math.sin(t * 91)
    sway += trem * 0.12 * math.sin(t * 67 + 1)
    return x, y, sway, trem


def voice_pulse(t):
    o, v, talk = core.mouth("kira", S0 + t)
    return o if talk else 0.0


def lantern_light(t):
    """Brightness of the lantern (lit), including voice pulse and pre-burst gathering."""
    base = 0.92 + 0.05 * math.sin(t * 7.3) + 0.03 * noise1(t * 6, 2)
    base += 0.35 * voice_pulse(t)
    base += 0.9 * smoothstep(4.5, 4.8, t) ** 1.5
    return base


# ------------------------------------------------------------------ particles
def _rng(seed):
    return np.random.default_rng(seed)


_EMB = None


def _embers():
    global _EMB
    if _EMB is None:
        r = _rng(61)
        n = 150
        _EMB = dict(tb=np.sort(r.uniform(0.15, T_BURST, n)), life=r.uniform(0.9, 2.2, n),
                    vx=r.normal(0, 60, n), vy=r.uniform(40, 170, n), off=r.normal(0, 22, n),
                    ph=r.uniform(0, TAU, n), fr=r.uniform(5, 13, n), sz=r.uniform(0.6, 1.4, n))
    return _EMB


def draw_embers(ctx, t, cam):
    """Glowing ember sparks shed by the rising lantern (world space)."""
    E = _embers()
    for i in range(len(E["tb"])):
        tb = E["tb"][i]
        age = t - tb
        if age < 0 or age > E["life"][i]:
            continue
        lx, ly, sw, _ = lantern_world(tb)
        bx = lx - math.sin(sw) * G["body_bot"] * LANT_S + E["off"][i]
        by = ly + G["body_bot"] * LANT_S * 0.9
        wx = bx + E["vx"][i] * age + 25 * math.sin(age * 2 + E["ph"][i])
        wy = by + E["vy"][i] * age + 30 * age * age
        sx, sy = world.hilltop_to_screen(wx, wy, cam)
        f = age / E["life"][i]
        tw = 0.6 + 0.4 * math.sin(age * E["fr"][i] + E["ph"][i])
        a = (1 - f) ** 1.5 * tw
        r = 2.4 * E["sz"][i] * (1 - 0.5 * f)
        col = core.mix("#FFF2C4", "#FF9A3D", f)
        ctx.set_source(core.radial_gradient(sx, sy, 0, r * 4, [(0, col, a * 0.5), (1, col, 0)]))
        ctx.arc(sx, sy, r * 4, 0, TAU)
        ctx.fill()
        core.set_color(ctx, core.mix("#FFFFFF", col, 0.4), a)
        ctx.arc(sx, sy, r, 0, TAU)
        ctx.fill()


_SNOW = None


def _snow():
    global _SNOW
    if _SNOW is None:
        r = _rng(77)
        n = 240
        ang = r.uniform(0, TAU, n)
        sp = 620 * r.random(n) ** 0.7 + 60
        _SNOW = dict(vx=np.cos(ang) * sp, vy=np.sin(ang) * sp * 0.8 - 60, fall=r.uniform(40, 130, n),
                     ph=r.uniform(0, TAU, n), fr=r.uniform(3, 9, n), sz=r.uniform(0.5, 1.5, n),
                     sway=r.uniform(8, 26, n), life=r.uniform(2.3, 4.0, n), c=r.integers(0, 3, n))
    return _SNOW


def draw_snow(ctx, t, center):
    """Slow golden sparks drifting down like snow after the burst (screen space)."""
    dt = t - T_BURST
    if dt <= 0:
        return
    S = _snow()
    cx, cy = center
    k = 2.2
    pal = ["#FFE9A8", "#FFD66B", "#FFF6D6"]
    for i in range(len(S["vx"])):
        if dt > S["life"][i]:
            continue
        spread = (1 - math.exp(-k * dt)) / k
        x = cx + S["vx"][i] * spread + S["sway"][i] * math.sin(dt * 1.6 + S["ph"][i])
        y = cy + S["vy"][i] * spread + S["fall"][i] * dt
        f = dt / S["life"][i]
        tw = 0.5 + 0.5 * math.sin(dt * S["fr"][i] + S["ph"][i])
        a = min(1.0, dt * 6) * (1 - f) ** 1.2 * (0.35 + 0.65 * tw)
        world.draw_glint(ctx, x, y, 13 * S["sz"][i], a, pal[S["c"][i]], rot=S["ph"][i] + dt * 0.5, diag=0.35)


# ------------------------------------------------------------------ the new star
def star_screen(t, cam):
    """New star position (screen), attached to the sky (moves with the horizon parallax)."""
    hs_ref = world.hilltop_horizon_screen(cam_at(T_BURST))
    hs = world.hilltop_horizon_screen(cam)
    bx, by, _, _ = lantern_world(T_BURST)
    px, _ = world.hilltop_to_screen(bx, by, cam_at(T_BURST))
    return px + 70, 118 + (hs - hs_ref)


def burst_center(cam):
    lx, ly, sw, _ = lantern_world(T_BURST)
    wx = lx - math.sin(sw) * G["body_cy"] * LANT_S
    wy = ly + math.cos(sw) * G["body_cy"] * LANT_S
    return world.hilltop_to_screen(wx, wy, cam)


def draw_new_star(ctx, t, cam, big=1.0):
    dt = t - T_ARRIVE
    if dt < 0:
        return
    x, y = star_screen(t, cam)
    pop = 1 + 1.4 * math.exp(-dt * 3.2) * (0.5 + 0.5 * math.cos(dt * 9))
    tw = 1 + 0.12 * math.sin(t * 5.3) + 0.08 * math.sin(t * 8.9 + 1)
    size = 95 * big * pop * tw
    rot = 0.05 * math.sin(t * 0.7)
    # soft blue-white halo + gold glint + white core
    ctx.set_source(core.radial_gradient(x, y, 0, size * 1.6,
                                        [(0, (0.85, 0.92, 1.0), 0.45), (0.3, (0.75, 0.85, 1.0), 0.16),
                                         (1, (0.7, 0.8, 1.0), 0)]))
    ctx.arc(x, y, size * 1.6, 0, TAU)
    ctx.fill()
    world.draw_glint(ctx, x, y, size, 1.0, "#FFE9A8", rot=rot, diag=0.42)
    world.draw_glint(ctx, x, y, size * 0.62, 0.9, "#DDEBFF", rot=rot + math.pi / 4 + 0.1, diag=0.3)


def draw_streak(ctx, t, cam):
    dt = t - T_BURST
    if dt < 0 or dt > 1.0:
        return
    p0 = burst_center(cam)
    p1 = star_screen(t, cam)
    dur = T_ARRIVE - T_BURST

    def head(tt):
        u = ease_out(clamp(tt / dur), 2.2)
        # slight arc: bows to the left then curves in
        bow = -60 * math.sin(u * math.pi)
        return lerp(p0[0], p1[0], u) + bow, lerp(p0[1], p1[1], u)

    hx, hy = head(dt)
    path = [head(max(0.0, dt - k * 0.04)) for k in range(14, -1, -1)]
    fade = 1.0 - smoothstep(dur, 1.0, dt)
    L = 520 * (1 - 0.8 * smoothstep(dur * 0.7, 1.0, dt))
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    world.draw_shooting_star(ctx, (hx, hy), ang, L, t=t, intensity=1.7 * fade, path=path, width=2.4,
                             color="#FFE9A8", seed=11)


def draw_shockwave(ctx, t, center):
    dt = t - T_BURST
    if dt < 0 or dt > 1.4:
        return
    cx, cy = center
    for k, (d0, sp, a0) in enumerate(((0.0, 1250, 0.6), (0.1, 700, 0.25))):
        u = dt - d0
        if u <= 0:
            continue
        r = sp * ease_out(clamp(u / 1.3), 2.6) + 10
        a = a0 * (1 - clamp(u / 1.3)) ** 1.6
        w = 12 * (1 - clamp(u / 1.3)) + 2
        ctx.set_line_width(w * 3)
        ctx.set_source_rgba(1.0, 0.8, 0.45, a * 0.25)
        ctx.arc(cx, cy, r, 0, TAU)
        ctx.stroke()
        ctx.set_line_width(w)
        ctx.set_source_rgba(1.0, 0.95, 0.8, a)
        ctx.arc(cx, cy, r, 0, TAU)
        ctx.stroke()


def draw_rays(ctx, t, center):
    """Brief radiant rays at the burst."""
    dt = t - T_BURST
    if dt < -0.12 or dt > 1.2:
        return
    a = smoothstep(-0.12, 0.0, dt) * (1 - smoothstep(0.0, 1.2, dt)) ** 1.3
    cx, cy = center
    r = _rng(5)
    n = 18
    ang = r.uniform(0, TAU, n)
    ln = r.uniform(300, 900, n)
    wd = r.uniform(0.02, 0.06, n)
    rot = dt * 0.25
    for i in range(n):
        A = ang[i] + rot
        L = ln[i] * (0.4 + 0.8 * ease_out(clamp(dt / 0.5 + 0.2)))
        ctx.move_to(cx, cy)
        ctx.line_to(cx + math.cos(A - wd[i]) * L, cy + math.sin(A - wd[i]) * L)
        ctx.line_to(cx + math.cos(A + wd[i]) * L, cy + math.sin(A + wd[i]) * L)
        ctx.close_path()
        g = cairo.RadialGradient(cx, cy, 0, cx, cy, L)
        g.add_color_stop_rgba(0, 1, 0.9, 0.62, 0.3 * a)
        g.add_color_stop_rgba(1, 1, 0.8, 0.5, 0)
        ctx.set_source(g)
        ctx.fill()


# ------------------------------------------------------------------ wide shot
def _haru_pose(t):
    """Arms: reach_up (release) → lowering → wave."""
    kw = dict(view="back", t=S0 + t, wind=1, head_pitch=0.8)
    if t < 0.9:
        kw["arms"] = "reach_up"
    else:
        # her right hand (screen-right in back view) stays high and waves; the left lowers
        u = ease_in_out((t - 0.9) / 0.5)
        w = clamp((t - 1.2) / 0.3)
        wv = math.sin((t - 1.2) * TAU * 1.5) * w
        wave = (-(128 + 30 * wv), 6, -300 + 10 * abs(wv))
        kw["hand_r"] = (lerp(-82, wave[0], u), lerp(24, wave[1], u), lerp(-268, wave[2], u))
        kw["hand_l"] = (lerp(82, 40, u), lerp(24, 16, u), lerp(-268, -24, u))
    # tiny lean forward-back following the lantern, hop at the burst
    kw["body_lean"] = 0.03 * math.sin(t * 1.3)
    b = t - T_BURST
    if 0 < b < 0.5:
        kw["bounce"] = -0.25 * math.sin(b / 0.5 * math.pi)
    return kw


def render_wide(t):
    tg = S0 + t
    cam = cam_at(t)
    cx, cy, z = cam
    ph = phase_at(t)
    s, ctx = core.new_surface()
    world.draw_hilltop(ctx, ph, tg, cam=cam, sun_x=1500)

    lx, ly, sw, trem = lantern_world(t)
    lit = lantern_light(t)
    alive = t < T_BURST
    bx = lx - math.sin(sw) * G["body_cy"] * LANT_S
    by = ly + math.cos(sw) * G["body_cy"] * LANT_S
    lsx, lsy = world.hilltop_to_screen(bx, by, cam)

    # warm light on the ground/tree while the lantern is near
    near = 1 - smoothstep(0.0, 2.6, t)
    ctx.save()
    ctx.translate(960, 540)
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)
    gy = world.hilltop_ground_y(HARU_X)
    if near > 0.01:
        world.draw_light_pool(ctx, HARU_X, gy + 10, 520, 90, "#FFB85C", 0.55 * near)
    # burst flash lights the ground too
    fl = math.exp(-max(0.0, t - T_BURST) * 2.2) if t >= T_BURST else 0.0
    if fl > 0.01:
        world.draw_light_pool(ctx, HARU_X, gy, 1500, 240, "#FFD9A0", 0.4 * fl)
    # Haru
    rimc = core.mix(world.sky_colors(ph)["rim"], core.rgb("#FFC27A"), max(near, 0.8 * fl))
    draw_haru(ctx, HARU_X, gy, HARU_S, rim=rimc, **_haru_pose(t))
    ctx.restore()

    img = core.surface_rgba(s)

    # --- lantern layer (with its own strong glow)
    if alive:
        lay_s, lc = core.new_surface()
        sc = LANT_S * z * (1 + 0.06 * trem * math.sin(t * 40))
        hx, hy = world.hilltop_to_screen(lx, ly, cam)
        draw_lantern_glow(lc, hx, hy, sc * 2.4, clamp(lit * 0.9), sw, tg)
        draw_lantern(lc, hx, hy, sc, lit=clamp(lit), sway=sw, t=tg, kira=1.0)
        # hot core that swells with the voice / pre-burst gathering
        cr = 40 * sc * (1 + 0.5 * voice_pulse(t) + 2.5 * trem ** 2)
        lc.set_source(core.radial_gradient(lsx, lsy, 0, cr, [(0, "#FFFBEA", 0.8 * min(1, lit)),
                                                              (1, "#FFD66B", 0)]))
        lc.arc(lsx, lsy, cr, 0, TAU)
        lc.fill()
        lay = core.surface_rgba(lay_s)
        over_fast(img, lay)
        gstr = 1.0 + 0.6 * voice_pulse(t) + 1.2 * trem
        glow_fast(img, lay, 26, gstr)
        glow_fast(img, lay, 90, 0.9 * gstr, tint=WARM)

    # --- the sky warms after the burst: rosy wash rising from the horizon + warm aura
    warm = 0.25 * smoothstep(0.0, 7.0, t) + 0.75 * ease_out(clamp((t - T_BURST) / 1.8), 2)
    if warm > 0.01:
        hs = world.hilltop_horizon_screen(cam)
        screen_rows(img, hs - 700, hs, [(0, "#E7A6A0", 0), (0.6, "#E7A6A0", 0.16 * warm), (1, "#F6C58F", 0.30 * warm)])
    # --- effects layer: embers, burst, streak, snow, new star
    fx_s, fc = core.new_surface()
    draw_embers(fc, t, cam)
    center = burst_center(cam)
    if t >= T_BURST - 0.12:
        dt = t - T_BURST
        draw_rays(fc, t, center)
        # sky-wide soft flash
        if dt > -0.05:
            a = math.exp(-max(0.0, dt) * 3.2) * smoothstep(-0.05, 0.03, dt)
            flash_s, flc = core.new_surface()
            flc.set_source(core.radial_gradient(center[0], center[1], 0, 1500,
                                               [(0, "#FFF4D6", 0.85 * a), (0.06, "#FFE0A0", 0.55 * a), (0.3, "#FFC37A", 0.28 * a),
                                                (1, "#FFA868", 0.16 * a)]))
            flc.paint()
            flash = core.surface_rgba(flash_s)
            img[..., :3] = 1 - (1 - img[..., :3]) * (1 - flash[..., :3])
        draw_shockwave(fc, t, center)
        if dt >= 0:
            world.draw_sparkles(fc, dt, center, n=150, speed=820, life=2.4, gravity=110, seed=4,
                                size=1.7, drag=1.9, flash=False)
            world.draw_sparkles(fc, dt, center, n=90, speed=420, life=2.2, gravity=70, seed=9,
                                size=1.1, drag=2.4, flash=False)
            draw_snow(fc, t, center)
        draw_streak(fc, t, cam)
    draw_new_star(fc, t, cam)
    fx = core.surface_rgba(fx_s)
    over_fast(img, fx)
    glow_fast(img, fx, 18, 1.0)
    glow_fast(img, fx, 70, 0.8, tint=GOLD)
    return img


# ------------------------------------------------------------------ close-up (l11)
_CU_EMB = None


def _cu_embers():
    global _CU_EMB
    if _CU_EMB is None:
        r = _rng(33)
        n = 60
        _CU_EMB = dict(x=r.uniform(-200, 2100, n), y0=r.uniform(-300, 1300, n), sp=r.uniform(120, 420, n),
                       sz=r.uniform(0.6, 2.4, n), ph=r.uniform(0, TAU, n), fr=r.uniform(3, 9, n))
    return _CU_EMB


def render_cu(t):
    tg = S0 + t
    u = (t - CU_IN) / (CU_OUT - CU_IN)
    ph = phase_at(t)
    s, ctx = core.new_surface()
    cam_y = -1250 - 260 * u              # tracking up with the lantern: stars slide down
    world.draw_sky(ctx, ph, tg, cam_y=cam_y, sun_x=1500)
    world.draw_stars(ctx, ph, tg, cam_y=cam_y)
    world.draw_clouds(ctx, ph, tg, cam_y=cam_y, alpha=0.5)

    # slow push-in around the lantern
    zoom = lerp(1.28, 1.42, ease_in_out(u))
    fx, fy = 930, 430
    o, vowel, talk = core.mouth("kira", tg)
    o = o if talk else 0.0
    sw = 0.05 * math.sin(t * 1.9 + 0.8) + 0.02 * math.sin(t * 3.3)
    hang_x = 930 + 30 * math.sin(t * 0.95 + 0.3)
    hang_y = 335 + 16 * math.sin(t * 1.4)
    LS = 3.9
    lit = clamp(0.95 + 0.05 * math.sin(t * 7.3))

    # far embers (behind)
    E = _cu_embers()

    def embers(c, front):
        for i in range(len(E["x"])):
            big = E["sz"][i] > 1.6
            if big != front:
                continue
            y = (E["y0"][i] + E["sp"][i] * (t - CU_IN)) % 1600 - 250
            x = E["x"][i] + 40 * math.sin(t * 1.3 + E["ph"][i])
            a = 0.35 + 0.35 * math.sin(t * E["fr"][i] + E["ph"][i])
            r = 3 * E["sz"][i] * (3.0 if front else 1.0)
            c.set_source(core.radial_gradient(x, y, 0, r, [(0, "#FFE2A0", a), (0.5, "#FFB45C", a * 0.4),
                                                           (1, "#FF9A3D", 0)]))
            c.arc(x, y, r, 0, TAU)
            c.fill()

    lay_s, lc = core.new_surface()
    lc.translate(fx, fy)
    lc.scale(zoom, zoom)
    lc.translate(-fx, -fy)
    embers(lc, False)
    draw_lantern_glow(lc, hang_x, hang_y, LS * 0.9, 0.6, sw, tg)
    # Kira first (sits in the top opening); lantern body covers her lower half
    kx = hang_x - math.sin(sw) * 40 + 4 * math.sin(t * 2.1)
    ky = hang_y + 30 - 10 * o + 4 * math.sin(t * 3.0)
    # line beats: "ありがとう、ハル！" wave down; "ずっと、みてるからね！" cheer
    lt = tg - core.line("l11_kira")["start"]
    pose = "wave" if lt < 1.3 else "cheer"
    look = (-0.25, 0.55) if lt < 1.3 else (-0.1, 0.25)
    rot = -0.12 + 0.1 * math.sin(t * 2.2) if lt < 1.3 else 0.06 * math.sin(t * 3.1)
    squash = -0.18 * o + 0.08 * math.sin(t * 5)
    if 1.2 < lt < 1.45:                        # little hop into the cheer
        squash += 0.35 * math.sin((lt - 1.2) / 0.25 * math.pi)
    draw_kira_glow(lc, kx, ky, 2.2, 0.55, tg)
    kl_s, kl = core.new_surface()
    kl.translate(fx, fy)
    kl.scale(zoom, zoom)
    kl.translate(-fx, -fy)
    kl.rectangle(-2000, -2000, 6000, 2000 + hang_y + (G["loop"] + 1.5) * LS)   # sits behind the top cap
    kl.clip()
    draw_kira(kl, kx, ky, 2.2, expr="happy", mouth_open=o, vowel=vowel, look=look, rot=rot, squash=squash,
              glow=0.85, t=tg, pose=pose, pose_phase=lt, blush=0.9)
    lc.save()
    lc.rectangle(-2000, hang_y + (G["loop"] - 2) * LS, 6000, 4000)   # hide the loop behind Kira
    lc.clip()
    pulse = 0.35 * o
    draw_lantern(lc, hang_x, hang_y, LS, lit=clamp(lit), sway=sw, t=tg, kira=0.55 + 0.45 * o)
    lc.restore()
    bcx = hang_x - math.sin(sw) * G["body_cy"] * LS
    bcy = hang_y + math.cos(sw) * G["body_cy"] * LS
    lc.set_source(core.radial_gradient(bcx, bcy, 0, 170, [(0, "#FFF8E0", 0.12 + 0.5 * pulse), (1, "#FFD66B", 0)]))
    lc.arc(bcx, bcy, 170, 0, TAU)
    lc.fill()
    lay = core.surface_rgba(lay_s)
    img = core.surface_rgba(s)
    over_fast(img, lay)
    glow_fast(img, lay, 24, 0.22 + 0.25 * o)
    glow_fast(img, lay, 120, 0.35 + 0.3 * o, tint=WARM)
    kly = core.surface_rgba(kl_s)
    over_fast(img, kly)
    glow_fast(img, kly, 40, 0.25 + 0.25 * o, tint=GOLD)

    fr_s, frc = core.new_surface()
    embers(frc, True)
    fr = core.surface_rgba(fr_s)
    fr = core.blur(fr, 5, downscale=2)
    over_fast(img, fr)
    glow_fast(img, fr, 20, 0.6)
    return img


# ------------------------------------------------------------------ entry
def render(t: float) -> np.ndarray:
    t = clamp(t, 0.0, T_END - 1e-3)
    if CU_IN <= t < CU_OUT:
        img = render_cu(t)
        img = bloom_fast(img, 0.88, 36, 0.3)
    else:
        img = render_wide(t)
        b = math.exp(-max(0.0, t - T_BURST) * 1.5) if t >= T_BURST - 0.2 else 0.0
        img = bloom_fast(img, lerp(0.72, 0.7, b), lerp(36, 50, b), lerp(0.55, 0.7, b))
    img = vignette_fast(img, 0.38)
    img = core.grain(img, int(round(t * core.FPS)))
    return core.finish(img)
