"""s03 — Meeting Kira in the garden (global 13.0–25.0 s).

Shot A (local 0.0–5.8): wide-ish garden, slow push-in.  Kira lies crumpled in a
glowing dent, sits up rubbing its head (l02); Haru runs in from the house,
stops, leans in, jumps back in surprise (l03), then crouches down.
Shot B (local 5.8–12.0): eye-level close two-shot, shallow depth of field.
Kira floats up to Haru's eye level and speaks l04 (sad); on 「きえちゃうんだ」
its glow stutters and dims; Haru goes happy -> worried -> determined (nod).
"""
from __future__ import annotations

import math

import cairo
import numpy as np

from common import core, world
from common.characters import draw_haru, draw_kira, draw_kira_glow, draw_sparkle, RIM_DEFAULT
from common.core import (clamp, lerp, smoothstep, ease_in_out, ease_out, ease_out_back, spring,
                         noise1, mix)

T0 = 13.0
CUT = 5.8
PHASE = 0.0
KIRA_WARM = "#FFE9A8"
HARU_WARM = "#FFC27A"
KX, KY = world.GARDEN_IMPACT          # (1000, 820)
KGROUND = KY + 26                    # grass level Kira rests on
KS_A = 0.82                          # Kira scale in the wide shot


def _hex(c):
    return "#%02X%02X%02X" % tuple(int(clamp(v) * 255 + 0.5) for v in c)


# ============================================================ shared timing curves
_FLICKS = ((10.36, 0.06, 0.35), (10.50, 0.05, 0.55), (10.60, 0.08, 0.30), (10.84, 0.05, 0.5),
           (10.96, 0.07, 0.35), (11.12, 0.05, 0.7))


def kira_glow(t):
    """Kira's life/glow level over the scene (local t)."""
    if t < 0.8:
        g = 0.55 + 0.12 * math.sin(t * 5.0)                 # dim pulsing while knocked out
    elif t < 1.8:
        g = lerp(0.6, 1.0, ease_out((t - 0.8) / 1.0))        # wakes up, brightens
    else:
        g = 1.0 + 0.02 * math.sin(t * 3.1)
    if t > 10.3:
        g = lerp(1.0, 0.45, smoothstep(10.3, 11.1, t)) + 0.03 * noise1(t * 7, 3)
        for tc, d, depth in _FLICKS:
            if tc <= t < tc + d:
                g *= depth
    return clamp(g, 0.05, 1.0)


# ============================================================ sparkles
def _settle_sparkles(ctx, t, cx, cy, n=26, seed=11):
    """Glitter from the impact drifting down and fading (0 .. ~3 s)."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        x0 = cx + rng.normal(0, 90)
        y0 = cy - rng.uniform(20, 240)
        life = rng.uniform(1.4, 3.2)
        vy = rng.uniform(25, 70)
        ph = rng.uniform(0, math.tau)
        age = t + 0.6
        if age > life:
            continue
        f = age / life
        x = x0 + 14 * math.sin(age * 1.8 + ph)
        y = y0 + vy * age
        tw = 0.5 + 0.5 * math.sin(age * 11 + ph)
        a = (1 - f) ** 1.2 * (0.35 + 0.65 * tw)
        draw_sparkle(ctx, x, y, 9 + 9 * rng.random(), a, ph, "#FFF6D6" if i % 3 else "#FFD66B")


def _ambient_sparkles(ctx, t, cx, cy, scale, g, n=8, seed=5, rise=38.0, spread=70.0):
    """A few glints continuously lifting off Kira."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        P = rng.uniform(1.2, 2.2)
        off = rng.uniform(0, P)
        age = (t + off) % P
        k = math.floor((t + off) / P)
        r2 = np.random.default_rng(seed * 1000 + i * 37 + k)
        ang = r2.uniform(0, math.tau)
        rad = r2.uniform(0.4, 1.0) * spread * scale
        x = cx + math.cos(ang) * rad + 8 * math.sin(age * 3 + i)
        y = cy + math.sin(ang) * rad * 0.8 - rise * scale * age
        f = age / P
        a = math.sin(f * math.pi) * g * 0.9
        draw_sparkle(ctx, x, y, (7 + 6 * r2.random()) * scale, a, ang, "#FFF6D6" if i % 2 else "#FFD66B")


def _trail(ctx, t, pos_fn, scale, g, t_from, dt=1 / 48, n=30, seed=21):
    """Sparkle trail: glints emitted along Kira's path, drifting and fading."""
    e_now = math.floor(t / dt)
    for k in range(n):
        e = e_now - k
        te = e * dt
        if te < t_from:
            break
        age = t - te
        x0, y0 = pos_fn(te)
        x1, y1 = pos_fn(te - dt)
        sp = math.hypot(x0 - x1, y0 - y1) / dt
        amt = clamp(sp / 500.0) * 0.85 + 0.15
        r = np.random.default_rng(seed * 7919 + e)
        if r.random() > amt:
            continue
        jx, jy = r.normal(0, 26 * scale), r.normal(0, 18 * scale)
        life = 0.9 + 0.5 * r.random()
        f = age / life
        if f >= 1:
            continue
        x = x0 + jx + r.normal(0, 10) * age
        y = y0 + jy + 30 * scale * age
        tw = 0.6 + 0.4 * math.sin(age * 14 + e)
        draw_sparkle(ctx, x, y, (6 + 9 * r.random()) * scale * (1 - 0.5 * f), (1 - f) ** 1.4 * tw * g,
                     r.uniform(0, 3), "#FFF6D6" if e % 3 else "#FFD66B")


# ============================================================ post helpers
_VIG = None


def _bloom(img, threshold, radius, strength):
    """core.bloom with the small-radius pass blurred at reduced resolution (same look, ~4x cheaper)."""
    rgbp = img[..., :3]
    lum = rgbp.max(axis=2, keepdims=True)
    bright = rgbp * np.clip((lum - threshold) / (1 - threshold + 1e-6), 0, 1)
    g = core.blur(bright, radius) * strength
    g += core.blur(bright, radius * 0.3, downscale=3) * (strength * 0.6)
    img[..., :3] = 1 - (1 - rgbp) * (1 - np.clip(g, 0, 1))
    return img


def _vignette(img, strength=0.42, color=(0.02, 0.02, 0.08)):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:core.H, 0:core.W].astype(np.float32)
        d = np.sqrt(((xx - core.W / 2) / (core.W * 0.62)) ** 2 + ((yy - core.H / 2) / (core.H * 0.62)) ** 2)
        _VIG = (np.clip(d - 0.55, 0, 1)[..., None] ** 1.5).astype(np.float32)
    v = _VIG * strength
    img[..., :3] = img[..., :3] * (1 - v) + np.array(color, np.float32) * v
    return img


def _post(img, t, bloom=(0.8, 38, 0.5)):
    _bloom(img, *bloom)
    _vignette(img, 0.42)
    core.grain(img, int(round(t * core.FPS)))
    return core.finish(img)


def _cam(ctx, cx, cy, z, ox=0.0, oy=0.0):
    ctx.translate(960 + ox, 540 + oy)
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)


def _handheld(t, amp=5.0):
    return amp * noise1(t * 0.6, 41), amp * 0.7 * noise1(t * 0.55, 57)


# ============================================================ shot A — wide
def _haru_a(t, tg):
    """Pose dict + ground point for Haru in the wide shot."""
    p = dict(view="three_quarter_right", arms="hold_lantern", lantern=True, lantern_lit=0.0, t=tg,
             expr="neutral", look=(0.7, 0.25), wind=0.3)
    mo, vw, talk = core.mouth("haru", tg)
    X_STOP = 700.0
    FEET = 928.0
    x, y = X_STOP, FEET
    # run in from the house (screen-left), decelerating
    RUN_END = 1.05
    if t < RUN_END:
        u = t / RUN_END
        # position ease-out; speed -> 0 at RUN_END
        x = lerp(-260, X_STOP, 1 - (1 - u) ** 2)
        dist = (X_STOP + 260) * (1 - (1 - u) ** 2)
        p.update(view="side_right", run_phase=(dist / 360.0) % 1.0, expr="surprised", look=(0.6, 0.1),
                 wind=1.0, wind_dir=-1, body_lean=0.05)
    else:
        st = t - RUN_END
        # stop: forward follow-through of the lean, settling
        lean = 0.16 * math.exp(-4 * st) * math.cos(st * 9) + 0.05
        bounce = 0.35 * math.exp(-7 * st) * math.cos(st * 14)
        p.update(body_lean=lean, bounce=bounce, expr="surprised" if st < 0.5 else "neutral")
        # cautious lean in while Kira talks (1.6–3.0), with a double-blink at 2.05
        li = smoothstep(1.7, 2.8, t)
        p["body_lean"] += 0.13 * li
        p["head_tilt"] = 0.10 * li
        p["look"] = (0.8, 0.35)
        bl = core.blink(tg, seed=3)
        for tb in (2.02, 2.28):
            if tb <= t < tb + 0.14:
                bl = max(bl, math.sin((t - tb) / 0.14 * math.pi))
        p["blink"] = bl
        if t >= 2.4:
            p["expr"] = "gentle"
    # ---- 3.0–3.5 jump back in surprise
    if t >= 2.95:
        a = t - 2.95
        if a < 0.14:       # anticipation: dip down
            k = math.sin(a / 0.14 * math.pi / 2)
            p.update(crouch=0.25 * k, bounce=0.35 * k, expr="surprised")
        elif a < 0.46:     # airborne backwards
            u = (a - 0.14) / 0.32
            y -= 42 * math.sin(u * math.pi)
            x -= 80 * ease_in_out(u)
            p.update(bounce=-0.15 * math.sin(u * math.pi), body_lean=-0.14, expr="surprised", crouch=0.0)
        else:              # land, squash, recover
            b = a - 0.46
            x -= 80
            p.update(bounce=0.4 * math.exp(-6 * b) * math.cos(b * 12), body_lean=-0.12 * math.exp(-3 * b),
                     expr="surprised")
        # free hand flies to her mouth
        if 0.08 < a < 1.3:
            p["hand_r"] = (-11, 58, -186)
        p["look"] = (0.7, 0.3)
        p["blink"] = 0.0 if a < 1.0 else core.blink(tg, seed=3)
        if a > 1.45:
            p["expr"] = "happy"
        if a > 1.45 and a < 1.6:
            p["blink"] = math.sin((a - 1.45) / 0.15 * math.pi)
    # ---- 5.0–5.8 crouch down to Kira's level
    if t >= 4.95:
        c = ease_in_out((t - 4.95) / 0.7)
        p["crouch"] = 0.8 * c
        p["body_lean"] = p.get("body_lean", 0) + 0.12 * c
        p["look"] = (0.8, 0.5 - 0.4 * c)
        p["expr"] = "happy"
    if talk:
        p.update(mouth_open=mo, vowel=vw)
    return x, y, p


def _kira_a(t, tg):
    """Position + pose for Kira in the wide shot."""
    g = kira_glow(t)
    mo, vw, talk = core.mouth("kira", tg)
    x, y = KX, KGROUND - 22
    pose = dict(expr="hurt", glow=g, t=tg, look=(-0.6, -0.1))
    rot, sq = -1.25, 0.28
    if t < 0.8:
        pose["blink"] = 1.0 if t < 0.55 else 0.0
        sq = 0.28 + 0.05 * math.sin(t * 5)
        y = KGROUND - 10
    elif t < 0.95:          # anticipation squash
        k = (t - 0.8) / 0.15
        sq = lerp(0.28, 0.5, k)
        y = KGROUND - 10 + 4 * k
    elif t < 1.25:          # pop up with stretch, rotate upright
        k = (t - 0.95) / 0.3
        rot = lerp(-1.25, 0.0, ease_out_back(k, 1.2))
        sq = lerp(0.5, -0.35, math.sin(k * math.pi / 2))
        y = lerp(KGROUND - 6, KGROUND - 34, math.sin(k * math.pi))
        y = min(y, KGROUND - 6)
    else:                    # settle with spring squash
        k = t - 1.25
        rot = 0.08 * math.exp(-4 * k) * math.sin(k * 11)
        sq = -0.35 * math.exp(-5 * k) * math.cos(k * 15) + 0.05 * math.sin(tg * 2.4)
        y = KGROUND - 24
    if 0.95 <= t < 3.0:
        pose["pose"] = "rub_head"
        rot += 0.10 * math.sin(tg * 5.0) * smoothstep(1.2, 1.5, t)
    # jolted by Haru's shout
    if t >= 3.0:
        a = t - 3.0
        pose["expr"] = "surprised"
        hop = max(0.0, math.sin(clamp(a / 0.35) * math.pi))
        y -= 26 * hop
        sq = -0.3 * hop + 0.25 * math.exp(-8 * max(0, a - 0.35)) * (a > 0.35)
        if a > 1.8:
            pose["expr"] = "neutral"
            sq = 0.05 * math.sin(tg * 2.4)
        pose["look"] = (-0.8, -0.35)
        pose["blink"] = core.blink(tg, seed=8, period=2.9) if a > 0.6 else 0.0
    # starts lifting off before the cut
    if t >= 5.1:
        u = ease_in_out((t - 5.1) / 0.7)
        y -= 70 * u
        sq = -0.2 * u
        pose["expr"] = "sad"
        pose["look"] = (-0.8, -0.6)
    if talk:
        pose.update(mouth_open=mo, vowel=vw)
    return x, y, rot, sq, pose


def _crater(ctx, g, t):
    """Small glowing dent in the grass where Kira landed."""
    ctx.save()
    ctx.translate(KX, KGROUND + 4)
    ctx.scale(1, 0.26)
    rg = core.radial_gradient(0, 0, 0, 120, [(0, "#0D1A22", 0.75), (0.7, "#10202A", 0.45), (1, "#10202A", 0)])
    ctx.set_source(rg)
    ctx.arc(0, 0, 120, 0, math.tau)
    ctx.fill()
    ctx.restore()
    world.draw_light_pool(ctx, KX, KGROUND + 2, 190, 48, KIRA_WARM, 0.75 * g)
    world.draw_light_pool(ctx, KX, KGROUND - 6, 70, 22, "#FFF6D6", 0.6 * g)
    # flattened grass ring: short lit blades leaning outwards
    rng = np.random.default_rng(4)
    for i in range(46):
        a = rng.uniform(0, math.tau)
        r = rng.uniform(95, 135)
        bx, by = KX + math.cos(a) * r, KGROUND + 4 + math.sin(a) * r * 0.26
        L = rng.uniform(14, 30)
        lean = math.cos(a) * 1.1 + 0.08 * math.sin(t * 1.3 + i)
        tx, ty = bx + math.sin(lean) * L, by - math.cos(lean) * L * 0.8
        f = clamp(1 - r / 150) * g
        c = mix("#2C5046", KIRA_WARM, 0.2 + 0.6 * f)
        ctx.set_source_rgba(*c, 0.9)
        ctx.set_line_width(3)
        ctx.move_to(bx, by)
        ctx.line_to(tx, ty)
        ctx.stroke()


def shot_a(t, tg):
    g = kira_glow(t)
    z = lerp(1.38, 1.62, ease_in_out(t / CUT))
    cx = lerp(830, 860, ease_in_out(t / CUT))
    cy = lerp(715, 770, ease_in_out(t / CUT))
    hx_, hy_ = _handheld(tg, 4)
    # camera tilts up a touch to follow Haru's jump back
    cy -= 55 * smoothstep(2.85, 3.2, t) * (1 - smoothstep(3.7, 4.6, t))
    # little camera jolt when Haru jumps back
    if 3.1 < t < 3.8:
        hy_ += 6 * math.sin((t - 3.1) * 30) * math.exp(-6 * (t - 3.1))
    kx, ky, krot, ksq, kpose = _kira_a(t, tg)
    light = (kx, ky, 330, KIRA_WARM, 0.95 * g)

    s, ctx = core.new_surface()
    ctx.save()
    _cam(ctx, cx, cy, z, hx_, hy_)
    world.draw_garden(ctx, tg, PHASE, light=light)
    _crater(ctx, g, tg)
    draw_kira_glow(ctx, kx, ky, KS_A, g * 0.7, tg)
    draw_kira(ctx, kx, ky, KS_A, rot=krot, squash=ksq, **kpose)
    # Haru
    hx, hy, hp = _haru_a(t, tg)
    # warm rim when she's close to Kira's light
    near = clamp(1 - abs(hx - kx) / 700)
    rim = _hex(mix(RIM_DEFAULT, HARU_WARM, 0.35 + 0.5 * near * g))
    draw_haru(ctx, hx, hy, 0.86, rim=rim, light="right", **hp)
    ctx.restore()
    # glow / particle layer (in camera space)
    fx, fctx = core.new_surface()
    _cam(fctx, cx, cy, z, hx_, hy_)
    world.draw_fireflies(fctx, tg, 22, (380, 420, 1920, 1000), 3)
    _settle_sparkles(fctx, t, KX, KGROUND - 20)
    _ambient_sparkles(fctx, tg, kx, ky, KS_A, g, n=7)
    if t > 5.1:
        _trail(fctx, t, lambda tt: _kira_a(tt, tt + T0)[:2], KS_A, g, 5.1)
    draw_kira(fctx, kx, ky, KS_A, rot=krot, squash=ksq, **kpose)   # bloom source
    glow = core.surface_rgba(fx)
    # foreground grass on top (keep a gap in front of Kira)
    s2, gctx = core.new_surface()
    _cam(gctx, cx, cy, z, hx_, hy_)
    world.draw_grass(gctx, tg, 1080, "fg", x_range=(-60, 840), phase=PHASE, light=light, height=0.8)
    world.draw_grass(gctx, tg, 1080, "fg", x_range=(1160, 2000), phase=PHASE, light=light, seed=9, height=0.8)
    world.draw_grass(gctx, tg, 1060, "fg", x_range=(840, 1160), phase=PHASE, light=light, seed=6,
                     height=0.55, density=1.3)
    img = core.surface_rgba(s)
    kira_only = glow.copy()
    core.over(img, glow)   # particles (+ Kira again, harmless)
    core.over(img, core.surface_rgba(s2))
    core.add_glow(img, kira_only, 22, 0.8)
    return img


# ============================================================ shot B — close two-shot
BG_Z = 2.0
KS_B = 2.1
BG_C = (1020.0, 640.0)


def _kira_b_pos(t):
    """Kira's centre in the close shot (screen space, before camera push)."""
    u = ease_out(clamp((t - CUT) / 0.95), 3)
    x = lerp(1250, 1270, u)
    y = lerp(1250, 515, u)
    bob = math.sin((t - CUT) * 2.3) * 12 * smoothstep(CUT + 0.6, CUT + 1.4, t)
    drift = 10 * math.sin((t - CUT) * 0.9)
    # a tiny sag as it dims
    sag = 30 * smoothstep(10.4, 11.3, t)
    return x + drift, y + bob + sag


def _kira_b(t, tg):
    g = kira_glow(t)
    mo, vw, talk = core.mouth("kira", tg)
    x, y = _kira_b_pos(t)
    vy = (_kira_b_pos(t)[1] - _kira_b_pos(t - 1 / 48)[1]) * 48
    sq = clamp(-vy / 2200, -0.35, 0.0) + 0.06 * math.sin((t - CUT) * 2.3 + 1.2)
    rot = 0.06 * math.sin((t - CUT) * 1.4)
    pose = dict(expr="sad", glow=g, t=tg, look=(-0.85, 0.05), blink=core.blink(tg, seed=8, period=3.1))
    if t < CUT + 0.4:
        pose["look"] = (-0.3, 0.6)
    if 6.6 < t < 7.4:        # 「キラ」 — little introductory bow
        k = math.sin((t - 6.6) / 0.8 * math.pi)
        rot += 0.12 * k
        sq += 0.1 * k
    if t > 10.3:
        pose["pose"] = "droop"
        pose["look"] = (-0.5, 0.45)
        rot += 0.03 * noise1(tg * 9, 2)
    if talk:
        pose.update(mouth_open=mo, vowel=vw)
    return x, y, rot, sq, pose


def _haru_b(t, tg, g):
    mo, vw, talk = core.mouth("haru", tg)
    p = dict(view="three_quarter_right", arms="hold_lantern", lantern=True, lantern_lit=0.0, t=tg,
             crop="bust", expr="happy", wind=0.4, light="right")
    ky = _kira_b_pos(t)[1]
    # eyes follow Kira rising
    p["look"] = (0.8, clamp((500 - ky) / 900, -0.6, 0.35))
    p["head_pitch"] = clamp((480 - ky) / 1400, -0.35, 0.1)
    bl = core.blink(tg, seed=3, period=3.8)
    # listening: slow tilt
    p["head_tilt"] = 0.05 * math.sin((t - CUT) * 0.7) + 0.05
    if t >= 8.35:
        p["expr"] = "worried"
        p["head_tilt"] = lerp(p["head_tilt"], 0.1, smoothstep(8.3, 9.0, t))
    for tb in (8.3,):
        if tb <= t < tb + 0.16:
            bl = max(bl, math.sin((t - tb) / 0.16 * math.pi))
    if t >= 10.35:          # 「きえちゃう」 — shock
        k = smoothstep(10.35, 10.6, t)
        p["eye_open"] = 1.0 + 0.18 * k
        p["head_pitch"] = p["head_pitch"] + 0.06 * k
        bl = 0.0
    if t >= 11.25:
        a = t - 11.25
        if a < 0.14:
            bl = math.sin(a / 0.14 * math.pi)
        if a >= 0.07:
            p["expr"] = "determined"
            p.pop("eye_open", None)
        # small nod: down then up with follow-through
        nod = math.sin(clamp((a - 0.1) / 0.34) * math.pi) * (1 if a < 0.44 else 0)
        nod -= 0.25 * math.exp(-6 * max(0, a - 0.44)) * math.sin(max(0, a - 0.44) * 12)
        p["head_pitch"] = -0.2 * nod
        p["look"] = (0.9, 0.0)
    p["blink"] = bl
    if talk:
        p.update(mouth_open=mo, vowel=vw)
    rim = _hex(mix(RIM_DEFAULT, HARU_WARM, 0.35 + 0.65 * clamp((g - 0.3) / 0.7)))
    p["rim"] = rim
    return p


_BG_CACHE = {}


def _half_blur(draw, sigma):
    """Draw at half resolution, gaussian blur there, upscale -> cheap shallow-DOF layer."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    s, ctx = core.new_surface(core.W // 2, core.H // 2)
    ctx.scale(0.5, 0.5)
    draw(ctx)
    a = core.surface_rgba(s)
    out = np.empty((core.H, core.W, 4), np.float32)
    for ch in range(4):
        b = gaussian_filter(a[..., ch], sigma, mode="nearest")
        out[..., ch] = np.asarray(Image.fromarray(b, mode="F").resize((core.W, core.H), Image.BILINEAR))
    return out


def _bg_b(tg):
    """Blurred (shallow DOF) garden behind the close shot."""
    def draw(ctx):
        _cam(ctx, BG_C[0], BG_C[1], BG_Z)
        world.draw_garden(ctx, tg, PHASE)
        world.draw_fireflies(ctx, tg, 22, (380, 420, 1920, 1000), 3, size=0.9)
    return _half_blur(draw, 3.5)


def shot_b(t, tg):
    g = kira_glow(t)
    push = lerp(1.0, 1.06, ease_in_out((t - CUT) / (11.2 - CUT))) + 0.035 * smoothstep(11.2, 12.0, t)
    hx_, hy_ = _handheld(tg, 5)
    cam = (960.0, 540.0, push)
    kx, ky, krot, ksq, kpose = _kira_b(t, tg)

    img = _bg_b(tg)
    # warm fill from Kira over the blurred background
    s, ctx = core.new_surface()
    _cam(ctx, *cam, hx_, hy_)
    world.draw_light_pool(ctx, kx, ky + 60, 700, 520, KIRA_WARM, 0.22 * g, op=cairo.OPERATOR_OVER)
    # Haru, big, screen-left, facing Kira
    hp = _haru_b(t, tg, g)
    HX, HFEET, HS = 600.0, 1790.0, 2.9
    anc = draw_haru(ctx, HX, HFEET, HS, **hp)
    # warm light falling on her face (flickers with Kira)
    fx, fy = anc["face"]
    ctx.save()
    ctx.push_group()
    world.draw_light_pool(ctx, fx + 200, fy + 10, 360, 320, HARU_WARM, 0.42 * g, op=cairo.OPERATOR_OVER)
    ctx.pop_group_to_source()
    ctx.set_operator(cairo.OPERATOR_ATOP)
    ctx.paint()
    ctx.restore()
    ctx.set_operator(cairo.OPERATOR_OVER)
    # Kira
    draw_kira_glow(ctx, kx, ky, KS_B, g * 0.6, tg)
    draw_kira(ctx, kx, ky, KS_B, rot=krot, squash=ksq, **kpose)

    lay = core.surface_rgba(s)
    core.over(img, lay)
    # particles + bloom source
    ps, pctx = core.new_surface()
    _cam(pctx, *cam, hx_, hy_)
    _trail(pctx, t, _kira_b_pos, 1.4, g, CUT)
    _ambient_sparkles(pctx, tg, kx, ky, 1.5, g, n=9, spread=60)
    # dimming: a few glints fall away from Kira on the flicker
    if t > 10.3:
        rng = np.random.default_rng(99)
        for i in range(10):
            t0 = 10.35 + rng.uniform(0, 0.8)
            a = t - t0
            if 0 < a < 1.2:
                draw_sparkle(pctx, kx + rng.normal(0, 70), ky + 20 + 90 * a + 40 * a * a,
                             10 + 8 * rng.random(), (1 - a / 1.2) * 0.8, rng.uniform(0, 3), "#FFD66B")
    parts = core.surface_rgba(ps)
    core.over(img, parts)
    ks, kctx = core.new_surface()
    _cam(kctx, *cam, hx_, hy_)
    draw_kira(kctx, kx, ky, KS_B, rot=krot, squash=ksq, **kpose)
    kl = core.surface_rgba(ks)
    kl += parts
    core.add_glow(img, kl, 36, 0.55 * g + 0.1)
    # soft out-of-focus foreground grass tips at the bottom
    fg = _half_blur(lambda c: world.draw_grass(c, tg, 1180, "fg", x_range=(-60, 1980), phase=PHASE, height=1.0,
                                               density=0.55, light=(kx, ky, 900, KIRA_WARM, 0.5 * g), seed=13), 4.5)
    core.over(img, fg)
    return img


# ============================================================ entry
def render(t: float) -> np.ndarray:
    tg = T0 + t
    if t < CUT:
        img = shot_a(t, tg)
    else:
        img = shot_b(t, tg)
    return _post(img, t)
