"""s05 — Summit: the paper-lantern idea (global 37.0–48.0 s, local 0–11 s).

Shots
  A  0.0–4.0  medium, slow push-in: Haru runs in, skids, pants; Kira tries to fly and
              falls back; 「そうだ……！」; she swings the lantern round to her front,
              Kira drifts inside; 3.2 WHOOMP — the lantern ignites.
  B  4.0–9.0  close (bust) shot, shallow depth of field: Haru's face lit warm by the
              lantern; Kira speaks from inside (light pulses with speech); lantern tugs up.
  C  9.0–11.0 low medium: Haru raises the lantern, 「いって、キラ！」, releases at 10.2,
              it rises out of frame; wind gust, arms stay up.
"""
from __future__ import annotations

import math

import cairo
import numpy as np

from common import core
from common.core import (clamp, lerp, smoothstep, ease_in_out, ease_out, ease_in, noise1, invlerp,
                         mix, TAU)
from common import world
from common.characters import draw_haru, draw_kira, draw_lantern, draw_sparkle
from common.characters import _lantern_geom

T0, T1 = core.scene_bounds("s05")
WARM = "#FFC27A"
COOL_RIM = "#A9BEFF"

HX = 900.0                       # Haru world x on the summit
HY = world.hilltop_ground_y(HX)  # her feet
HS = 0.85                        # Haru world scale
LS = 0.95                        # lantern scale (relative to Haru)
T_IGN = 3.2
T_REL = 10.2
CUT_B = 4.0
CUT_C = 9.0

_G = _lantern_geom()


# ------------------------------------------------------------------ helpers
def _hexmix(a, b, k):
    return mix(a, b, clamp(k))


def _cam_ctx(ctx, cam):
    cx, cy, z = cam
    ctx.translate(960, 540)
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)


def _haru_anchors(cam, **kw):
    """Dry run of draw_haru on a tiny surface to learn anchors (same transform)."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
    c = cairo.Context(s)
    _cam_ctx(c, cam)
    return draw_haru(c, HX, HY, HS, **kw)


def _ignite(t):
    """(lit, flash) of the lantern at local t."""
    if t < T_IGN - 0.25:
        return 0.0, 0.0
    if t < T_IGN:
        # anticipation: a flicker inside the paper
        k = invlerp(T_IGN - 0.25, T_IGN, t)
        return 0.12 * k * (0.6 + 0.4 * math.sin(t * 70)), 0.0
    d = t - T_IGN
    lit = clamp(ease_out(d / 0.18) )
    flash = math.exp(-d * 4.5) * clamp(d / 0.05)
    return lit, flash


def _lantern_kira(t):
    if t < 2.45:
        return 0.0
    if t < T_IGN:
        return 0.35 * smoothstep(2.45, 3.0, t)
    return lerp(0.35, 1.0, ease_out((t - T_IGN) / 0.3))


def _hand_targets(t):
    """Hand overrides (hand_l, hand_r) in Haru body space, or None for preset arms."""
    cup = (9, 34, -98)
    back = (-6, -30, -112)
    cradle = (22, 60, -78)
    if t < 1.9:
        return None
    if t < 2.08:
        k = ease_in_out(invlerp(1.9, 2.08, t))
        p = tuple(lerp(a, b, k) for a, b in zip(cup, back))
    elif t < 2.4:
        k = ease_in_out(invlerp(2.08, 2.4, t))
        p = tuple(lerp(a, b, k) for a, b in zip(back, cradle))
    elif t < 8.0:
        p = cradle
        # gentle breathing lift once lit
        p = (p[0], p[1], p[2] - 3 * math.sin(t * 2.1))
    elif t < CUT_C:
        k = ease_in_out(invlerp(8.0, 9.0, t))
        p = (22, lerp(60, 95, k), lerp(-78, -112, k) - 6 * math.sin(t * 5) * k)
    else:
        # anticipation dip then raise
        dip = 12 * math.sin(math.pi * invlerp(9.0, 9.25, t))
        k = ease_in_out(invlerp(9.15, 9.75, t))
        v = lerp(-112, -290, k) + dip
        lat = lerp(22, 30, k)
        fwd = lerp(95, 130, k)
        if t > T_REL:
            d = t - T_REL
            k2 = ease_out(d / 0.45)
            v = lerp(v, -262, k2) - 14 * math.sin(math.pi * clamp(d / 0.5))   # follow-through reach
            fwd = lerp(fwd, 160, k2)
            lat += 12 * k2                                                     # hands open apart
        p = (lat, fwd, v)
    return (p[0], p[1], p[2]), (-p[0], p[1], p[2])


def _haru_pose(t, tg):
    """Everything draw_haru needs except lantern-held stuff."""
    mo, vw, talk = core.mouth("haru", tg)
    kw = dict(t=tg, blink=core.blink(tg, seed=3), mouth_open=mo, vowel=vw, view="three_quarter_right")
    # ---------------- run-in and skid
    if t < 0.62:
        kw.update(view="side_right", run_phase=(_run_x(t) / 250.0) % 1.0, arms="cup_hands",
                  expr="worried", body_lean=0.12, head_pitch=0.1)
    elif t < 1.0:
        k = invlerp(0.62, 1.0, t)
        kw.update(view="side_right" if t < 0.8 else "three_quarter_right", arms="cup_hands",
                  expr="worried", body_lean=lerp(-0.16, 0.0, ease_out(k)),
                  crouch=0.35 * math.sin(math.pi * k), head_pitch=-0.1 * math.sin(math.pi * k))
    else:
        kw.update(arms="cup_hands", expr="worried")
    # panting after the stop
    pant = clamp(1 - (t - 0.9) / 2.4) if t > 0.8 else 0.0
    kw["bounce"] = kw.get("bounce", 0.0) + 0.05 * pant * math.sin((t - 0.8) * TAU * 2.3)
    if 0.95 < t < 2.0 and not talk:
        kw["mouth_open"] = max(kw["mouth_open"], 0.35 * pant * (0.5 + 0.5 * math.sin((t - 0.8) * TAU * 2.3)))
        kw["vowel"] = "a"
    # ---------------- Kira tries to fly (1.0–1.6); Haru watches it
    if 1.0 <= t < 1.9:
        kw["look"] = (0.35, 0.7)
        kw["head_pitch"] = -0.25
        kw["expr"] = "sad" if t < 1.25 else "worried"
    # look over her shoulder at the lantern on her back → idea
    if 1.25 <= t < 1.55:
        k = ease_in_out(invlerp(1.25, 1.42, t))
        kw["head_turn"] = -0.9 * k
        kw["look"] = (-0.9 * k + 0.35 * (1 - k), 0.2)
        kw["expr"] = "surprised" if t > 1.4 else "worried"
    if 1.55 <= t < 1.9:
        k = ease_out(invlerp(1.55, 1.75, t))
        kw["head_turn"] = -0.9 * (1 - k)
        kw["expr"] = "surprised" if t < 1.62 else "determined"
        kw["look"] = (0.2, 0.0)
        kw["head_pitch"] = 0.05
    if 1.9 <= t < CUT_B:
        kw["expr"] = "determined" if t < 2.4 else "gentle"
        kw["look"] = (0.45, 0.45)
        kw["head_pitch"] = -0.2
        if t >= T_IGN:
            k = t - T_IGN
            kw["expr"] = "surprised" if k < 0.45 else "happy"
            kw["head_pitch"] = -0.2 + 0.12 * math.exp(-k * 3) * (k > 0)
            kw["body_lean"] = -0.05 * math.exp(-k * 4)
    # ---------------- close shot
    if CUT_B <= t < CUT_C:
        kw["expr"] = "gentle" if t < 6.2 else "happy"
        kw["look"] = (0.55, 0.35 + 0.1 * math.sin(t * 0.7))
        kw["head_pitch"] = -0.12
        kw["head_tilt"] = 0.06 * math.sin(t * 0.9) + 0.04
        kw["blush"] = 0.8
        if t > 8.0:
            k = ease_in_out(invlerp(8.0, 9.0, t))
            kw["expr"] = "surprised" if t < 8.5 else "determined"
            kw["look"] = (0.5, 0.35 - 0.8 * k)
            kw["head_pitch"] = lerp(-0.12, 0.35, k)
    # ---------------- raise & release
    if t >= CUT_C:
        k = ease_in_out(invlerp(9.1, 9.7, t))
        kw["expr"] = "determined" if t < 10.6 else "happy"
        kw["head_pitch"] = lerp(0.3, 0.35, k)
        kw["look"] = (0.6, lerp(-0.6, -0.1, k))
        kw["view"] = "side_right" if t > 9.2 else "three_quarter_right"
        if t > T_REL:
            d = t - T_REL
            kw["wind"] = 2.0 * math.exp(-max(0, d - 0.3) * 0.8) * clamp(d / 0.15)
            kw["wind_dir"] = 1
            kw["head_pitch"] = 0.35 + 0.6 * ease_in_out(d / 0.7)
            kw["look"] = (0.3, -0.2 - 0.8 * ease_out(d / 0.5))
    # arms
    ho = _hand_targets(t)
    if ho is not None:
        kw["arms"] = "rest"
        kw["hand_l"], kw["hand_r"] = ho
    return kw


def _run_x(t):
    """Haru's world x during the run-in (decelerating to HX)."""
    k = clamp(t / 0.95)
    return lerp(260.0, HX, 1 - (1 - k) ** 2.2)


def _kira_state(t, tg, cup_xy, s):
    """Kira while in/above the cupped hands. Returns dict(x, y, squash, glow, expr, alpha, scale)."""
    x, y = cup_xy
    lift, squash, expr = 0.0, 0.0, "sad"
    glow = 0.28 + 0.06 * noise1(tg * 7, 4)
    if 1.0 <= t < 1.62:
        k = invlerp(1.0, 1.62, t)
        up = math.sin(math.pi * min(1.0, k / 0.8)) if k < 0.8 else 0.0
        lift = 70 * up * (1 - 0.3 * k)
        # sputter: glow flickers while trying
        glow = 0.3 + 0.18 * max(0, math.sin(t * 38)) * (k < 0.7)
        squash = -0.25 * up if k < 0.45 else 0.0
        if k >= 0.8:
            ki = invlerp(0.8, 1.0, k)
            squash = 0.5 * math.sin(math.pi * ki) * (1 - ki * 0.3)
        expr = "surprised" if k < 0.45 else "sad"
        if t > 1.5:
            expr = "hurt"
    elif 1.62 <= t < 1.9:
        expr = "sad"
        squash = 0.12 * math.exp(-(t - 1.62) * 10)
    return dict(x=x, y=y - lift * s, squash=squash, glow=glow, expr=expr)


# ------------------------------------------------------------------ cameras
def _cam(t):
    j = 1.0
    if t < CUT_B:
        # track the runner, then slow push-in
        k = ease_out(clamp(t / 1.1), 2)
        cx = lerp(760, 1010, k) + 20 * ease_in_out(invlerp(1.1, 4.0, t))
        cy = lerp(575, 565, k) - 10 * ease_in_out(invlerp(1.1, 4.0, t))
        z = lerp(1.3, 1.42, k) + 0.14 * ease_in_out(invlerp(1.1, 4.0, t))
        j = 1.0
    elif t < CUT_C:
        k = ease_in_out(invlerp(CUT_B, CUT_C, t))
        cx = lerp(975, 962, k)
        cy = lerp(512, 500, k) - 25 * ease_in_out(invlerp(8.0, 9.0, t))
        z = lerp(3.0, 3.3, k)
        j = 0.6
    else:
        k = ease_in_out(invlerp(CUT_C, 11.0, t))
        cx = lerp(1000, 990, k)
        up = ease_in_out(invlerp(10.3, 11.0, t))
        cy = lerp(530, 515, k) - 110 * up
        z = lerp(1.5, 1.44, k)
        j = 1.0
    # handheld drift
    cx += j * 6 * noise1(t * 0.6, 11) / z
    cy += j * 5 * noise1(t * 0.6, 23) / z
    # impact shake at ignition
    if T_IGN <= t < T_IGN + 0.4:
        d = t - T_IGN
        a = 7 * math.exp(-d * 10) / z
        cx += a * math.sin(d * 90)
        cy += a * math.cos(d * 75)
    return (cx, cy, z)


# ------------------------------------------------------------------ lantern placement
def _lantern_pose(t, anchors):
    """World-space hang point for the lantern + sway, or None when on her back."""
    if t < 2.08:
        return None
    hl, hr = anchors["hand_l"], anchors["hand_r"]
    mx, my = (hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2
    s = HS * LS
    # body rests on the palms (in front of hands a bit)
    hang = (mx + 4, my - (_G["body_bot"] - 8) * s)
    if t >= CUT_C:
        # arms up: hands hold the lantern's sides/back, body centred ahead of her face
        kr = ease_in_out(invlerp(9.15, 9.75, t))
        hang = (mx + lerp(4, 34, kr) * HS, my - lerp(_G["body_bot"] - 8, _G["body_cy"] + 4, kr) * s)
    sway = 0.04 * math.sin(t * 1.8)
    if t < 2.4:
        # arriving from her back: lag + swing
        k = ease_out(invlerp(2.08, 2.4, t))
        sway = 0.5 * (1 - k) * math.cos(k * 6)
    if 8.0 <= t < CUT_C:
        sway += 0.05 * math.sin(t * 5.3) * invlerp(8.0, 9.0, t)
    if t >= CUT_C:
        sway = 0.03 * math.sin(t * 2.1)
    if t >= T_REL:
        d = t - T_REL
        rise = 120 * d + 230 * d * d
        hang = (hang[0] + 18 * math.sin(d * 2.2) + 20 * d, hang[1] - rise)
        sway = 0.10 * math.sin(d * 3.1 + 0.5) * clamp(d / 0.3)
    return hang, sway


# frozen hand midpoint at release (lantern keeps rising from there, hands keep moving)
_REL_CACHE = {}


def _release_hang():
    if "h" not in _REL_CACHE:
        cam = _cam(T_REL)
        kw = _haru_pose(T_REL, T0 + T_REL)
        a = _haru_anchors(cam, **kw)
        _REL_CACHE["h"] = a
    return _REL_CACHE["h"]


# ------------------------------------------------------------------ render
_GRAIN_I = [0]


def render(t: float) -> np.ndarray:
    tg = T0 + t
    phase = lerp(0.25, 0.45, clamp(t / 11.0))
    cam = _cam(t)
    close = CUT_B <= t < CUT_C
    lit, flash = _ignite(t)
    lk = _lantern_kira(t)
    # Kira speaking inside → light pulses
    km, kv, ktalk = core.mouth("kira", tg)
    pulse = 1.0 + (0.22 * km if ktalk else 0.0) + 0.04 * noise1(tg * 6, 8) * lit

    # ---------------- background
    surf, ctx = core.new_surface()
    world.draw_hilltop(ctx, phase, tg, cam=cam, wind=1.0 + (1.0 if t > T_REL else 0.0))
    if close:
        bg = core.surface_rgba(surf)
        bg = np.concatenate([core.blur(bg[..., :3], 5.0, downscale=4), bg[..., 3:]], axis=2)
        surf = core.rgba_surface(bg)
        ctx = cairo.Context(surf)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    # ---------------- Haru pose + anchors
    hx = _run_x(t) if t < 0.95 else HX
    kw = _haru_pose(t, tg)
    lantern_back = t < 2.08
    rim = COOL_RIM if lit <= 0 else tuple(_hexmix(COOL_RIM, WARM, lit))
    ctx.save()
    _cam_ctx(ctx, cam)

    # light pool on the ground (under characters) once lit
    L = None
    if not lantern_back:
        # need hand anchors → dry run
        a0 = _haru_anchors(cam, **kw) if abs(hx - HX) < 1e-6 else None
        if t >= T_REL:
            a0 = _release_hang()
        L = _lantern_pose(t, a0)
    gl = lit * pulse + 0.8 * flash
    if L is not None and gl > 0.01:
        (lx, ly), _ = L
        world.draw_light_pool(ctx, lx, HY + 6, 330 * (1 + 0.5 * flash), 60 * (1 + 0.3 * flash), "#FFB85C",
                              0.55 * min(1.2, gl) * clamp(1 - max(0.0, (HY - ly) - 450) / 500))
        world.draw_light_pool(ctx, lx, ly + 60, 360 * (1 + flash), None, "#FF9A3D", 0.22 * min(1.3, gl))

    # skid dust
    if 0.55 < t < 1.6:
        d = t - 0.55
        rng = np.random.default_rng(5)
        for i in range(10):
            vx = rng.uniform(-120, 60)
            vy = rng.uniform(-60, -15)
            px = hx + 40 + vx * d
            py = HY - 4 + vy * d + 40 * d * d
            r = (6 + 10 * rng.random()) * (1 + 1.5 * d)
            ctx.arc(px, py, r, 0, TAU)
            ctx.set_source_rgba(0.62, 0.62, 0.78, 0.28 * clamp(1 - d / 1.0))
            ctx.fill()

    # Kira in the cupped hands (held callback)
    cupstate = {}

    def held(c, x, y, s):
        ks = _kira_state(t, tg, (x, y), s)
        cupstate["xy"] = (ks["x"], ks["y"])
        from common.characters import draw_kira_glow
        draw_kira_glow(c, ks["x"], ks["y"], 0.4 * s, ks["glow"], tg)
        draw_kira(c, ks["x"], ks["y"], 0.4 * s, expr=ks["expr"], squash=ks["squash"], glow=ks["glow"],
                  t=tg, blink=core.blink(tg, seed=9), look=(-0.2, -0.6))

    # shadow
    ctx.save()
    ctx.translate(hx, HY + 2)
    ctx.scale(1, 0.18)
    ctx.arc(0, 0, 95 * HS, 0, TAU)
    ctx.restore()
    ctx.set_source_rgba(0.03, 0.03, 0.1, 0.35)
    ctx.fill()

    haru_kw = dict(kw)
    haru_kw.update(rim=rim, light="right" if lit > 0.3 else "left")
    if kw.get("arms") == "cup_hands":
        haru_kw["held"] = held
    if lantern_back:
        haru_kw.update(lantern="back", lantern_lit=0.0, lantern_kira=0.0)
    if close:
        haru_kw["crop"] = "bust"
    # the lantern: behind her hands once raised (she grips its sides), otherwise in front
    def draw_lan():
        (lhx, lhy), sway = L
        s = HS * LS
        res = draw_lantern(ctx, lhx, lhy, s, lit=min(1.0, lit * pulse + 0.4 * flash), sway=sway, t=tg, kira=lk)
        # Kira faintly visible through the paper (lip sync)
        if lk > 0.3:
            bx, by = res["body_center"]
            ka = 0.5 * clamp((lk - 0.3) / 0.4)
            draw_kira(ctx, bx, by + 2 * s, 0.42 * s, expr="happy" if not ktalk else "neutral",
                      mouth_open=km, vowel=kv, blink=core.blink(tg, seed=9), glow=1.0, t=tg,
                      alpha=ka, rot=sway)
        return res

    lan = None
    behind = (CUT_C <= t < T_REL + 0.25) or (2.08 <= t < 2.26)
    if L is not None and behind:
        lan = draw_lan()
    anchors = draw_haru(ctx, hx, HY, HS, **haru_kw)
    if L is not None and not behind:
        lan = draw_lan()

    # Kira hovering after Haru lets go of her (1.9 .. ~3.0) and drifting into the lantern
    if 1.9 <= t < 3.0 and L is not None or 1.9 <= t < 2.08:
        if "k_hover" not in _REL_CACHE:
            k_kw = _haru_pose(1.899, T0 + 1.899)
            a = _haru_anchors(_cam(1.899), held=lambda c, x, y, s: None, **k_kw)
            _REL_CACHE["k_hover"] = (a["cup"][0], a["cup"][1] - 4)
        kx, ky = _REL_CACHE["k_hover"]
        kx += 4 * math.sin(tg * 3)
        ky += -55 * HS * ease_out(invlerp(1.9, 2.35, t)) + 3 * math.sin(tg * 4)
        kx += 85 * HS * ease_out(invlerp(1.9, 2.35, t))
        sc = 0.4 * HS
        alpha = 1.0
        sq = 0.0
        if L is not None and t >= 2.45:
            k = ease_in_out(invlerp(2.45, 2.95, t))
            (lhx, lhy), _ = L
            tx, ty = lhx, lhy + (_G["body_top"] + 6) * HS * LS
            # arc over the lantern top and dip inside
            kx = lerp(kx, tx, k)
            ky = lerp(ky, ty, k) - 40 * math.sin(math.pi * k) * HS
            sc *= lerp(1.0, 0.45, k)
            alpha = 1 - ease_in(invlerp(0.6, 1.0, k), 2)
            sq = -0.3 * math.sin(math.pi * k)
        if alpha > 0.01:
            from common.characters import draw_kira_glow
            g = 0.3 + 0.05 * noise1(tg * 7, 4)
            draw_kira_glow(ctx, kx, ky, sc, g, tg)
            draw_kira(ctx, kx, ky, sc, expr="sad" if t < 2.6 else "surprised", glow=g, t=tg,
                      squash=sq, alpha=alpha, look=(0.3, 0.2), blink=core.blink(tg, seed=9))


    # sparks after ignition, floating up
    if L is not None and t >= T_IGN and lan is not None:
        bx, by = lan["body_center"]
        if t < T_IGN + 2.0:
            world.draw_sparkles(ctx, t - T_IGN, (bx, by), n=60, speed=380 * HS, life=1.6,
                                gravity=-40, seed=21, spread=TAU, size=0.8)
        # slow embers rising all the time
        rng = np.random.default_rng(33)
        for i in range(14):
            per = rng.uniform(1.6, 2.8)
            off = rng.uniform(0, per)
            u = ((t - T_IGN + off) % per) / per
            ex = bx + rng.uniform(-40, 40) * HS + 18 * math.sin(u * 5 + i)
            ey = by - u * rng.uniform(160, 260) * HS
            a = math.sin(math.pi * u) * lit
            draw_sparkle(ctx, ex, ey, rng.uniform(4, 8) * HS, a, u * 3, "#FFD9A0")

    ctx.restore()  # camera

    # ---------------- foreground grass (not in the close-up: it would sit at her chest)
    if not close:
        world.draw_hilltop_fg(ctx, phase, tg, cam=cam, wind=1.0 + (1.0 if t > T_REL else 0.0))

    img = core.surface_rgba(surf)

    # ---------------- warm wash + glows
    if lan is not None and gl > 0.01:
        (bx, by) = lan["body_center"]
        sx = (bx - cam[0]) * cam[2] + 960
        sy = (by - cam[1]) * cam[2] + 540
        R = 520 * cam[2] * HS * (1 + 0.6 * flash)
        gs, gctx = core.new_surface()
        world.draw_light_pool(gctx, sx, sy, R, None, "#FFB060", 0.38 * min(1.2, gl), op=cairo.OPERATOR_OVER)
        glay = core.surface_rgba(gs)
        img[..., :3] = 1 - (1 - img[..., :3]) * (1 - glay[..., :3])
        # lantern bloom
        ls, lctx = core.new_surface()
        _cam_ctx(lctx, cam)
        (lhx, lhy), sway = L
        draw_lantern(lctx, lhx, lhy, HS * LS, lit=1.0, sway=sway, t=tg, kira=lk)
        lay = core.surface_rgba(ls)
        core.add_glow(img, lay, radius=30 * cam[2], strength=0.9 * min(1.3, gl), tint="#FFB060")
        core.add_glow(img, lay, radius=90 * cam[2], strength=0.5 * min(1.3, gl) + 0.3 * flash, tint="#FF9A3D")
        if flash > 0.02:
            img[..., :3] = 1 - (1 - img[..., :3]) * (1 - 0.18 * flash * np.array(core.rgb("#FFD9A0"), np.float32))

    # cool pre-dawn grade on the far edges, warm centre is handled by glow
    core.bloom(img, 0.72, 36, 0.5)
    core.vignette(img, 0.42 if close else 0.36)
    core.grain(img, int(round(t * core.FPS)), 0.018)
    return core.finish(img)
