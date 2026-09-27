"""s02 — Haru at her attic window sees the star land (global 7.0–13.0 s).

Beats (local seconds):
  0.0–0.6  drowsy Haru, chin on hands, at the round window; slow push-in
  0.6      a shooting star dives across the view, cool flash; eyes pop (squash → stretch)
  1.0      warm glow blooms in the garden below (lower-left of the view) and pulses; she leans in
  1.6–2.7  「……いまの、なに？」 (lip sync), curious head tilt
  2.9–3.4  anticipation dip, jump, grabs the lantern from the sill
  3.4–4.4  turns 3/4 left and dashes out screen-left (smear ghosts, lantern swinging, shake)
  4.4–6.0  empty window, glow pulsing, chime swinging from the rush of air, dust motes
"""
from __future__ import annotations

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cairo  # noqa: E402
import numpy as np  # noqa: E402

from common import core, world  # noqa: E402
from common.core import (clamp, lerp, smoothstep, ease_in_out, ease_out, ease_in, noise1,  # noqa: E402
                         spring, TAU)
from common.characters import draw_haru, draw_lantern  # noqa: E402

START = core.scene_bounds("s02")[0]
CX, CY, R = world.ATTIC_WIN
K = world.ATTIC_VIEW_SCALE
SILL_Y = world.ATTIC_SILL_Y

T_STAR = 0.6          # streak across the view
T_LAND = 1.0          # glow blooms in the garden
T_DIP = 2.9           # anticipation
T_JUMP = 3.12         # take-off
T_GRAB = 3.30         # lantern leaves the sill
T_TURN = 3.45
T_RUN = 3.62
T_GONE = 4.40

# garden-glow position in the window view (town coords) -> screen
GLOW_T = (650.0, 960.0)
GLOW_S = (CX + (GLOW_T[0] - 1120) * K, CY + (GLOW_T[1] - 500) * K)

HARU_X = 1045.0
HARU_Y = 1845.0       # virtual feet point (off-screen) for the bust shot
HARU_S = 3.0

LANT_REL = 0.62       # lantern scale relative to Haru
LANT_S = HARU_S * LANT_REL
LANT_X = 1395.0
LANT_Y = SILL_Y - 16 - 110 * LANT_S   # hang point so the base rests on the sill top


def _hex(c):
    return "#%02X%02X%02X" % tuple(int(clamp(v) * 255 + 0.5) for v in c)


# ------------------------------------------------------------------ timing helpers
def flash_amt(t):
    if t < T_STAR + 0.05:
        return 0.0
    u = t - (T_STAR + 0.05)
    return clamp(u / 0.05) * math.exp(-u * 5.0)


def glow_amt(t):
    """Garden glow intensity (0..~1.2), pulsing softly."""
    if t < T_LAND:
        return 0.0
    u = t - T_LAND
    bloom = ease_out(u / 0.35) * (1 + 0.6 * math.exp(-u * 4))
    return bloom * (0.82 + 0.18 * math.sin(u * 3.2) + 0.05 * math.sin(u * 7.3 + 1))


def star_state(t):
    """Head position (town coords), path, intensity of the diving streak."""
    t0, t1 = T_STAR - 0.08, T_LAND
    if t < t0 or t > t1 + 0.25:
        return None
    p0 = (1900.0, -120.0)
    p1 = GLOW_T

    def pos(u):
        u = clamp(u)
        e = ease_in(u, 1.6)
        # slight arc: bow upward then dive
        x = lerp(p0[0], p1[0], e)
        y = lerp(p0[1], p1[1], e ** 1.25) - 260 * math.sin(math.pi * e)
        return x, y

    u = (t - t0) / (t1 - t0)
    head = pos(u)
    path = [pos(u - 0.035 * k) for k in range(12, -1, -1)]
    inten = 1.0 if u <= 1 else clamp(1 - (t - t1) / 0.25)
    return head, path, inten


# ------------------------------------------------------------------ layers
def view_fn_factory(t):
    def view_fn(ctx):
        st = star_state(t)
        if st is not None:
            head, path, inten = st
            if (t - (T_STAR - 0.08)) / (T_LAND - T_STAR + 0.08) <= 1.0:
                world.draw_shooting_star(ctx, head, 0, 900, t, intensity=1.2 * inten, path=path, width=4.0,
                                         seed=3)
        g = glow_amt(t)
        if g > 0:
            gx, gy = GLOW_T
            # soft warm bloom among the garden roofs
            for rr, a in ((700, 0.25), (360, 0.5), (160, 0.85), (60, 1.0)):
                ctx.set_source(core.radial_gradient(gx, gy, 0, rr * (0.8 + 0.2 * g),
                                                    [(0, "#FFE9A8", a * g), (0.5, "#FFC94D", a * g * 0.45),
                                                     (1, "#FF9A3D", 0)]))
                ctx.arc(gx, gy, rr * (0.8 + 0.2 * g), 0, TAU)
                ctx.fill()
            world.draw_light_pool(ctx, gx, gy + 60, 420, 120, "#FFB85C", 0.5 * g)
            # impact sparkles right after landing
            u = t - T_LAND
            if 0 < u < 1.6:
                world.draw_sparkles(ctx, u, (gx, gy), n=36, speed=380, life=1.2, gravity=260, seed=5,
                                    spread=math.pi * 0.9, direction=-math.pi / 2)
    return view_fn


def chime_swing(t):
    """Extra furin swing from Haru's rush of air."""
    a = 0.0
    u = t - (T_RUN + 0.25)
    if u > 0:
        a += 0.32 * math.exp(-u * 1.1) * math.sin(u * 5.2)
    u2 = t - (T_STAR + 0.1)
    if u2 > 0:
        a += 0.05 * math.exp(-u2 * 2) * math.sin(u2 * 7)
    return a


def draw_chime(ctx, t):
    ctx.save()
    ctx.translate(CX + R * 0.98, 92)
    ctx.scale(1.35, 1.35)
    ctx.rotate(chime_swing(t))
    world.draw_furin(ctx, 0, 0, t)
    ctx.restore()


def _sill_hands(k, press=0.0):
    """Wrist overrides: chin_on_hands (k=0) -> hands gripping the sill (k=1); press pushes down."""
    chin = (12.0, 48.0, -168.0)
    sill = (78.0, 46.0, -128.0 + 18 * press)
    l = tuple(lerp(a, b, k) for a, b in zip(chin, sill))
    return dict(hand_l=l, hand_r=(-l[0], l[1], l[2]))


def haru_pose(t):
    r = _haru_pose(t)
    kw, x, y, s = r
    # squash/stretch pivots on the (off-screen) feet: keep the chin anchored instead
    y -= 1090 * (kw.get('bounce') or 0.0) * s / 3.0
    return kw, x, y, s


def _haru_pose(t):
    """Returns (kwargs for draw_haru, x, y, scale, visible_ghosts) or None when gone."""
    tg = START + t
    fl = flash_amt(t)
    g = glow_amt(t)
    m_open, vowel, talking = core.mouth("haru", tg)

    # rim light: moonlight -> flash white -> warm garden glow
    rim = core.mix("#9DB4FF", "#FFFFFF", clamp(fl * 1.6))
    rim = core.mix(rim, "#FFC27A", clamp(g * 0.55) * (1 - clamp(fl * 1.6)))

    kw = dict(view="front", crop="bust", arms="chin_on_hands", expr="sleepy", t=t, rim=_hex(rim),
              light="left", mouth_open=m_open, vowel=vowel)
    x, y, s = HARU_X, HARU_Y, HARU_S

    if t < T_STAR + 0.08:
        # drowsy: slow heavy blinks, head bob
        bob = 0.5 - 0.5 * math.cos(TAU * t / 1.2)
        kw.update(expr="sleepy", eye_open=0.45 - 0.2 * bob,
                  blink=max(core.blink(tg, seed=2, period=2.2), smoothstep(0.2, 0.45, t) * (1 - smoothstep(0.45, 0.62, t)) * 0.9),
                  head_tilt=0.08 + 0.03 * math.sin(t * 2.0), head_pitch=-0.15 - 0.12 * bob,
                  look=(0.1, -0.4))
        y += 10 * bob
        return kw, x, y, s

    if t < T_DIP:
        # eyes pop: squash then stretch up, then settle; lean towards the glow
        u = t - (T_STAR + 0.08)
        pop = spring(u, freq=2.2, damp=5.0)
        sq = 0.0
        if u < 0.09:
            sq = 0.12 * math.sin(u / 0.09 * math.pi)          # quick squash
        else:
            sq = -0.14 * math.exp(-(u - 0.09) * 6) * math.cos((u - 0.09) * 14)   # stretch & settle
        lean_in = ease_in_out((t - T_LAND - 0.1) / 0.6)
        look_x = lerp(0.2, -0.85, ease_out((t - T_LAND) / 0.35))
        look_y = lerp(-0.1, 0.55, ease_out((t - T_LAND) / 0.35))
        expr = "surprised"
        tilt = -0.02 * pop
        if t > 1.55:
            # curious puzzlement during the line
            k = ease_in_out((t - 1.55) / 0.5)
            tilt = lerp(tilt, -0.16, k) + 0.03 * math.sin(t * 2.4) * k
            if t > 2.15:
                expr = "worried"
        kw.update(expr=expr, bounce=sq, head_tilt=tilt, look=(look_x, look_y),
                  eye_open=None if t > 2.15 else 1.0 + 0.12 * math.exp(-u * 3),
                  blink=core.blink(tg, seed=5, period=2.6) if u > 0.6 else 0.0,
                  head_pitch=-0.25 * lean_in, head_turn=-0.18 * lean_in, body_lean=-0.05 * lean_in)
        x += -40 * lean_in
        y += 5 * lean_in - 40 * ease_out((t - (T_STAR + 0.12)) / 0.3) + (-22 * math.exp(-u * 5) * math.sin(u * 16) if u > 0 else 0)
        s *= 1 + 0.03 * lean_in
        if talking:
            kw["blink"] = 0.0
        # hands drop from the chin to grip the sill edge (frees the mouth)
        k = ease_out((t - (T_STAR + 0.12)) / 0.3)
        kw.update(_sill_hands(k))
        return kw, x, y, s

    # ---- action: dip, jump, grab, turn, run
    kw.update(expr="determined", eye_open=None, mouth_open=0.0)
    if t < T_JUMP:
        u = (t - T_DIP) / (T_JUMP - T_DIP)
        d = ease_out(u, 2)
        kw.update(bounce=0.12 * d, head_pitch=-0.2 * (1 - d) - 0.05 * d, head_turn=-0.18 * (1 - d),
                  look=(0.5 * d - 0.6 * (1 - d), 0.3), body_lean=-0.05 * (1 - d))
        kw.update(_sill_hands(1.0, press=d))
        x += -40 + 20 * d
        y += -35 + 45 * d
        s *= 1.03
        return kw, x, y, s

    if t < T_TURN:
        # take-off: stretch up to standing, snatch the lantern with the screen-right hand
        u = (t - T_JUMP) / (T_TURN - T_JUMP)
        up = ease_out(u, 3)
        kw.update(bounce=-0.14 * math.sin(math.pi * min(1.0, u * 1.2)), head_pitch=0.05, look=(0.8, 0.3),
                  crop=None, expr="determined", mouth_open=0.3, vowel="u", arms="hold_lantern",
                  hand="screen_right", lantern=t >= T_GRAB, lantern_scale=LANT_REL, wind=1.0,
                  head_turn=0.25 * up)
        # the screen-right hand reaches out to the sill lantern, then yanks it up
        g = ease_out((t - T_GRAB) / (T_TURN - T_GRAB), 2)
        reach = ease_out((t - T_JUMP) / (T_GRAB - T_JUMP), 2)
        kw["hand_l"] = (lerp(lerp(40, 72, reach), 70, g), lerp(lerp(26, 50, reach), 60, g),
                        lerp(lerp(-52, -138, reach), -210, g))
        x += lerp(-20, 90, up)
        y += lerp(10, -235, up)
        return kw, x, y, s

    # turn & run to screen-left
    u = t - T_TURN
    turn = ease_in_out(u / 0.18)
    yaw = lerp(-0.75, -math.pi / 2, turn)
    run_u = max(0.0, t - T_RUN)
    dist = 1900 * run_u ** 1.6 + 700 * run_u
    rp = (run_u * 3.2 + 0.1) % 1.0
    kw.update(yaw=yaw, view=None, crop=None, arms="hold_lantern", lantern=True, hand="near",
              hand_r=(40, 60 + 20 * math.sin(TAU * run_u * 3.2), -170 + 15 * math.cos(TAU * run_u * 3.2)),
              lantern_scale=LANT_REL, run_phase=rp if t >= T_RUN - 0.06 else None,
              wind=2.0, wind_dir=1, look=(-0.3, 0), head_pitch=0.0,
              bounce=0.1 * math.exp(-u * 10) * math.sin(u * 30) if u < 0.3 else 0.0)
    x += 90 - 60 * turn - dist
    y += -235 + 10 * math.sin(TAU * rp * 2)
    return kw, x, y, s



# ------------------------------------------------------------------ camera
def camera(t):
    """(zoom, cx, cy, dx, dy) — slow push-in around the window, shake on the jump."""
    z = 1.05 + 0.08 * ease_in_out(t / 3.0)
    z += 0.025 * math.exp(-max(0, t - T_STAR - 0.1) * 3) * (t > T_STAR + 0.1)   # punch on the flash
    hold = ease_in_out((t - 4.0) / 2.0)
    z += 0.08 * hold                                                             # drift toward the glow
    fx = lerp(975, 935, ease_in_out((t - 3.3) / 1.2)) - 40 * hold
    fy = 520 + 35 * hold
    dx = 3 * noise1(t * 0.7, 3)
    dy = 3 * noise1(t * 0.6, 9)
    sh = 0.0
    if t > T_JUMP - 0.02:
        u = t - (T_JUMP - 0.02)
        sh = 16 * math.exp(-u * 7)
    if t > T_STAR + 0.05:
        u = t - (T_STAR + 0.05)
        sh += 5 * math.exp(-u * 8)
    dx += sh * noise1(t * 30, 1)
    dy += sh * noise1(t * 30, 2)
    return z, fx, fy, dx, dy


def apply_cam(ctx, t):
    z, fx, fy, dx, dy = camera(t)
    ctx.translate(960 + dx, 540 + dy)
    ctx.scale(z, z)
    ctx.translate(-fx, -fy)


# ------------------------------------------------------------------ render
def _draw_scene(ctx, t):
    tg = START + t
    apply_cam(ctx, t)
    fl = flash_amt(t)
    world.draw_attic_window(ctx, tg, 0.0, view_fn=view_fn_factory(t), flash=0.6 * fl, lamp=1.0, chime=False)
    draw_chime(ctx, tg)

    # warm light spilling in from the garden glow onto the inner frame / room
    g = glow_amt(t)
    if g > 0:
        world.draw_light_pool(ctx, GLOW_S[0] - 40, GLOW_S[1] + 80, 360, 240, "#FFB060", 0.18 * g)

    pose = haru_pose(t)
    lantern_on_sill = t < T_GRAB
    if pose is not None:
        kw, x, y, s = pose
        # motion-smear ghosts while dashing
        if t > T_RUN:
            for k, a in ((2, 0.12), (1, 0.22)):
                dt = 0.03 * k
                pk = haru_pose(t - dt)
                if pk is None:
                    continue
                kw2, x2, y2, s2 = pk
                kw2 = dict(kw2)
                kw2["alpha"] = a
                kw2["rim"] = None
                draw_haru(ctx, x2, y2, s2, **{k_: v for k_, v in kw2.items() if v is not None or k_ == "rim"})
        draw_haru(ctx, x, y, s, **{k_: v for k_, v in kw.items() if v is not None or k_ == "rim"})

    world.draw_attic_sill(ctx, tg, 0.0)

    if lantern_on_sill:
        # the unlit lantern sits on the sill top at her side; nudged by the dip
        sway = 0.0
        if t > T_DIP:
            sway = 0.05 * math.sin((t - T_DIP) * 20)
        draw_lantern(ctx, LANT_X, LANT_Y, LANT_S, lit=0.0, sway=sway, t=tg)
        # contact shadow
    # speed lines during the dash
    if T_RUN < t < T_GONE + 0.1:
        rng = np.random.default_rng(int(t * 24))
        a = 0.35 * (1 - smoothstep(T_GONE - 0.2, T_GONE + 0.1, t))
        for i in range(10):
            yy = rng.uniform(250, 950)
            x0 = rng.uniform(300, 1500)
            ln = rng.uniform(150, 420)
            ctx.move_to(x0, yy)
            ctx.line_to(x0 + ln, yy)
            ctx.set_line_width(rng.uniform(1.5, 4))
            ctx.set_source_rgba(1, 0.9, 0.75, a * rng.uniform(0.3, 1))
            ctx.stroke()

    # extra motes stirred by her rush
    stir = smoothstep(T_RUN, T_GONE, t)
    if stir > 0:
        u = t - T_RUN
        world.draw_dust_motes(ctx, tg, (200, 250, 1500, 1000), 40, seed=31, alpha=0.5 * stir,
                              light=(500, 800, 1100), drift=(-60 * math.exp(-u * 0.8) - 5, -8))


def render(t: float) -> np.ndarray:
    s, ctx = core.new_surface()
    _draw_scene(ctx, t)
    img = core.surface_rgba(s)
    core.bloom(img, 0.72, 36, 0.55)
    fl = flash_amt(t)
    if fl > 0.01:
        img[..., :3] = 1 - (1 - img[..., :3]) * (1 - np.array([0.55, 0.62, 0.8], np.float32) * 0.2 * fl)
    core.vignette(img, 0.38)
    core.grain(img, int(round(t * core.FPS)))
    return core.finish(img)
