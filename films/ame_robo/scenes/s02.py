"""s02 (global 6.0–15.5): Mina stranded under the dripping awning; Robo rolls in and scans her.

Shots (local t):
  A 0.00–1.55  medium shot of the shopfront in heavy rain, slow push; kitten mews (0.4), Mina glances down.
  B 1.55–4.90  close-up push-in (dof): 「あーあ……かさ、わすれちゃった。」, looks up, drip on hat (3.8),
               flinch + pout; squeak (4.6) -> head snaps screen-right.
  C 4.90–6.40  wide: Robo rolls in from the side street, brakes at x 1180 (5.3) with a rock, turns its
               screen to her (5.5), 「ピピッ」, scan beam starts (6.0).
  D 6.40–9.50  two-shot push-in: scan beam sweeps Mina, 「ぬれている、こどもを、はっけん。」, face flips to
               surprised/happy on はっけん; Mina leans back hugging herself, kitten hides behind her leg.
"""
import math

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from common import core
from common.core import clamp, lerp, smoothstep, ease_in_out, ease_out, ease_out_back, noise1, TAU
from lib import world
from lib.characters import draw_mina, draw_robo, draw_kitten, draw_droplet

T0 = 6.0
W, H = 1920, 1080
WEATHER = 0.0
GY = world.GROUND_Y            # 880
MX = 772.0                     # Mina's feet x (world)
KX0 = 652.0                    # kitten start x
S = 0.93                       # character scale at GY
RX_STOP = 1180.0
ROLL_T0, ROLL_T1 = 4.6, 5.3    # robo rolling in (local)
RX_START = 2060.0
DRIP_X, DRIP_Y = 785.0, 190.0  # awning scallop tip above Mina
HIT_T = 3.8
HAT_Y_EST = GY - 448.0
G_DRIP = 2600.0
FALL_T = math.sqrt(2 * (HAT_Y_EST - DRIP_Y) / G_DRIP)

CUT_B, CUT_C, CUT_D = 1.55, 4.90, 6.40
SCAN_T0, SCAN_T1 = 6.0, 8.55
HAKKEN = 8.58


# ------------------------------------------------------------------ helpers
def _damp(t, freq, damp):
    """Damped oscillation starting at +1 when t = 0 (0 before)."""
    if t < 0:
        return 0.0
    return math.exp(-damp * t) * math.cos(TAU * freq * t)


def _pulse(t, a, b, fade=0.06):
    return smoothstep(a - fade, a, t) * (1 - smoothstep(b, b + fade, t))


def camera(t):
    """-> (cam, dof)"""
    hx = noise1(t * 0.55, 3) * 4 + noise1(t * 1.3, 5) * 1.5
    hy = noise1(t * 0.5, 9) * 3 + noise1(t * 1.2, 11) * 1.2
    if t < CUT_B:
        u = ease_in_out(t / CUT_B)
        cx, cy, z = lerp(890, 850, u), lerp(560, 575, u), lerp(1.12, 1.19, u)
        return (cx + hx, cy + hy, z), 0.0
    if t < CUT_C:
        u = ease_in_out((t - CUT_B) / (CUT_C - CUT_B))
        cx, cy, z = lerp(790, 800, u), lerp(585, 560, u), lerp(2.05, 2.3, u)
        # impact shake when the drip hits
        k = math.exp(-10 * max(0.0, t - HIT_T)) * (t >= HIT_T)
        sx = math.sin(t * 95) * 5 * k
        sy = math.cos(t * 81) * 7 * k
        # tiny whip toward screen-right at the squeak
        sx += 10 * smoothstep(4.62, 4.85, t)
        return (cx + hx * 0.7 + sx, cy + hy * 0.7 + sy, z), 11.0
    if t < CUT_D:
        u = ease_out((t - CUT_C) / (CUT_D - CUT_C), 2)
        cx, cy, z = lerp(1030, 990, u), lerp(568, 575, u), lerp(1.06, 1.09, u)
        k = math.exp(-12 * max(0.0, t - ROLL_T1)) * (t >= ROLL_T1)
        return (cx + hx + math.sin(t * 70) * 4 * k, cy + hy + math.cos(t * 63) * 4 * k, z), 0.0
    u = ease_in_out((t - CUT_D) / (9.5 - CUT_D))
    cx, cy, z = lerp(978, 972, u), lerp(622, 608, u), lerp(1.36, 1.5, u)
    k = math.exp(-12 * max(0.0, t - HAKKEN)) * (t >= HAKKEN)
    return (cx + hx * 0.8 + math.sin(t * 80) * 3 * k, cy + hy * 0.8, z), 0.0


_BASE = [(CUT_B, (870.0, 568.0, 1.08), 0.0), (CUT_C, (797.0, 575.0, 1.95), 11.0),
         (CUT_D, (1010.0, 571.0, 1.02), 0.0), (99.0, (975.0, 615.0, 1.32), 0.0)]


def base_cam(t):
    for tend, c, d in _BASE:
        if t < tend:
            return c, d
    return _BASE[-1][1], _BASE[-1][2]


# ------------------------------------------------------------------ performance
def mina_pose(t, tg):
    o, v, talking = core.mouth("mina", tg)
    bl = core.blink(tg, seed=21)
    p = dict(yaw=0.5, expr="sad", mouth_open=o, vowel=v, blink=bl, look=(0.35, -0.1),
             head_tilt=0.0, head_turn=0.0, head_pitch=0.0, body_lean=0.0, bounce=0.0, crouch=0.0,
             wet=0.3, wind=0.35, wind_dir=-1, t=tg, arms="rest", light="left")
    if t < CUT_B:
        # glum look out at the rain; kitten mews at 0.4 -> she glances down at it (0.55–1.3)
        g = smoothstep(0.5, 0.68, t) * (1 - smoothstep(1.2, 1.45, t))
        p["look"] = (lerp(0.35, -0.75, g), lerp(-0.1, 0.8, g))
        p["head_pitch"] = lerp(0.05, -0.35, g)
        p["head_turn"] = lerp(0.0, -0.32, g)
        p["head_tilt"] = lerp(0.03, -0.06, g)
        p["eye_open"] = lerp(0.75, 0.9, g)
        p["smile"] = 0.15 * g
        p["blink"] = max(bl, _pulse(t, 0.49, 0.55, 0.05))
        return p
    if t < CUT_C:
        # sigh: shoulders drop, head sags
        sag = smoothstep(1.6, 2.15, t) * (1 - smoothstep(2.75, 3.2, t))
        up = smoothstep(2.9, 3.45, t) * (1 - smoothstep(HIT_T, HIT_T + 0.12, t))
        p["crouch"] = 0.10 * sag
        p["head_pitch"] = -0.32 * sag + 0.6 * up
        p["head_tilt"] = 0.06 * sag + 0.04 * math.sin(t * 1.3)
        p["look"] = (lerp(0.1, 0.15, up), lerp(0.35 * sag, -0.95, up))
        p["eye_open"] = lerp(0.62, 0.92, up)
        p["body_lean"] = 0.03 * sag
        p["breathe"] = 1.0 + 1.5 * sag
        p["yaw"] = 0.38
        if t >= HIT_T:
            d = t - HIT_T
            p["bounce"] = 0.35 * _damp(d, 3.2, 6.5)
            p["head_pitch"] = -0.22 * math.exp(-5 * d)
            shut = 1 - smoothstep(0.38, 0.5, d)
            p["eye_open"] = lerp(0.9, 0.0, shut)
            p["blink"] = 0.0
            p["head_tilt"] = -0.1 * math.exp(-4 * d) + 0.05
            p["crouch"] = 0.08 * math.exp(-6 * d)
            if d > 0.18:
                p["expr"] = "pout"
                p["eye_open"] = lerp(0.85, 0.0, shut)
                p["look"] = (0.0, -0.6)
        if t >= 4.6:
            # squeak! head snaps screen-right (snappy with overshoot)
            d = t - 4.6
            s = ease_out_back(clamp(d / 0.14), 2.4)
            p["expr"] = "surprised"
            p["yaw"] = lerp(0.38, 0.72, ease_out(clamp((d - 0.06) / 0.25)))
            p["head_turn"] = 0.42 * s
            p["look"] = (1.0, -0.05)
            p["head_pitch"] = 0.05
            p["eye_open"] = 1.0
            p["blink"] = 0.0
            p["bounce"] = -0.12 * _damp(d, 3.0, 7)
        return p
    if t < CUT_D:
        p["expr"] = "surprised"
        p["yaw"] = 0.75
        p["head_turn"] = 0.3
        follow = clamp((t - 4.9) / 0.4)
        p["look"] = (1.0, lerp(-0.05, 0.1, follow))
        p["eye_open"] = 1.0
        if t >= ROLL_T1:
            d = t - ROLL_T1
            p["bounce"] = 0.18 * _damp(d, 3.0, 7)
            p["blink"] = max(bl, _pulse(d, 0.0, 0.08, 0.04))
            p["body_lean"] = -0.05 * smoothstep(0, 0.25, d)
        return p
    # D: scanned — lean back, hug herself, wide eyes; take on はっけん
    lean = smoothstep(6.5, 7.0, t)
    hug = smoothstep(6.6, 7.05, t)
    p["expr"] = "surprised"
    p["yaw"] = 0.75
    p["head_turn"] = 0.25
    p["body_lean"] = -0.07 * lean - 0.05 * (1 - lean) + 0.012 * math.sin(t * 2.1)
    p["look"] = (0.95, lerp(0.05, 0.25, smoothstep(7.4, 8.0, t)) - 0.25 * smoothstep(8.6, 8.9, t))
    p["blink"] = max(bl, _pulse(t, 7.25, 7.33, 0.04), _pulse(t, 7.45, 7.52, 0.04))
    p["eye_open"] = 1.0
    if hug > 0:
        p["hand_l"] = (lerp(-18, -30, hug), lerp(8, 30, hug), lerp(-200, -262, hug))
        p["hand_r"] = (lerp(18, 30, hug), lerp(8, 30, hug), lerp(-200, -262, hug))
    p["crouch"] = 0.06 * hug
    p["head_tilt"] = -0.05 * lean
    if t >= HAKKEN:
        d = t - HAKKEN
        p["bounce"] = -0.25 * _damp(d, 2.6, 6)       # stretch-up take, settles
        p["eye_open"] = 1.0
        p["head_tilt"] = -0.05 + 0.1 * smoothstep(0.45, 0.8, d)
        p["blush"] = 0.6 * smoothstep(0.3, 0.7, d)
    return p


def kitten_pose(t, tg):
    p = dict(pose="curl", expr="neutral", blink=core.blink(tg, seed=33, period=2.7), t=tg, wet=0.35, facing=1,
             look=(0.0, 0.0))
    x, y = KX0, GY
    if t >= 0.28:
        p["pose"] = "sit"
        p["look"] = (0.3, -0.4)
    if 0.38 <= t < 0.9:
        p["expr"] = "mew"
        p["blink"] = 1.0 if t < 0.75 else p["blink"]
    if t >= 0.28:
        y -= 6 * _damp(t - 0.28, 3, 9) * (t < 0.6)
    if t >= 4.62:
        p["look"] = (1.0, 0.0)
        y -= 10 * max(0.0, math.sin(clamp((t - 4.62) / 0.2) * math.pi))
    if t >= ROLL_T1:
        y -= 14 * max(0.0, math.sin(clamp((t - ROLL_T1) / 0.22) * math.pi))
    if t >= 6.5:
        # scurries behind her leg and peeks out
        u = ease_in_out(clamp((t - 6.5) / 0.45))
        x = lerp(KX0, MX - 22, u)
        y -= 8 * abs(math.sin(u * math.pi * 3)) * (u < 1)
        p["pose"] = "stand" if u < 1 else "sit"
        p["look"] = (1.0, -0.1)
        p["expr"] = "neutral"
    return x, y, p


def robo_state(t, tg):
    if t < ROLL_T0:
        return None
    if t < ROLL_T1:
        u = (t - ROLL_T0) / (ROLL_T1 - ROLL_T0)
        # near-constant roll, hard brake at the end
        pu = 1 - (1 - u) ** 1.7
        x = lerp(RX_START, RX_STOP, pu)
        moving = 1.0
    else:
        d = t - ROLL_T1
        x = RX_STOP - 10 * math.sin(clamp(d / 0.18) * math.pi) * (d < 0.18)
        moving = 0.0
    yaw = -1.3
    if t >= 5.5:
        yaw = lerp(-1.3, -0.55, ease_out_back(clamp((t - 5.5) / 0.32), 1.8))
    face, speak = "dots", core.mouth("robo", tg)[0]
    look = (-1.0, 0.0)
    if t >= ROLL_T1 + 0.02:
        look = (-0.6, 0.2)
    if SCAN_T0 <= t < HAKKEN:
        face = "scan"
    elif t >= HAKKEN:
        face = "surprised" if t < HAKKEN + 0.38 else "happy"
    tilt, bob, squash = 0.0, 0.0, 0.0
    if moving:
        tilt = 0.05 * math.sin(tg * 9.0) + 0.04 * math.sin(tg * 23.0) - 0.03
        bob = 2.5 * abs(math.sin(tg * 26))
        squash = 0.03 * math.sin(tg * 26)
    else:
        d = t - ROLL_T1
        tilt = -0.24 * _damp(d, 2.4, 5.0)         # rocks forward (screen-left), back, settles
        squash = 0.12 * _damp(d, 3.5, 7.0)
        bob = 0.8 * math.sin(tg * 2.0)
    if t >= HAKKEN:
        d = t - HAKKEN
        squash += -0.12 * _damp(d, 3.2, 7.0)       # boing on はっけん
    head_tilt = 0.0
    if t >= 5.5:
        head_tilt = -0.06 * _damp(t - 5.5, 2.0, 5.0)
    if face == "scan":
        head_tilt += 0.02 * math.sin(tg * 3.0)
    antenna = True
    if ROLL_T1 <= t < ROLL_T1 + 0.25:
        antenna = 1.0
    elif SCAN_T0 <= t < HAKKEN:
        antenna = 1.0 if (tg * 6) % 1 < 0.5 else 0.25
    elif HAKKEN <= t < HAKKEN + 0.3:
        antenna = 1.0
    return dict(x=x, yaw=yaw, face=face, speak=speak, look=look, body_tilt=tilt, bob=bob, squash=squash,
                tread_phase=(RX_START - x) / 9.0, antenna_blink=antenna, head_tilt=head_tilt,
                moving=moving, wet=0.4, t=tg)


# ------------------------------------------------------------------ fx
def draw_hat_drip(ctx, t, hat_xy):
    """The fat drip from the scallop that lands on her hat at HIT_T, then splashes."""
    tf0 = HIT_T - FALL_T
    hx, hy = hat_xy
    if tf0 - 0.5 <= t < tf0:          # bead swelling on the scallop tip
        g = smoothstep(tf0 - 0.5, tf0, t)
        draw_droplet(ctx, DRIP_X, DRIP_Y + 4 * g, 4 + 4 * g, alpha=0.9, stretch=1 + 0.5 * g)
    elif tf0 <= t < HIT_T:
        d = t - tf0
        y = DRIP_Y + 0.5 * G_DRIP * d * d
        x = lerp(DRIP_X, hx + 4, d / FALL_T)
        y = min(y, hy)
        draw_droplet(ctx, x, y, 7.5, alpha=0.95, stretch=1.0 + min(1.6, d * 4))
    elif HIT_T <= t < HIT_T + 0.5:
        d = t - HIT_T
        # crown of droplets
        for i in range(9):
            a = -math.pi * (0.1 + 0.8 * i / 8)
            sp = 260 + 90 * ((i * 37) % 5) / 4
            vx, vy = math.cos(a) * sp, math.sin(a) * sp
            x = hx + vx * d
            y = hy - 4 + vy * d + 0.5 * 2200 * d * d
            r = 3.6 * (1 - d / 0.5) + 1
            draw_droplet(ctx, x, y, r, alpha=0.9 * (1 - d / 0.5), stretch=1.3)
        # flat splat on the hat
        k = 1 - smoothstep(0, 0.35, d)
        ctx.save()
        ctx.translate(hx, hy + 3)
        ctx.scale(1, 0.32)
        ctx.arc(0, 0, 12 + 40 * ease_out(d / 0.25), 0, TAU)
        ctx.restore()
        ctx.set_source_rgba(0.88, 0.94, 1.0, 0.55 * k)
        ctx.set_line_width(3)
        ctx.stroke()
        # comic impact ticks
        if d < 0.22:
            a0 = 1 - d / 0.22
            ctx.set_source_rgba(1, 1, 1, 0.9 * a0)
            ctx.set_line_width(4)
            for ang in (-2.5, -1.9, -1.25, -0.65):
                r0, r1 = 36 + 40 * d, 58 + 70 * d
                ctx.move_to(hx + math.cos(ang) * r0, hy + math.sin(ang) * r0)
                ctx.line_to(hx + math.cos(ang) * r1, hy + math.sin(ang) * r1)
            ctx.stroke()


def draw_speed_lines(ctx, t, rx, s):
    """Light motion streaks behind (screen-right of) the rolling robot."""
    ctx.save()
    ctx.set_line_width(3)
    for i in range(7):
        y = GY - 40 - i * 48 * s - 12 * ((i * 7) % 3)
        L = (120 + 60 * ((i * 5) % 3)) * s
        x0 = rx + 150 * s + 40 * ((i * 3) % 4) + (t * 900 + i * 61) % 60
        g = cairo.LinearGradient(x0, y, x0 + L, y)
        g.add_color_stop_rgba(0, 0.9, 0.95, 1, 0.5)
        g.add_color_stop_rgba(1, 0.9, 0.95, 1, 0.0)
        ctx.set_source(g)
        ctx.move_to(x0, y)
        ctx.line_to(x0 + L, y)
        ctx.stroke()
    ctx.restore()


def draw_brake_spray(ctx, t, rx):
    d = t - ROLL_T1 + 0.05
    if not (0 <= d < 0.55):
        return
    for i in range(14):
        h = ((i * 2654435761) % 1000) / 1000
        a = math.pi + 0.25 + 0.9 * h                 # up-left
        sp = 240 + 260 * (((i * 7919) % 100) / 100)
        x = rx - 70 + math.cos(a) * sp * d
        y = GY - 6 - abs(math.sin(a)) * sp * d + 0.5 * 2400 * d * d
        if y > GY + 4:
            continue
        draw_droplet(ctx, x, y, 2.4 + 2 * h, alpha=0.85 * (1 - d / 0.55), stretch=1.4)


def draw_scan(ctx, t, src, mina_surf, cam, scr_h):
    """Cyan scanning beam from Robo's screen onto Mina + scanlines masked to her silhouette."""
    if not (SCAN_T0 <= t < SCAN_T1 + 0.15):
        return
    k = smoothstep(SCAN_T0, SCAN_T0 + 0.18, t) * (1 - smoothstep(SCAN_T1, SCAN_T1 + 0.15, t))
    top, bot = GY - 460, GY + 4
    ph = (t - SCAN_T0) / 1.25
    ys = top + (bot - top) * (0.5 - 0.5 * math.cos(ph * TAU))
    sx, sy = src
    tx = MX + 55
    cyan = (0.44, 0.95, 1.0)
    ctx.save()
    world._apply_cam(ctx, cam)
    ctx.set_operator(cairo.OPERATOR_ADD)
    # wide faint cone
    ctx.move_to(sx, sy - scr_h * 0.35)
    ctx.line_to(tx, top)
    ctx.line_to(tx, bot)
    ctx.line_to(sx, sy + scr_h * 0.35)
    ctx.close_path()
    g = cairo.LinearGradient(sx, sy, tx, sy)
    g.add_color_stop_rgba(0, *cyan, 0.22 * k)
    g.add_color_stop_rgba(1, *cyan, 0.06 * k)
    ctx.set_source(g)
    ctx.fill()
    # bright sweeping plane
    ctx.move_to(sx, sy - 3)
    ctx.line_to(MX - 75, ys - 5)
    ctx.line_to(MX - 75, ys + 5)
    ctx.line_to(sx, sy + 3)
    ctx.close_path()
    g = cairo.LinearGradient(sx, sy, MX - 75, ys)
    g.add_color_stop_rgba(0, *cyan, 0.55 * k)
    g.add_color_stop_rgba(1, *cyan, 0.25 * k)
    ctx.set_source(g)
    ctx.fill()
    ctx.restore()
    # scanlines + bright band, masked to Mina's silhouette (screen space)
    z = cam[2]
    ys_scr = world.to_screen(0, ys, cam)[1]
    y_top = world.to_screen(0, top, cam)[1]
    y_bot = world.to_screen(0, bot, cam)[1]
    ctx.save()
    ctx.push_group()
    ctx.set_source_rgba(*cyan, 0.13 * k)
    step = max(4.0, 7 * z)
    off = (t * 60) % step
    y = y_top - step + off
    while y < y_bot:
        ctx.rectangle(0, y, W, step * 0.35)
        y += step
    ctx.fill()
    g = cairo.LinearGradient(0, ys_scr - 26 * z, 0, ys_scr + 26 * z)
    g.add_color_stop_rgba(0, *cyan, 0)
    g.add_color_stop_rgba(0.5, 0.8, 1.0, 1.0, 0.75 * k)
    g.add_color_stop_rgba(1, *cyan, 0)
    ctx.set_source(g)
    ctx.rectangle(0, ys_scr - 26 * z, W, 52 * z)
    ctx.fill()
    pat = ctx.pop_group()
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.set_source(pat)
    ctx.mask_surface(mina_surf, 0, 0)
    # bright line edge across her
    ctx.restore()


# ------------------------------------------------------------------ post
_VIG = None


def _vig_mask():
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _VIG = (np.clip(d - 0.55, 0, 1) ** 1.5)[..., None].astype(np.float32)
    return _VIG


def _up(a, w, h):
    out = np.empty((h, w, a.shape[2]), np.float32)
    for c in range(a.shape[2]):
        out[..., c] = np.asarray(Image.fromarray(a[..., c], mode="F").resize((w, h), Image.BILINEAR))
    return out


def post(img, frame, close=False):
    rgbp = img[..., :3]
    q = rgbp.reshape(H // 4, 4, W // 4, 4, 3).mean(axis=(1, 3))
    lum = q.max(axis=2, keepdims=True)
    thr = 0.74
    bright = q * np.clip((lum - thr) / (1 - thr), 0, 1)
    g = np.empty_like(bright)
    g2 = np.empty_like(bright)
    for c in range(3):
        g[..., c] = gaussian_filter(bright[..., c], 34 / 4, mode="nearest")
        g2[..., c] = gaussian_filter(bright[..., c], 10 / 4, mode="nearest")
    gl = _up(g * 0.5 + g2 * 0.32, W, H)
    out = 1 - (1 - rgbp) * (1 - np.clip(gl, 0, 1))
    v = _vig_mask() * (0.4 if close else 0.32)
    out = out * (1 - v) + np.array((0.02, 0.02, 0.08), np.float32) * v
    img3 = np.concatenate([out, np.ones_like(out[..., :1])], axis=2)
    core.grain(img3, frame, 0.016)
    return core.finish(img3)


# ------------------------------------------------------------------ render
def _new():
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c = cairo.Context(s)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    return s, c


def render(t: float) -> np.ndarray:
    tg = T0 + t
    frame = int(round(t * core.FPS))
    cam, dof = camera(t)
    close = CUT_B <= t < CUT_C
    rd = dof / 20.0

    mp = mina_pose(t, tg)
    kx, ky, kp = kitten_pose(t, tg)
    rs = robo_state(t, tg)
    kitten_behind = t >= 6.5

    def draw_robo_at(c, glow_only=False):
        if rs is None or rs["x"] > 2300:
            return None
        return draw_robo(c, rs["x"], GY, S, yaw=rs["yaw"], face=rs["face"], speak=rs["speak"],
                         look=rs["look"], body_tilt=rs["body_tilt"], head_tilt=rs["head_tilt"], bob=rs["bob"],
                         squash=rs["squash"], tread_phase=rs["tread_phase"], antenna_blink=rs["antenna_blink"],
                         wet=rs["wet"], t=tg, glow_only=glow_only, screen_glow=1.0)

    def draw_chars(c, ccam):
        c.save()
        world._apply_cam(c, ccam)
        draw_kitten(c, kx, ky, S, **kp)
        draw_mina(c, MX, GY, S, **mp)
        draw_robo_at(c)
        c.restore()

    # background (shopfront + character reflections) is rendered at a fixed per-shot base cam, so the
    # expensive shop layers are built once per shot, then re-projected exactly to the moving cam.
    bcam, bdof = base_cam(t)
    bsurf, bctx = _new()
    world.draw_shopfront(bctx, tg, WEATHER, bcam, dof=bdof)
    if not close:
        world.draw_ground_reflection(bctx, tg, WEATHER, lambda c: draw_chars(c, bcam), bcam)
    surf, ctx = _new()
    ctx.save()
    ctx.translate(960, 540)
    ctx.scale(cam[2], cam[2])
    ctx.translate(bcam[0] - cam[0], bcam[1] - cam[1])
    ctx.scale(1 / bcam[2], 1 / bcam[2])
    ctx.translate(-960, -540)
    ctx.set_source_surface(bsurf, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_BILINEAR)
    ctx.paint()
    ctx.restore()
    world.draw_rain(ctx, tg, WEATHER, "bg", cam, dof=rd)

    # characters
    ctx.save()
    world._apply_cam(ctx, cam)
    if kitten_behind:
        draw_kitten(ctx, kx, ky, S, **kp)
    if rs is not None and rs["moving"] and t > CUT_C:
        draw_speed_lines(ctx, t, rs["x"], S)
    ra = draw_robo_at(ctx)
    if rs is not None:
        draw_brake_spray(ctx, t, rs["x"])
    ctx.restore()

    msurf, mctx = _new()
    mctx.save()
    world._apply_cam(mctx, cam)
    ma = draw_mina(mctx, MX, GY, S, **mp)
    mctx.restore()
    ctx.set_source_surface(msurf, 0, 0)
    ctx.paint()

    ctx.save()
    world._apply_cam(ctx, cam)
    if not kitten_behind:
        draw_kitten(ctx, kx, ky, S, **kp)
    hat = ma["hat_top"]
    draw_hat_drip(ctx, t, (hat[0], hat[1] + 6))
    ctx.restore()

    if ra is not None:
        sw, sh = ra.get("screen_size", (60, 40)) if isinstance(ra.get("screen_size"), (tuple, list)) else (60, 40)
        draw_scan(ctx, t, ra["screen_center"], msurf, cam, sh)

    world.draw_rain(ctx, tg, WEATHER, "mid", cam, dof=rd)
    world.draw_shopfront_fg(ctx, tg, WEATHER, cam)
    world.draw_rain(ctx, tg, WEATHER, "fg", cam, dof=rd)

    img = core.surface_rgba(surf)
    return post(img, frame, close)
