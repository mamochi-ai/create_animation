"""s03 — 15.5–27.0 s: meeting Robo; the tiny-umbrella gag (comedy beat).

Shots (local t):
  A  0.00–3.10  two-shot, slow push.  Mina 「え……ロボット？」, Robo rolls closer, nods.
  B  3.10–4.70  push-in on Robo's head: 「かさ、てんかい。」 salute, crouch/rattle drum-roll, loading.
  C  4.70–5.25  POP!  insert on the tiny umbrella (impact frame), comic hold, a raindrop hits it.
  D  5.25–6.85  Mina: deadpan → 「ちっちゃ！」 (points) → bursts out laughing; kitten looks up.
  E  6.85–7.90  Robo: ? ? + sweat pixel, head tilts.
  F  7.90–9.27  two-shot: loading spinner 「……さいけいさん。」, mini umbrella retracts.
  G  9.27–10.80 punch-in: 「もんだい、なし。」 snap happy, glint, power-up shimmer, salute.
  H  10.80–11.5 two-shot: Mina wipes a laugh-tear, amazed; Robo turns, back seam glows (→ s04).
"""
import math

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from common import core
from common.core import clamp, lerp, smoothstep, ease_in_out, ease_out, ease_in, ease_out_back, noise1, rgb
from lib import world
from lib.characters import draw_mina, draw_robo, draw_kitten, draw_droplet, draw_sparkle

T0 = 15.5
W, H = 1920, 1080
GY = world.GROUND_Y
SM = 0.92          # Mina scale
SR = 0.915         # Robo scale (0.85 x Mina height)
SK = 0.95          # kitten
MINA_X = 760.0
KIT_X = 650.0
ROBO_X0, ROBO_X1 = 1180.0, 1100.0
WEATHER = 0.0
YAW_M = 0.62       # Mina: between front and 3/4 right
YAW_R = -0.9       # Robo: three_quarter_left


def seg(t, a, b):
    return clamp((t - a) / (b - a))


def damp_osc(t, freq, decay):
    if t < 0:
        return 0.0
    return math.exp(-decay * t) * math.sin(t * freq * math.tau)


# ------------------------------------------------------------------ robo performance
def robo_x(t):
    u = seg(t, 1.72, 2.55)
    e = ease_in_out(u)
    return lerp(ROBO_X0, ROBO_X1, e)


def robo_pose(t):
    tg = T0 + t
    x = robo_x(t)
    sp, _, talk = core.mouth("robo", tg)
    p = dict(view=None, yaw=YAW_R, face="dots", speak=sp, look=(0.0, 0.0), body_tilt=0.0, head_tilt=0.0,
             bob=0.0, squash=0.0, tread_phase=(ROBO_X0 - x) / 9.0, antenna_blink=True, arms="rest",
             mini_umbrella=0.0, t=tg, wet=0.15, screen_glow=1.0)
    # idle hum: tiny bob
    p["bob"] = 1.2 * math.sin(tg * 4.1)
    # 0-1.6 dots, blink, looks at Mina
    if t < 1.6:
        p["face"] = "blink" if 0.95 < t < 1.07 else "dots"
        p["look"] = (0.6, -0.2) if t > 0.4 else (0.2, 0.0)
        p["head_tilt"] = 0.06 * smoothstep(0.6, 1.2, t)
    # 1.6-2.55 roll closer: anticipation lean back, lean forward, overshoot settle
    elif t < 3.1:
        p["face"] = "dots" if t < 2.45 else "happy"
        lean = -0.06 * smoothstep(1.6, 1.72, t) * (1 - smoothstep(1.72, 1.85, t))
        lean += 0.10 * smoothstep(1.72, 1.9, t) * (1 - smoothstep(2.3, 2.5, t))
        lean += -0.12 * damp_osc(t - 2.5, 2.2, 5.0)
        p["body_tilt"] = lean
        p["head_tilt"] = -0.5 * lean
        # nod (bob) 2.6-3.0
        nod = seg(t, 2.6, 3.05)
        p["bob"] += 14 * math.sin(nod * math.tau * 2) ** 2 * (nod < 1)
        p["squash"] = 0.05 * math.sin(nod * math.tau * 2) ** 2
        p["look"] = (0.6, 0.0)
    # 3.1-4.7 「かさ、てんかい。」 proud salute, then crouch + rattle
    elif t < 4.7:
        p["face"] = "happy"
        p["arms"] = "salute"
        p["body_tilt"] = -0.05   # chest out
        p["look"] = (0.4, -0.3)
        crouch = smoothstep(3.85, 4.6, t)
        p["squash"] = 0.22 * crouch
        p["bob"] += 16 * crouch
        rattle = smoothstep(3.95, 4.15, t)
        p["mini_umbrella"] = rattle * (0.03 + 0.05 * abs(math.sin(tg * 63)))
        p["head_tilt"] = rattle * 0.025 * math.sin(tg * 71)
        if 4.62 > t > 4.18:
            p["face"] = "loading"
            p["arms"] = "rest"
    # 4.7- POP & hold
    elif t < 6.85:
        tp = t - 4.7
        p["face"] = "happy"
        p["mini_umbrella"] = clamp(ease_out(seg(t, 4.7, 4.98), 2))
        p["squash"] = -0.2 * damp_osc(tp + 0.06, 3.0, 7.0) / 0.8 if tp < 0.9 else 0.0
        p["bob"] += -10 * math.exp(-tp * 9)
        p["body_tilt"] = -0.06
        p["look"] = (0.3, -0.6)
        # raindrop hit at 5.02
        hit = t - 5.02
        if hit > 0:
            p["head_tilt"] = 0.06 * damp_osc(hit, 3.2, 4.0)
            p["mini_umbrella"] = 1.0 - 0.1 * math.exp(-hit * 10)
        if t > 6.1:   # she laughs: confusion
            p["face"] = "question"
            p["look"] = (0.6, 0.0)
            p["head_tilt"] = 0.16 * ease_out_back(seg(t, 6.15, 6.45))
            p["body_tilt"] = -0.06 * (1 - seg(t, 6.1, 6.4))
        p["mini_umbrella"] = max(p["mini_umbrella"], 0.0)
    elif t < 7.9:
        p["face"] = "question"
        p["mini_umbrella"] = 1.0
        p["look"] = (0.6, 0.0)
        # tilt one way then the other (snappy)
        p["head_tilt"] = 0.16 if t < 7.25 else lerp(0.16, -0.18, ease_out_back(seg(t, 7.25, 7.42)))
        p["body_tilt"] = -0.03 * math.sin((t - 6.85) * 3)
    # 7.9-9.27 loading, retract umbrella
    elif t < 9.27:
        p["face"] = "loading"
        p["head_tilt"] = lerp(-0.18, 0.0, ease_in_out(seg(t, 7.9, 8.3)))
        p["mini_umbrella"] = 1.0 - ease_in_out(seg(t, 8.05, 8.85))
        p["bob"] += 2 * math.sin(tg * 9) * seg(t, 8.9, 9.2)    # whirring
    # 9.27- snap happy, glint, salute
    else:
        p["face"] = "happy"
        p["look"] = (0.4, -0.2)
        snap = t - 9.27
        p["squash"] = -0.15 * damp_osc(snap, 2.6, 6.0)
        p["bob"] += -8 * math.exp(-snap * 8)
        if t > 9.42:
            p["arms"] = "salute"
        p["body_tilt"] = -0.05
        if t > 10.8:      # turns slightly (back toward camera)
            u = ease_in_out(seg(t, 10.85, 11.3))
            p["yaw"] = lerp(YAW_R, -1.3, u)
            p["arms"] = "rest" if t > 11.0 else "salute"
            p["face"] = "dots"
            p["look"] = (0.8, 0.0)
            p["body_tilt"] = lerp(-0.05, 0.0, u)
    if p["view"] is None:
        p.pop("view")
    return x, p


# ------------------------------------------------------------------ mina performance
def mina_pose(t):
    tg = T0 + t
    mo, vo, talk = core.mouth("mina", tg)
    p = dict(yaw=YAW_M, expr="neutral", mouth_open=mo, vowel=vo, blink=core.blink(tg, 3), look=(0.6, -0.1),
             head_tilt=0.0, body_lean=0.0, bounce=0.0, wet=0.3, wind=0.3, t=tg, arms="rest")
    p["bounce"] = 0.0
    if t < 3.1:
        # 「え……ロボット？」 0.5-1.6 : lean in, head tilt, surprised -> neutral
        lean = ease_in_out(seg(t, 0.3, 1.0))
        p["body_lean"] = 0.10 * lean
        p["head_tilt"] = 0.16 * ease_out_back(seg(t, 0.75, 1.15))
        p["expr"] = "surprised" if 0.35 < t < 1.25 else "neutral"
        p["arms"] = "hands_clasped" if t > 1.2 else "rest"
        if t > 1.72:   # robo rolls closer: small lean back, blink
            rb = smoothstep(1.72, 2.1, t) * (1 - smoothstep(2.6, 3.1, t))
            p["body_lean"] = lerp(0.10, -0.04, rb) * (1 - 0.3 * seg(t, 2.6, 3.1))
            p["head_tilt"] = lerp(0.16, 0.05, smoothstep(1.7, 2.2, t))
            p["expr"] = "surprised" if t < 2.6 else "happy"
            p["look"] = (0.8, 0.0)
    elif t < 5.25:
        p["expr"] = "happy" if t < 4.3 else "surprised"
        p["arms"] = "hands_clasped"
        p["look"] = (0.8, -0.4)
        p["body_lean"] = 0.05
    elif t < 5.52:
        # deadpan: flat lids, no breathing, staring at the tiny umbrella
        p["expr"] = "neutral"
        p["eye_open"] = 0.5
        p["brow_tilt"] = 0.0
        p["smile"] = 0.0
        p["blink"] = 0.0
        p["look"] = (0.7, -0.6)
        p["head_tilt"] = 0.0
        p["arms"] = "rest"
        p["breathe"] = 0.0
        p["t"] = T0 + 5.25
    elif t < 6.1:
        # 「ちっちゃ！」 point
        u = t - 5.52
        p["expr"] = "surprised"
        p["arms"] = "point"
        p["hand"] = "screen_right"
        p["look"] = (0.7, -0.6)
        p["body_lean"] = 0.08 * ease_out_back(seg(t, 5.52, 5.62))
        p["bounce"] = -0.06 * math.exp(-u * 12)          # stretch on the take
        p["head_pitch"] = 0.25
        p["blink"] = 0.0
    else:
        # bursts into laughter, shoulders shaking, small bounces; calms after ~9.4
        u = t - 6.1
        calm = smoothstep(9.0, 10.3, t)
        p["expr"] = "laugh" if t < 10.0 else "happy"
        p["arms"] = "hands_to_mouth" if t < 9.8 else "hands_clasped"
        shake = (1 - calm) * (0.045 * math.sin(u * 2 * math.pi * 7.5))
        hop = (1 - calm) * 0.10 * abs(math.sin(u * 2 * math.pi * 1.6))
        p["bounce"] = shake + hop * (1 if u > 0.08 else 0) - 0.12 * math.exp(-u * 18)
        p["body_lean"] = -0.06 + 0.10 * ease_out_back(seg(t, 6.1, 6.3)) * (1 - calm) + 0.03 * calm
        p["head_tilt"] = (1 - calm) * (0.10 + 0.05 * math.sin(u * 9))
        p["head_pitch"] = -0.15 * (1 - calm)
        p["look"] = (0.6, 0.0)
        if t > 10.8:
            # wipe a laugh-tear, then curious/amazed toward Robo's glowing back
            wipe = seg(t, 10.8, 11.2)
            p["expr"] = "happy" if t < 11.12 else "amazed"
            p["arms"] = "rest"
            hr = math.sin(wipe * math.pi)
            p["hand_l"] = (lerp(22, 12, hr), lerp(18, 34, hr), lerp(-150, -372, hr)) if t < 11.2 else None
            if p["hand_l"] is None:
                p.pop("hand_l")
            p["eye_open"] = 0.55 if t < 11.12 else None
            if p["eye_open"] is None:
                p.pop("eye_open")
            p["look"] = (0.9, -0.1)
            p["body_lean"] = 0.03 + 0.07 * ease_out_back(seg(t, 11.12, 11.35))
            p["head_tilt"] = 0.08 * seg(t, 11.12, 11.4)
    return p


def kitten_pose(t):
    tg = T0 + t
    look = (0.9, -0.2)
    expr = "neutral"
    if 6.25 < t < 10.5:
        look = (0.6, -1.0)            # looks up at laughing Mina
    if 6.6 < t < 6.9:
        expr = "mew"
    if 4.8 < t < 5.4:
        look = (1.0, -0.6)
    return dict(pose="sit", expr=expr, blink=core.blink(tg, 9, 2.7), t=tg, wet=0.35, facing=1, look=look)


# ------------------------------------------------------------------ camera
def _drift(t, amp=3.0, seed=0):
    return (amp * noise1(t * 0.6, seed + 11), amp * noise1(t * 0.55, seed + 23))


def _umb_anchor():
    """World position of the open tiny umbrella (Robo at rest at ROBO_X1, pose as at the hold)."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 8, 8)
    c = cairo.Context(s)
    a = draw_robo(c, ROBO_X1, GY, SR, yaw=YAW_R, mini_umbrella=1.0, t=T0 + 5.0, face="happy", body_tilt=-0.06)
    return a["umbrella_top"], a["head_top"], a["screen_center"]


UMB, HEAD_TOP, SCREEN_C = _umb_anchor()


def shot(t):
    for name, a, b in (("A", 0, 3.1), ("B", 3.1, 4.7), ("C", 4.7, 5.25), ("D", 5.25, 6.85),
                       ("E", 6.85, 7.9), ("F", 7.9, 9.27), ("G", 9.27, 10.8), ("H", 10.8, 99)):
        if a <= t < b:
            return name, t - a
    return "H", t - 10.8


def camera(t):
    """-> (cam, dof, shake_xy)"""
    s, u = shot(t)
    dx, dy = _drift(t, 2.5)
    if s == "A":
        e = ease_in_out(seg(t, 0, 3.1))
        cx, cy, z = lerp(955, 975, e), lerp(560, 585, e), lerp(1.0, 1.1, e)
        dof = 0.0
    elif s == "B":
        e = ease_in(seg(u, 0, 1.55), 2.2)
        hx, hy = HEAD_TOP[0] - 5, HEAD_TOP[1] + 40
        cx, cy, z = lerp(1060, hx, e), lerp(640, hy, e), lerp(1.45, 2.5, e)
        dof = lerp(4, 12, e)
        sh = smoothstep(3.95, 4.6, t) * 5
        dx += sh * math.sin(t * 83)
        dy += sh * math.cos(t * 97)
    elif s == "C":
        cx, cy = UMB[0] + 15, UMB[1] + 85
        z = 3.1 + 0.08 * math.exp(-u * 9) * 0 + 0.04 * u
        dof = 18
        k = math.exp(-u * 14) * 14
        dx += k * math.sin(u * 120)
        dy += k * math.cos(u * 140)
    elif s == "D":
        cx, cy, z = lerp(745, 750, u / 1.6), lerp(610, 600, u / 1.6), lerp(1.85, 1.9, u / 1.6)
        dof = 8
        if t > 6.1:   # laugh: camera bounces with her a touch
            dy += 3 * math.sin((t - 6.1) * 2 * math.pi * 1.6)
    elif s == "E":
        cx, cy, z = 1080, 575, lerp(1.95, 2.02, u)
        dof = 9
    elif s == "F":
        e = ease_in_out(seg(u, 0, 1.4))
        cx, cy, z = lerp(940, 965, e), lerp(610, 600, e), lerp(1.3, 1.38, e)
        dof = 3
    elif s == "G":
        cx, cy, z = 1085 - 10 * u, 575, 1.85 + 0.03 * u
        k = math.exp(-u * 10) * 9
        dx += k * math.sin(u * 90)
        dy += k * math.cos(u * 110)
        dof = 9
    else:
        e = ease_in_out(seg(u, 0, 0.7))
        cx, cy, z = lerp(930, 945, e), lerp(605, 600, e), lerp(1.28, 1.33, e)
        dof = 3
    return (cx + dx / z, cy + dy / z, z), dof


# ------------------------------------------------------------------ drawing helpers
def _tint_group(ctx, draw, mul, strength=0.65):
    """Draw characters, then multiply them by the ambient char tint (only where drawn)."""
    ctx.push_group()
    out = draw(ctx)
    pat = ctx.pop_group()
    ctx.push_group()
    ctx.set_source(pat)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_MULTIPLY)
    ctx.set_source_rgb(*[1 - strength * (1 - c) for c in mul])
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_DEST_IN)
    ctx.set_source(pat)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.pop_group_to_source()
    ctx.paint()
    return out


def _shimmer_robo(ctx, x, rp, k):
    """Robo with a diagonal power-up light band sweeping over the body (k = 0..1 sweep)."""
    ctx.push_group()
    a = draw_robo(ctx, x, GY, SR, **rp)
    ctx.set_operator(cairo.OPERATOR_ATOP)
    bx = lerp(x - 260 * SR, x + 260 * SR, k)
    g = cairo.LinearGradient(bx - 80, GY - 200, bx + 80, GY - 300)
    al = 0.7 * math.sin(k * math.pi)
    g.add_color_stop_rgba(0, 0.6, 1, 1, 0)
    g.add_color_stop_rgba(0.5, 0.85, 1, 1, al)
    g.add_color_stop_rgba(1, 0.6, 1, 1, 0)
    ctx.set_source(g)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.pop_group_to_source()
    ctx.paint()
    return a


def _pixel_sweat(ctx, a, u):
    """Tiny cyan pixel sweat drop on Robo's screen (u: 0..1 slide)."""
    cx, cy = a["screen_center"]
    sw, sh = a["screen_size"]
    if sw < 5:
        return
    px = max(2.0, sh / 13)
    x0 = cx + sw * 0.30
    y0 = cy - sh * 0.38 + u * sh * 0.12
    cells = [(0, 0), (0, 1), (-1, 2), (0, 2), (1, 2), (-1, 3), (0, 3), (1, 3), (0, 4)]
    ctx.set_source_rgba(0.44, 0.95, 1.0, 0.25)
    for i, j in cells:
        ctx.rectangle(x0 + i * px - px * 0.5 - px * 0.4, y0 + j * px - px * 0.4, px * 1.8, px * 1.8)
    ctx.fill()
    ctx.set_source_rgba(0.6, 0.97, 1.0, 0.95)
    for i, j in cells:
        ctx.rectangle(x0 + i * px - px * 0.45, y0 + j * px + 0.05 * px, px * 0.9, px * 0.9)
    ctx.fill()


def _speed_lines(ctx, x0, x1, ys, alpha, seed=0, col=(0.92, 0.96, 1.0)):
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i, y in enumerate(ys):
        r = noise1(i * 3.1, seed) * 0.5 + 0.5
        ctx.set_source_rgba(*col, alpha * (0.4 + 0.6 * r))
        ctx.set_line_width(2 + 3 * r)
        ctx.move_to(x0 + r * 40, y)
        ctx.line_to(x1 - r * 60, y)
        ctx.stroke()


def _impact(ctx, cx, cy, k, t):
    """Comic impact: radial lines + ring around (cx,cy) in screen space, k = 1 -> 0."""
    if k <= 0.01:
        return
    ctx.save()
    ctx.set_source_rgba(1, 1, 1, 0.22 * k)
    ctx.paint()
    n = 28
    for i in range(n):
        a = i / n * math.tau + 0.13 * noise1(i, 5)
        r0 = 170 + 70 * (noise1(i * 1.7, 2) * 0.5 + 0.5) + (1 - k) * 160
        r1 = r0 + 260 + 200 * (noise1(i * 2.3, 4) * 0.5 + 0.5)
        w = 10 * k * (0.5 + 0.5 * noise1(i * 0.9, 1))
        ca, sa = math.cos(a), math.sin(a)
        nx, ny = -sa, ca
        ctx.move_to(cx + ca * r0, cy + sa * r0)
        ctx.line_to(cx + ca * r1 + nx * w, cy + sa * r1 + ny * w)
        ctx.line_to(cx + ca * r1 - nx * w, cy + sa * r1 - ny * w)
        ctx.close_path()
    ctx.set_source_rgba(1, 1, 1, 0.75 * k)
    ctx.fill()
    ctx.arc(cx, cy, 90 + (1 - k) * 260, 0, math.tau)
    ctx.set_line_width(7 * k)
    ctx.set_source_rgba(1, 0.97, 0.85, 0.8 * k)
    ctx.stroke()
    ctx.restore()


# ------------------------------------------------------------------ post
_VIG = None


def _post(img, frame, threshold=0.72, radius=30.0, strength=0.55, vig=0.32):
    global _VIG
    rgbp = img[..., :3]
    small = rgbp[::2, ::2]
    lum = small.max(axis=2, keepdims=True)
    bright = small * np.clip((lum - threshold) / (1 - threshold), 0, 1)
    q = bright[::2, ::2]   # quarter res for the wide halo
    g1 = np.stack([gaussian_filter(q[..., c], radius / 4, mode="nearest") for c in range(3)], -1)
    g2 = np.stack([gaussian_filter(bright[..., c], radius * 0.15, mode="nearest") for c in range(3)], -1)
    g = np.repeat(np.repeat(g1, 2, 0), 2, 1)[:540, :960] * strength + g2 * strength * 0.6
    gi = Image.fromarray(np.clip(g * 255, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    g = np.asarray(gi, np.float32) / 255.0
    out = 1 - (1 - rgbp) * (1 - g)
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _VIG = (1 - np.clip(d - 0.55, 0, 1) ** 1.5 * vig)[..., None].astype(np.float32)
    out = out * _VIG + 0.02 * (1 - _VIG)
    core.grain(out, frame, 0.016)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ render
def render(t: float) -> np.ndarray:
    tg = T0 + t
    frame = int(round(tg * core.FPS))
    cam, dof = camera(t)
    s, u = shot(t)
    C = world.palette(WEATHER)
    surf, ctx = core.new_surface()
    world.draw_shopfront(ctx, tg, WEATHER, cam, dof=dof)

    rx, rp = robo_pose(t)
    mp = mina_pose(t)
    kp = kitten_pose(t)
    shimmer = seg(t, 9.3, 9.95) if 9.27 <= t < 10.2 else None

    def chars(c, reflect=False):
        c.save()
        world._apply_cam(c, cam)
        ka = draw_kitten(c, KIT_X, GY, SK * 0.92, **kp)
        ma = draw_mina(c, MINA_X, GY, SM, **mp)
        if shimmer is not None and not reflect:
            ra = _shimmer_robo(c, rx, rp, shimmer)
        else:
            ra = draw_robo(c, rx, GY, SR, **rp)
        c.restore()
        return ma, ra, ka

    if cam[2] < 1.6:
        world.draw_ground_reflection(ctx, tg, WEATHER, lambda c2: chars(c2, True), cam)

    # smear / speed lines behind Robo while rolling
    if 1.8 < t < 2.45:
        sp = math.sin(seg(t, 1.8, 2.45) * math.pi)
        ctx.save()
        world._apply_cam(ctx, cam)
        _speed_lines(ctx, rx + 90, rx + 330, [GY - 40 - 34 * i for i in range(9)], 0.35 * sp, seed=3)
        ctx.restore()

    ma, ra, ka = _tint_group(ctx, chars, C["char_mul"])

    ctx.save()
    world._apply_cam(ctx, cam)
    # sweat pixel on Robo's screen
    if 6.4 < t < 7.9:
        _pixel_sweat(ctx, ra, seg(t, 6.4, 7.9))
    # laugh tear at Mina's eye
    if 6.6 < t < 11.05:
        eye = ma["eyes"][0] if ma["eyes"] else ma["face"]
        ex = max(e[0] for e in ma["eyes"]) if ma["eyes"] else eye[0]
        ey = ma["eyes"][0][1]
        grow = smoothstep(6.6, 7.2, t) * (1 - smoothstep(10.9, 11.05, t))
        draw_droplet(ctx, ex + 14 * SM, ey + 6 * SM, 5.5 * SM * grow, 0.9, 1.15)
    # determined glint on Robo's screen corner (G)
    if 9.27 <= t < 10.1:
        k = seg(t, 9.3, 9.75)
        gx, gy = ra["screen_center"]
        sw, sh = ra["screen_size"]
        g = math.sin(k * math.pi)
        draw_sparkle(ctx, gx + sw * 0.42, gy - sh * 0.45, 26 * SR * g, g, k * 1.4, "#E8FFFF")
        # little power-up sparkles around the body
        for i in range(7):
            ph = (t * 1.8 + i * 0.37) % 1.0
            if t < 9.35:
                continue
            ang = i * 2.1
            px = rx + math.cos(ang) * 120 * SR
            py = GY - 180 * SR + math.sin(ang) * 140 * SR - ph * 60
            a_ = math.sin(ph * math.pi) * (1 - seg(t, 9.9, 10.1))
            draw_sparkle(ctx, px, py, 12 * SR * a_, a_, i, "#C8FFFF")
    # back panel seam glow (sets up s04)
    if t > 10.75:
        gl = smoothstep(10.9, 11.4, t) * (0.75 + 0.25 * math.sin(tg * 11))
        ch = ra["chest"]
        bx = ch[0] + 70 * SR
        y0, y1 = ch[1] - 30 * SR, ch[1] + 120 * SR
        for wdt, al in ((16, 0.10), (8, 0.22), (3, 0.9)):
            ctx.set_line_width(wdt * SR)
            ctx.set_source_rgba(0.55, 1.0, 0.95, al * gl)
            ctx.move_to(bx, y0)
            ctx.curve_to(bx + 12 * SR, y0 + 50 * SR, bx + 14 * SR, y1 - 50 * SR, bx + 6 * SR, y1)
            ctx.stroke()
    # C: the raindrop that lands on the tiny umbrella
    if s == "C":
        ut = ra.get("umbrella_top") or UMB
        hit = 5.02
        if t < hit:
            fy = ut[1] - (hit - t) * 1400
            draw_droplet(ctx, ut[0] + 2, fy, 4.2, 0.95, 2.4)
        elif t < hit + 0.35:
            k = (t - hit) / 0.35
            for i in range(6):
                a = math.pi * (1.1 + 0.8 * i / 5)
                d = 8 + 45 * k
                draw_droplet(ctx, ut[0] + math.cos(a) * d, ut[1] - 4 + math.sin(a) * d * 0.9 + 140 * k * k,
                             2.2 * (1 - k * 0.6), 0.9 * (1 - k), 1.0)
    ctx.restore()

    world.draw_rain(ctx, tg, WEATHER, "bg", cam, dof=dof / 20)
    world.draw_rain(ctx, tg, WEATHER, "mid", cam, dof=dof / 20)
    world.draw_shopfront_fg(ctx, tg, WEATHER, cam)
    world.draw_rain(ctx, tg, WEATHER, "fg", cam, dof=dof / 20)

    # screen-space comic FX
    if s == "C" and u < 0.2:
        ux, uy = world.to_screen(UMB[0], UMB[1], cam)
        _impact(ctx, ux, uy, 1 - u / 0.2, t)
    if s == "D" and 5.52 <= t < 5.75:     # take: radial "!" burst lines behind the frame edges
        k = 1 - seg(t, 5.52, 5.75)
        hx, hy = world.to_screen(ma["head"][0], ma["head"][1], cam)
        ctx.save()
        for i in range(3):
            a = -1.2 + i * 0.35
            ctx.move_to(hx + 200 * math.cos(a) + 40, hy + 200 * math.sin(a) - 40)
            ctx.line_to(hx + 290 * math.cos(a) + 40, hy + 290 * math.sin(a) - 40)
        ctx.set_line_width(9)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_source_rgba(1, 1, 1, 0.9 * k)
        ctx.stroke()
        ctx.restore()
    if s == "B" and t > 4.0:   # drum-roll: rattling tick marks around the hatch
        hx, hy = world.to_screen(ra["hatch"][0], ra["hatch"][1], cam)
        r = smoothstep(4.0, 4.4, t)
        ctx.save()
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_width(5)
        ctx.set_source_rgba(1, 1, 1, 0.75 * r)
        j = int(t * 24) % 2
        for side in (-1, 1):
            for i in range(3):
                a = -math.pi / 2 + side * (0.5 + 0.28 * i) + 0.06 * j
                ctx.move_to(hx + math.cos(a) * 70, hy + math.sin(a) * 70)
                ctx.line_to(hx + math.cos(a) * (100 + 10 * j), hy + math.sin(a) * (100 + 10 * j))
        ctx.stroke()
        ctx.restore()

    img = core.surface_rgba(surf)
    if s == "C":
        return _post(img, frame, threshold=0.7, strength=0.6)
    return _post(img, frame)
