"""s05 (38.0-49.0 s) -- Robo glitches in the rain; Mina shields him with her randoseru.

Shots (local seconds):
  A 0.0-3.6  close-up on Robo's screen: X X glitch, sparks, droplets running down the glass,
             stutter jitter / frame skips / RGB split in sync with 「ぼうすいきのう……ゼロ、です。」
             (Mina defocused in the left foreground).
  B 3.6-4.5  close-up on Mina under the canopy: stares, then puffs up (pout + anger mark), trembles.
  C 4.5-6.0  medium-wide: 「もう！」 stomp, swings the randoseru off her back in a whoosh arc, steps out
             into the rain beside Robo and holds it over his head like a little roof (5.5).
  D 6.0-8.5  gentle two-shot: static clears, Robo looks up at the roof, then at Mina; kitten nuzzles
             his tread; 「……あたたかい。」 -> warm orange face, warm glow spreading.
  E 8.5-11.0 wide slow push-in on the trio under canopy + randoseru; warm power-up hum, steady antenna,
             Mina giggles, the rain thins.
"""
import math

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from common import core
from common.core import clamp, lerp, smoothstep, ease_in_out, ease_out, ease_out_back, noise1, W, H
from lib import world
from lib.characters import draw_mina, draw_robo, draw_kitten, draw_droplet, draw_drops_on, draw_rim_drips

T0 = 38.0
FPS = 24
WEATHER = 0.0

GY = 975.0
S = world.shop_scale(GY)
MX0 = 820.0                       # Mina's spot under the canopy (s04)
RX = MX0 + 300 * S                # Robo
MX1 = RX - 170 * S                # Mina beside Robo, holding the roof
KX0 = MX0 - 70 * S                # kitten at Mina's feet
KX1 = RX - 66 * S                 # kitten at Robo's tread
CAN_OFF = (560.0, 0.0)            # canopy pushed a bit further over Mina: Robo is fully in the rain
WARM = core.rgb("#FFB45E")

# canopy geometry (world coords, from draw_robo anchors with CAN_OFF)
CAN_CX, CAN_CY, CAN_R, CAN_RY = 488.0, 287.0, 580.0, 55.0


def _h(*a):
    x = math.sin(sum((i + 1) * 12.9898 * v for i, v in enumerate(a)) + 78.233) * 43758.5453
    return x - math.floor(x)


# ------------------------------------------------------------------ timing helpers
GLITCH_HITS = [0.62, 0.71, 0.80, 1.05, 1.16, 1.58, 1.66, 1.92, 2.06, 2.18, 2.42, 2.88, 2.97, 3.12, 3.32]


def glitch_amt(t):
    """0..1 stutter intensity (bursts synced to the stutters + occasional random hits)."""
    if t >= 6.3:
        return 0.0
    g = 0.0
    for h in GLITCH_HITS:
        d = t - h
        if 0 <= d < 0.12:
            g = max(g, 1.0 - d / 0.12)
    f = int(round(t * FPS))
    if _h(f, 3.1) < (0.10 if t < 3.6 else 0.05):
        g = max(g, 0.5 + 0.5 * _h(f, 7.7))
    if 6.0 <= t < 6.3:       # last flickers while clearing
        g *= 0.6
    return g


def frame_hold(t):
    """Frame-skip feel: on hit frames Robo's animation time freezes on the previous frame."""
    f = int(round(t * FPS))
    if t < 6.2 and glitch_amt(t) > 0.55 and _h(f, 5.5) < 0.6:
        return (f - 1 - int(_h(f, 9.1) * 2)) / FPS
    return t


def spring_k(x, freq=2.4, damp=5.0):
    if x <= 0:
        return 0.0
    return math.exp(-damp * x) * math.sin(TAU * freq * x)


TAU = 2 * math.pi


def ramp(t, a, b, f=ease_in_out):
    return f(clamp((t - a) / (b - a)))


# ------------------------------------------------------------------ cameras
def camera(t):
    if t < 3.6:          # A
        u = ramp(t, 0.0, 3.6)
        cx = 1172 + 6 * math.sin(t * 0.9) + 3 * noise1(t * 1.3, 3)
        cy = 585 + 4 * math.sin(t * 0.7 + 1) + 3 * noise1(t * 1.1, 4)
        z = 2.25 + 0.18 * u
        g = glitch_amt(t)
        if g > 0.5:
            cx += (_h(int(t * FPS), 1.3) - 0.5) * 6
        return (cx, cy, z), "A"
    if t < 4.5:          # B
        u = ramp(t, 3.6, 4.5)
        cx = 860 + 4 * noise1(t * 1.2, 5)
        cy = 480 + 3 * noise1(t * 1.4, 6)
        z = 2.05 + 0.12 * u
        # tiny tremble as she fumes
        k = smoothstep(3.95, 4.15, t)
        cx += k * 1.5 * math.sin(t * 61)
        return (cx, cy, z), "B"
    if t < 6.0:          # C
        u = ramp(t, 4.7, 5.9)
        cx = lerp(950, 1040, u)
        cy = lerp(615, 600, u)
        z = 1.1 + 0.05 * u
        # stomp shake
        sh = max(0.0, 1 - (t - 4.62) / 0.25) if t >= 4.62 else 0.0
        cy += sh * 9 * math.sin((t - 4.62) * 90)
        # randoseru landing thump
        sh2 = max(0.0, 1 - (t - 5.5) / 0.2) if t >= 5.5 else 0.0
        cy += sh2 * 4 * math.sin((t - 5.5) * 80)
        return (cx, cy, z), "C"
    if t < 8.5:          # D
        u = ramp(t, 6.0, 8.5, lambda x: x)
        cx = lerp(1105, 1122, u) + 2 * noise1(t * 0.8, 8)
        cy = lerp(665, 648, u) + 2 * noise1(t * 0.7, 9)
        z = 1.42 + 0.16 * ease_in_out(u)
        return (cx, cy, z), "D"
    u = ramp(t, 8.5, 11.0)                                            # E
    cx = lerp(930, 1035, u)
    cy = lerp(545, 600, u)
    z = 0.93 + 0.17 * u
    return (cx, cy, z), "E"


SHOT_RANGES = {"A": (0.0, 3.6), "B": (3.6, 4.5), "C": (4.5, 6.0), "D": (6.0, 8.5), "E": (8.5, 11.0)}
_MASTER = {}


def master_cam(shot):
    """One fixed background camera per shot covering every frame's view (the shop plates are
    expensive to rebuild per cam, so each frame re-frames this plate with a 2D transform)."""
    if shot not in _MASTER:
        a, b = SHOT_RANGES[shot]
        n = int(round((b - a) * FPS))
        x0 = y0 = 1e9
        x1 = y1 = -1e9
        zmin = 1e9
        for i in range(n + 1):
            c, _ = camera(min(b - 1e-4, a + i / FPS))
            hw, hh = 960 / c[2], 540 / c[2]
            x0, x1 = min(x0, c[0] - hw), max(x1, c[0] + hw)
            y0, y1 = min(y0, c[1] - hh), max(y1, c[1] + hh)
            zmin = min(zmin, c[2])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        z = min(zmin, 1920 / (x1 - x0 + 8), 1080 / (y1 - y0 + 8))
        _MASTER[shot] = (round(cx, 1), round(cy, 1), round(z, 4))
    return _MASTER[shot]


def plate_xform(ctx, cam, mcam):
    """Map master-cam screen space onto the current cam's screen space."""
    cx, cy, z = cam
    mx, my, mz = mcam
    ctx.translate(960 + (mx - cx) * z, 540 + (my - cy) * z)
    ctx.scale(z / mz, z / mz)
    ctx.translate(-960, -540)


# ------------------------------------------------------------------ character states
def robo_pose(t):
    tg = T0 + t
    th = frame_hold(t)
    g = glitch_amt(t)
    sp = core.mouth("robo", T0 + th)[0]
    if 0.6 <= t <= 3.5 and g > 0.6:
        sp *= 0.3 + 0.7 * _h(int(t * FPS), 2.2)      # stutter chops the waveform
    p = dict(yaw=lerp(-0.62, -0.2, ramp(t, 6.05, 6.7)), canopy=1.0, canopy_offset=CAN_OFF, wet=1.0, t=th,
             speak=sp, arms="rest", seed=7)
    # face
    f = int(round(t * FPS))
    if t < 6.15:
        face = "glitch"
    elif t < 6.4:
        face = "glitch" if _h(f, 4.4) < 0.45 else ("blink" if t < 6.25 else "dots")
    elif t < 7.55:
        face = "blink" if 7.0 <= t < 7.12 else "dots"
    else:
        face = "warm"
    p["face"] = face
    # screen brightness
    if t < 6.3:
        p["screen_glow"] = 0.55 + 0.45 * _h(f, 6.6) if g > 0.3 else 0.8
    elif t < 7.55:
        p["screen_glow"] = 0.9
    else:
        p["screen_glow"] = lerp(0.85, 1.0, ramp(t, 7.55, 7.9))
    # look
    look = (0.0, 0.3)
    if 6.35 <= t < 7.0:
        look = (0.2, -1.0 * ramp(t, 6.35, 6.6))
    elif 7.0 <= t < 7.6:
        k = ramp(t, 7.0, 7.3)
        look = (lerp(0.2, -1.0, k), lerp(-1.0, -0.3, k))
    elif t >= 7.6:
        look = (-0.6, -0.2)
    p["look"] = look
    # head / body
    droop = -0.07 * ramp(t, 0.0, 1.2) if t < 6.0 else lerp(-0.07, 0.0, ramp(t, 6.0, 6.5))
    jerk = (_h(f, 8.8) - 0.5) * 0.08 * g if t < 6.2 else 0.0
    lookup = 0.07 * ramp(t, 6.35, 6.7) * (1 - ramp(t, 7.0, 7.4)) if 6.3 < t < 7.6 else 0.0
    lean = -0.045 * ramp(t, 7.6, 8.4) if t >= 7.6 else 0.0      # leans into the warmth (toward Mina)
    p["head_tilt"] = droop + jerk + lookup + lean
    p["body_tilt"] = 0.015 * math.sin(tg * 1.3) + 0.3 * lean
    bob = 2.0 * math.sin(tg * 2.1)
    if t >= 4.62:   # flinch at the stomp
        bob += 6 * spring_k(t - 4.62, 3.0, 6.0)
    if t >= 8.4:    # power-up hum: a soft rising lift
        bob -= 3 * ramp(t, 8.4, 9.2) + 1.0 * math.sin(tg * 9) * ramp(t, 8.4, 9.0) * (1 - ramp(t, 9.0, 10.0))
    p["bob"] = bob
    # sparks
    if t < 6.0:
        sk = 0.45 + 0.55 * g
    else:
        sk = 0.45 * (1 - ramp(t, 6.0, 6.45))
    p["sparks"] = sk
    # antenna bulb
    if t < 6.3:
        p["antenna_blink"] = 0.15 + 0.5 * _h(f, 1.9) * (0.4 + g)
    elif t < 8.3:
        p["antenna_blink"] = True
    else:
        p["antenna_blink"] = lerp(0.6, 1.0, ramp(t, 8.3, 8.8)) * (0.92 + 0.08 * math.sin(tg * 4))
    jit = (0.0, 0.0)
    if t < 6.2 and g > 0.5:
        jit = ((_h(f, 2.7) - 0.5) * 14 * g, (_h(f, 3.9) - 0.5) * 6 * g)
    return p, jit


def mina_state(t):
    tg = T0 + t
    o, v, talking = core.mouth("mina", tg)
    st = dict(view="three_quarter_right", t=tg, wet=0.3, wind=0.4, wind_dir=1,
              blink=core.blink(tg, 11), look=(0.6, 0.0))
    x = MX0
    bounce = 0.0
    lean = 0.0
    crouch = 0.0
    expr = "worried"
    if t < 3.6:
        expr = "worried"
        st["look"] = (0.8, -0.1)
        st["head_tilt"] = 0.04 * ramp(t, 1.4, 2.4)
    elif t < 3.95:
        expr = "neutral"
        st["look"] = (0.9, -0.05)
        st["eye_open"] = 1.0
        st["blink"] = 0.0 if t < 3.8 else core.blink(tg, 11)
    elif t < 6.0:
        expr = "pout"
        k = smoothstep(3.95, 4.05, t)
        bounce += 0.06 * spring_k(t - 3.95, 3.0, 5.0)
        # fuming tremble
        x += k * 1.4 * math.sin(t * 57) * (1 - smoothstep(4.5, 4.6, t))
    if talking:
        st["mouth_open"], st["vowel"] = o, v
    # stomp at 4.6
    if 4.45 <= t < 5.0:
        if t < 4.6:
            bounce -= 0.12 * smoothstep(4.45, 4.6, t)          # anticipation: rises on her toes
            st["tiptoe"] = smoothstep(4.45, 4.58, t) * 0.6
        else:
            bounce += 0.28 * math.exp(-(t - 4.6) * 9) * math.cos((t - 4.6) * 20)
    # randoseru swing 5.0 - 5.5, steps to Robo
    ht = (RX - 10 * S, GY - 453 * S)  # roof spot over Robo's head (world)
    if t < 5.02:
        st["arms"] = "rest"
        st["randoseru"] = "back"
        if t >= 4.88:   # anticipation: shoulders down, lean back
            k = smoothstep(4.88, 5.02, t)
            crouch = 0.15 * k
            lean = -0.08 * k
    else:
        st["arms"] = "hold_randoseru_overhead"
        u = clamp((t - 5.02) / 0.48)
        ue = ease_out(u, 2.4)
        x = lerp(MX0, MX1, ease_in_out(clamp((t - 5.05) / 0.45)))
        # two little steps
        step = math.sin(clamp((t - 5.05) / 0.45) * 2 * math.pi)
        bounce += -0.05 * abs(step) * (1 if t < 5.5 else 0)
        # arc of the box: from behind her shoulder up over her head and over to Robo
        p0 = (x - 60 * S, GY - 300 * S)
        p1 = (x - 30 * S, GY - 590 * S)
        p2 = (ht[0] - 70 * S, ht[1] - 90 * S)
        p3 = ht
        a = 1 - ue
        bx = a ** 3 * p0[0] + 3 * a * a * ue * p1[0] + 3 * a * ue * ue * p2[0] + ue ** 3 * p3[0]
        by = a ** 3 * p0[1] + 3 * a * a * ue * p1[1] + 3 * a * ue * ue * p2[1] + ue ** 3 * p3[1]
        # settle: tiny overshoot down then up after landing, then breathing
        if t >= 5.5:
            by += 10 * S * spring_k(t - 5.5, 2.2, 5.0) + 2 * S * math.sin(tg * 2.0)
        # follow Robo's bob a little
        st["hold_target"] = (bx, by)
        st["hold_tilt"] = lerp(-1.3, 0.0, ease_out(u, 2)) + 0.25 * spring_k(t - 5.5, 1.8, 4.0)
        st["hold_flap"] = clamp(0.9 * (1 - u) + 0.35 * abs(spring_k(t - 5.2, 1.6, 2.6)))
        st["tiptoe"] = 0.35 * smoothstep(5.3, 5.5, t)
        st["wet"] = lerp(0.3, 0.6, ramp(t, 5.5, 8.5))
        if t >= 6.0:
            expr = "gentle"
            st["look"] = (0.7, -0.15)
            st["head_tilt"] = 0.06 * ramp(t, 6.0, 7.0)
        if 7.4 <= t < 8.6:   # listens to あたたかい, a little surprised then soft
            expr = "gentle"
            st["eye_open"] = lerp(1.15, 1.0, ramp(t, 7.6, 8.3))
            st["blush"] = lerp(0.5, 0.9, ramp(t, 7.8, 8.6))
        if 9.0 <= t < 10.0:   # giggle
            expr = "laugh"
            k = (t - 9.0)
            bounce += 0.04 * abs(math.sin(k * 14)) * (1 - k)
            st["head_tilt"] = 0.08 + 0.03 * math.sin(k * 14)
            st["mouth_open"] = 0.35 + 0.2 * abs(math.sin(k * 14))
            st["vowel"] = "i"
        elif t >= 10.0:
            expr = "happy"
            st["look"] = (0.6, -0.2)
            st["head_tilt"] = 0.08
            st["blush"] = 0.9
    st["expr"] = expr
    st["bounce"] = bounce
    st["body_lean"] = lean
    st["crouch"] = crouch
    return x, st


def kitten_state(t):
    tg = T0 + t
    if t < 4.62:
        return KX0, GY + 10, dict(pose="sit", expr="neutral", facing=1, look=(0.5, -0.5))
    if t < 5.6:
        hop = 30 * S * max(0.0, math.sin(clamp((t - 4.62) / 0.3) * math.pi)) if t < 4.92 else 0.0
        return KX0, GY + 10 - hop, dict(pose="sit", expr="mew" if t < 5.0 else "neutral", facing=1)
    if t < 6.25:
        u = ease_in_out(clamp((t - 5.6) / 0.65))
        return lerp(KX0, KX1, u), GY + 12 - 4 * S * abs(math.sin(t * 18)), dict(pose="stand", facing=1)
    return KX1, GY + 12, dict(pose="nuzzle", expr="happy" if t > 6.6 else "neutral", facing=1)


# ------------------------------------------------------------------ drawing helpers
def _with_cam(ctx, cam):
    ctx.save()
    world._apply_cam(ctx, cam)


def draw_screen_rivulets(ctx, a, t, amt=1.0):
    """Water running down Robo's screen glass (world coords via anchors)."""
    (sx, sy), (sw, sh) = a["screen_center"], a["screen_size"]
    sc = a["scale"]
    ctx.save()
    ctx.rectangle(sx - sw / 2, sy - sh / 2, sw, sh)
    ctx.clip()
    for i in range(7):
        per = 1.1 + 0.9 * _h(i, 1)
        u = ((t + _h(i, 2) * per) % per) / per
        x = sx - sw * 0.45 + sw * 0.9 * _h(i, 3) + 3 * sc * math.sin(u * 5 + i)
        y = sy - sh / 2 + (sh + 20 * sc) * (u ** 1.6)
        # trail
        ctx.move_to(x, y - 26 * sc * (0.3 + u))
        ctx.line_to(x, y)
        ctx.set_source_rgba(0.8, 0.95, 1.0, 0.18 * amt)
        ctx.set_line_width(2.2 * sc)
        ctx.stroke()
        draw_droplet(ctx, x, y, (2.2 + 1.2 * _h(i, 4)) * sc, 0.75 * amt, 1.3)
    for i in range(10):   # static beads
        x = sx - sw * 0.46 + sw * 0.92 * _h(i, 11)
        y = sy - sh * 0.46 + sh * 0.92 * _h(i, 12)
        draw_droplet(ctx, x, y, (1.0 + 1.0 * _h(i, 13)) * sc, 0.55 * amt)
    ctx.restore()


def draw_roof_splashes(ctx, m, t, sc, amt=1.0):
    """Rain drops bouncing off the top of the held randoseru."""
    x0, x1 = m["randoseru_span"]
    ty = m["randoseru_top"][1]
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i in range(9):
        per = 0.28 + 0.2 * _h(i, 21)
        u = ((t + _h(i, 22) * per) % per) / per
        if u > 0.45:
            continue
        k = u / 0.45
        x = lerp(x0 + 8 * sc, x1 - 8 * sc, _h(i, 23, math.floor((t + _h(i, 22) * per) / per)))
        y = ty + 2 * sc
        a = (1 - k) * 0.8 * amt
        r = (4 + 9 * k) * sc
        for sgn in (-1, 1):
            ctx.move_to(x + sgn * 2 * sc, y)
            ctx.curve_to(x + sgn * r * 0.4, y - r * 1.1, x + sgn * r * 0.9, y - r * 0.9, x + sgn * r, y - r * 0.2)
            ctx.set_source_rgba(0.9, 0.96, 1.0, a)
            ctx.set_line_width(1.6 * sc)
            ctx.stroke()
        draw_droplet(ctx, x - 0.6 * r, y - r * 1.3 * (1 - k) - 2 * sc, 1.4 * sc, a)
        draw_droplet(ctx, x + 0.7 * r, y - r * 1.1 * (1 - k) - 2 * sc, 1.2 * sc, a)
    ctx.restore()


def draw_whoosh(ctx, t, x_of, sc):
    """Smear arcs following the randoseru swing (5.08-5.5)."""
    if not (5.06 <= t < 5.55):
        return
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    head = clamp((t - 5.02) / 0.48)
    tail = clamp((t - 5.16) / 0.48)
    fade = 1 - smoothstep(5.4, 5.55, t)
    for j, (off, a, lw) in enumerate([(0, 0.55, 10), (-26, 0.35, 5), (24, 0.3, 4)]):
        pts = []
        for k in range(len(x_of)):
            u = lerp(tail, head, k / (len(x_of) - 1))
            ue = ease_out(u, 2.4)
            st = _interp_path(x_of, u)
            if st is None:
                continue
            pts.append((st[0] + off * sc * (1 - ue), st[1] + off * sc * 0.5))
        if len(pts) < 2:
            continue
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.set_source_rgba(1, 1, 1, a * fade)
        ctx.set_line_width(lw * sc)
        ctx.stroke()
    ctx.restore()


_DUMMY = cairo.ImageSurface(cairo.FORMAT_ARGB32, 4, 4)


def _box_path(t):
    """Actual (reach-clamped) randoseru centres along the swing, sampled by drawing Mina off-screen."""
    pts = []
    for k in range(9):
        tt = 5.02 + 0.48 * k / 8
        x, st = mina_state(tt)
        c = cairo.Context(_DUMMY)
        a = draw_mina(c, x, GY, S, **st)
        pts.append(a.get("randoseru_center"))
    return pts


def _interp_path(pts, u):
    f = clamp(u) * (len(pts) - 1)
    i = min(len(pts) - 2, int(f))
    q = f - i
    return (lerp(pts[i][0], pts[i + 1][0], q), lerp(pts[i][1], pts[i + 1][1], q))


def canopy_shadow(ctx, cam, extra=None):
    """Clear rain under the canopy (and under the held randoseru) in screen coords."""
    ctx.save()
    world._apply_cam(ctx, cam)
    # dome
    ctx.move_to(CAN_CX - CAN_R - 30, CAN_CY - 10)
    ctx.curve_to(CAN_CX - CAN_R * 0.7, CAN_CY - 260, CAN_CX - 120, CAN_CY - 310, CAN_CX, CAN_CY - 320)
    ctx.curve_to(CAN_CX + 120, CAN_CY - 310, CAN_CX + CAN_R * 0.7, CAN_CY - 260, CAN_CX + CAN_R + 30, CAN_CY - 10)
    ctx.line_to(CAN_CX + CAN_R - 20, CAN_CY + 120)
    ctx.line_to(CAN_CX + CAN_R - 90, GY + 400)
    ctx.line_to(CAN_CX - CAN_R + 10, GY + 400)
    ctx.close_path()
    if extra:
        x0, y0, x1, y1 = extra
        ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.set_operator(cairo.OPERATOR_DEST_OUT)
    ctx.set_source_rgba(0, 0, 0, 0.92)
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    ctx.fill()
    ctx.restore()


def blurred_layer(ctx, draw_fn, cam, k=4, sigma=1.6):
    """Cheap defocus: draw at 1/k resolution, scale back up bilinearly."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W // k, H // k)
    c = cairo.Context(s)
    c.scale(1 / k, 1 / k)
    world._apply_cam(c, cam)
    draw_fn(c)
    s.flush()
    a = np.ndarray((H // k, W // k, 4), np.uint8, s.get_data(), strides=(s.get_stride(), 4, 1))
    b = a.astype(np.float32)
    for ch in range(4):
        b[..., ch] = gaussian_filter(b[..., ch], sigma)
    a[...] = np.clip(b + 0.5, 0, 255).astype(np.uint8)
    s.mark_dirty()
    ctx.save()
    ctx.scale(k, k)
    ctx.set_source_surface(s, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_BILINEAR)
    ctx.paint()
    ctx.restore()


# ------------------------------------------------------------------ post
_VIG = None


def _vignette_mask():
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _VIG = (np.clip(d - 0.55, 0, 1) ** 1.5)[..., None].astype(np.float32)
    return _VIG


def _up2(a, w=W, h=H):
    out = np.empty((h, w, a.shape[2]), np.float32)
    for c in range(a.shape[2]):
        out[..., c] = np.asarray(Image.fromarray(a[..., c], mode="F").resize((w, h), Image.BILINEAR))
    return out


def fast_bloom(img, threshold=0.72, radius=36, strength=0.5, tint=None):
    half = img[::2, ::2, :3]
    lum = half.max(axis=2, keepdims=True)
    bright = (half * np.clip((lum - threshold) / (1 - threshold), 0, 1)).astype(np.float32)
    g = core.blur(bright, radius / 2, downscale=4) * strength + core.blur(bright, radius * 0.15, downscale=1) * strength * 0.6
    if tint is not None:
        g = g * np.asarray(tint, np.float32)
    g = _up2(g)
    img[..., :3] = 1 - (1 - img[..., :3]) * (1 - np.clip(g, 0, 1))
    return img


def glow_from(img, draw_fn, radius, strength, tint=None, bbox=None):
    """Additive glow of emissive parts drawn by draw_fn(ctx) (screen space), blurred at 1/2 res in a bbox."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W // 2, H // 2)
    c = cairo.Context(s)
    c.scale(0.5, 0.5)
    draw_fn(c)
    s.flush()
    a = np.ndarray((H // 2, W // 2, 4), np.uint8, s.get_data(), strides=(s.get_stride(), 4, 1))
    x0, y0, x1, y1 = bbox or (0, 0, W, H)
    pad = int(radius * 2)
    x0, y0 = max(0, int(x0) - pad) // 2, max(0, int(y0) - pad) // 2
    x1, y1 = min(W, int(x1) + pad) // 2, min(H, int(y1) + pad) // 2
    if x1 - x0 < 4 or y1 - y0 < 4:
        return img
    src = a[y0:y1, x0:x1][..., [2, 1, 0]].astype(np.float32) / 255.0
    g = np.stack([gaussian_filter(src[..., k], radius / 2) for k in range(3)], -1)
    if tint is not None:
        g *= np.asarray(tint, np.float32)
    g = np.clip(g * strength, 0, 1)
    gu = np.repeat(np.repeat(g, 2, 0), 2, 1)
    Y0, X0 = y0 * 2, x0 * 2
    Y1, X1 = Y0 + gu.shape[0], X0 + gu.shape[1]
    reg = img[Y0:Y1, X0:X1, :3]
    img[Y0:Y1, X0:X1, :3] = 1 - (1 - reg) * (1 - gu)
    return img


def warm_light(img, cx, cy, radius, amt):
    if amt <= 0.003:
        return img
    xs = ((np.arange(W, dtype=np.float32) - cx) / radius) ** 2
    ys = ((np.arange(H, dtype=np.float32) - cy) / radius) ** 2
    m = np.exp(-(ys[:, None] + xs[None, :]))[..., None] * amt
    col = np.asarray(WARM, np.float32)
    img[..., :3] = 1 - (1 - img[..., :3]) * (1 - m * col * 0.55)
    # also a gentle warm multiply grade
    img[..., :3] *= 1 + m * (col - 0.85) * 0.25
    return img


def rgb_split(img, bbox, k, t):
    """Glitch: shift R/B channels and tear some row bands inside bbox (screen px)."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return img
    reg = img[y0:y1, x0:x1, :3].copy()
    d = max(1, int(k))
    reg[:, d:, 0] = img[y0:y1, x0:x1 - d, 0]
    reg[:, :-d, 2] = img[y0:y1, x0 + d:x1, 2]
    f = int(round(t * FPS))
    hh = y1 - y0
    for b in range(3):
        if _h(f, b, 4.2) < 0.7:
            by0 = int(hh * _h(f, b, 5.1))
            bh = int(6 + 30 * _h(f, b, 6.3))
            sh = int((_h(f, b, 7.4) - 0.5) * 4 * k)
            if sh:
                reg[by0:by0 + bh] = np.roll(reg[by0:by0 + bh], sh, axis=1)
    img[y0:y1, x0:x1, :3] = reg
    return img


# ------------------------------------------------------------------ main
def render(t: float) -> np.ndarray:
    tg = T0 + t
    f = int(round(t * FPS))
    cam, shot = camera(t)
    zoom = cam[2]
    close = shot in ("A", "B")
    dof = {"A": 14, "B": 12, "C": 0, "D": 5, "E": 0}[shot]
    dens = 1.0 - 0.45 * ramp(t, 8.8, 11.0)

    rp, jit = robo_pose(t)
    mx, ms = mina_state(t)
    kx, ky, kp = kitten_state(t)

    surf, ctx = core.new_surface()
    mcam = master_cam(shot)
    ctx.save()
    plate_xform(ctx, cam, mcam)
    world.draw_shopfront(ctx, tg, WEATHER, mcam, dof=dof)
    ctx.restore()

    anchors = {}

    def draw_chars(c, reflect=False, only=None):
        if only in (None, "kitten") and shot != "A" and shot != "B":
            pass
        if only in (None, "robo"):
            c.save()
            c.translate(jit[0] * S, jit[1] * S)
            anchors["robo"] = draw_robo(c, RX, GY, S, **rp)
            c.restore()
        if only in (None, "mina"):
            anchors["mina"] = draw_mina(c, mx, GY, S, **ms)
        if only in (None, "kitten") and not close:
            anchors["kitten"] = draw_kitten(c, kx, ky, S, t=tg, wet=0.6, blink=core.blink(tg, 23), **kp)

    if shot in ("C", "E"):
        ctx.save()
        plate_xform(ctx, cam, mcam)
        world.draw_ground_reflection(ctx, tg, WEATHER,
                                     lambda c: (_with_cam(c, mcam), draw_chars(c, True), c.restore()), mcam)
        ctx.restore()

    world.draw_rain(ctx, tg, WEATHER, "bg", cam, density=dens, dof=dof / 20)

    if shot == "A":
        blurred_layer(ctx, lambda c: draw_chars(c, only="mina"), cam, k=5)
        _with_cam(ctx, cam)
        draw_chars(ctx, only="robo")
        ctx.restore()
    elif shot == "B":
        blurred_layer(ctx, lambda c: draw_chars(c, only="robo"), cam, k=4)
        _with_cam(ctx, cam)
        draw_chars(ctx, only="mina")
        ctx.restore()
    else:
        _with_cam(ctx, cam)
        draw_chars(ctx)
        ctx.restore()

    ra = anchors.get("robo")
    ma = anchors.get("mina")
    holding = t >= 5.42

    # ---- water details on the characters (world coords)
    _with_cam(ctx, cam)
    if shot != "B" and ra is not None:
        draw_screen_rivulets(ctx, ra, tg, 1.0 if t < 6.0 else lerp(1.0, 0.45, ramp(t, 6.0, 7.0)))
        if holding:
            draw_drops_on(ctx, ra["drop_points"], tg, S, seed=3, amount=0.4)
        else:
            draw_drops_on(ctx, ra["drop_points"], tg, S, seed=3, amount=1.0)
    if ma is not None and shot != "A":
        draw_drops_on(ctx, ma["drop_points"], tg, S, seed=5, amount=0.8)
        if holding and "randoseru_span" in ma:
            draw_roof_splashes(ctx, ma, tg, S, amt=dens)
            x0, x1 = ma["randoseru_span"]
            by = ma["randoseru_center"][1] + 40 * S
            world.draw_drips_from(ctx, tg, [(x0 + 6 * S, by), (x1 - 6 * S, by), (x1 - 30 * S, by + 4 * S)],
                                  ground_y=GY, size=3.2 * S, weather=WEATHER, seed=17, splash=True)
    if shot == "C":
        if 5.06 <= t < 5.55:
            draw_whoosh(ctx, t, _box_path(t), S)
    ctx.restore()

    # ---- rain in front: drawn in a group so the canopy / roof shadow can be cut out
    ctx.push_group()
    world.draw_rain(ctx, tg, WEATHER, "mid", cam, density=dens, dof=dof / 20)
    if not close:
        _with_cam(ctx, cam)
        draw_rim_drips(ctx, [p for p in ra["canopy_rim_points"]] if ra else [], tg, S, seed=9,
                       amount=0.8 * dens, fall=GY - CAN_CY - 40)
        ctx.restore()
        cr = world.to_screen(CAN_CX, CAN_CY + 20, cam)
        world.draw_canopy_rain(ctx, tg, cr, CAN_R * zoom, WEATHER, ry=CAN_RY * zoom,
                               ground_y=world.to_screen(0, GY + 10, cam)[1], intensity=0.45 * dens, hits=True)
    world.draw_rain(ctx, tg, WEATHER, "fg", cam, density=dens * (0.7 if close else 1.0), dof=dof / 14 + 0.3)
    extra = None
    if holding and ma is not None:
        x0, x1 = ma["randoseru_span"]
        extra = (x0 + 4 * S, ma["randoseru_top"][1] + 6 * S, x1 - 4 * S, GY - 140 * S)
    canopy_shadow(ctx, cam, extra)
    ctx.pop_group_to_source()
    ctx.paint()

    if shot == "A":   # rim drips falling past Robo's face from the canopy edge
        _with_cam(ctx, cam)
        rim = [p for p in ra["canopy_rim_points"] if p[0] > 820]
        draw_rim_drips(ctx, rim, tg, S, seed=4, amount=0.9, fall=GY - CAN_CY)
        ctx.restore()

    world.draw_shopfront_fg(ctx, tg, WEATHER, cam)

    # ---------------------------------------------------------------- post
    img = core.surface_rgba(surf)
    g = glitch_amt(t) if shot == "A" else (glitch_amt(t) * 0.6 if shot in ("C", "D") else 0.0)
    if ra is not None:
        sx, sy = world.to_screen(*ra["screen_center"], cam)
        sw, sh = ra["screen_size"][0] * zoom, ra["screen_size"][1] * zoom
        # emissive glow (LEDs, bulb, sparks)
        rp_g = dict(rp)
        warm_phase = clamp((t - 7.55) / 0.5)

        def gfn(c):
            world._apply_cam(c, cam)
            c.translate(jit[0] * S, jit[1] * S)
            draw_robo(c, RX, GY, S, glow_only=True, **{k: v for k, v in rp_g.items() if k != "canopy"},
                      canopy=0.0)

        bb = (sx - sw * 2.5, sy - sh * 3.2, sx + sw * 2.5, sy + sh * 4.5)
        if shot in ("A", "D") or (shot == "E") or (shot == "C"):
            glow_from(img, gfn, 10 * zoom * S, 1.1 + 0.6 * warm_phase,
                      tint=core.mix((1, 1, 1), WARM, warm_phase) if warm_phase > 0 else None, bbox=bb)
        if g > 0.25:
            rgb_split(img, (sx - sw * 1.0, sy - sh * 0.9, sx + sw * 1.0, sy + sh * 0.9), 3 + 9 * g, t)
        # warm glow spreading from Robo's screen + body hum
        if t >= 7.6:
            spread = ease_out(clamp((t - 7.6) / 1.6), 2)
            hum = 1 + 0.06 * math.sin(tg * 7) * ramp(t, 8.4, 9.0)
            amt = (0.32 + 0.28 * spread) * hum
            img = warm_light(img, sx, sy + sh * 0.6, (170 + 360 * spread) * zoom * 0.8, amt)
        # sparks flash: brief global bloom kick
    flash = 0.0
    if t < 6.0 and g > 0.7 and shot == "A":
        flash = 0.15
    warm_k = ramp(t, 7.6, 9.0)
    tint = core.mix((0.9, 0.95, 1.05), (1.08, 0.92, 0.72), warm_k)
    fast_bloom(img, 0.72 - 0.05 * warm_k, 36, 0.5 + 0.12 * warm_k + flash, tint=tint)
    v = _vignette_mask() * (0.34 if not close else 0.42)
    img[..., :3] = img[..., :3] * (1 - v) + np.array((0.02, 0.02, 0.08), np.float32) * v
    core.grain(img, f, 0.016)
    return core.finish(img)
