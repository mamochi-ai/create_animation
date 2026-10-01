"""Character rigs for 「あめふりロボ」: Mina, Robo (No.7) and the calico kitten.

Flat cel-shaded vector art drawn procedurally with pycairo: one shadow tone, a
cool rain rim-light, coloured outlines.  Deterministic for a given set of
arguments (randomness is seeded by `t`/`seed`).  Every draw function honours
the current cairo transform and returns anchors in the caller's user space.

Public API (full option lists in the docstrings / docs/characters_api.md)
--------------------------------------------------------------------------
draw_mina(ctx, x, y, scale=1.0, **pose)   -> anchors      (scale 1 ~ 480 px tall)
draw_robo(ctx, x, y, scale=1.0, **pose)   -> anchors      (scale 1 ~ 410 px tall)
robo_glow_layer(x, y, scale=1.0, **pose)  -> RGBA float layer of the LEDs (for core.add_glow)
draw_kitten(ctx, x, y, scale=1.0, **pose) -> anchors      (scale 1 ~ 70 px tall)
draw_droplet(ctx, x, y, r, alpha=1.0)     single shiny water drop
draw_drops_on(ctx, points, t, scale=1.0, seed=0, amount=1.0)  drops beading/running/falling off surfaces
draw_rim_drips(ctx, rim_points, t, scale=1.0, seed=0, amount=1.0)   drips falling from an umbrella rim
draw_sparkle(ctx, x, y, size, alpha, rot, color)  4-point twinkle (re-exported from film 1)
"""
from __future__ import annotations

import math

import cairo

from common.core import (TAU, clamp, ease_in_out, ease_out, ease_out_back, lerp, mix, noise1, rgb,
                         smoothstep)
from common.characters import _cel, _ik, _limb, _sample_spline, _spline, _taper, draw_sparkle  # noqa: F401

PI = math.pi

# =============================================================== palettes
RIM = "#D6EAF8"          # cool overcast rim light (rainy day)

# Mina
M_HAIR, M_HAIR_SH, M_HAIR_HI, M_HAIR_LINE = "#5A3A2A", "#3F271C", "#8C624A", "#26140C"
M_SKIN, M_SKIN_SH, M_SKIN_LINE = "#FFE6D6", "#F4C1AA", "#B9705E"
M_CHEEK = "#FF8F95"
M_HAT, M_HAT_SH, M_HAT_HI, M_HAT_LINE = "#F7C934", "#D99F1C", "#FFEB9C", "#8A5810"
M_BLOUSE, M_BLOUSE_SH, M_BLOUSE_LINE = "#FDFDF8", "#CFD8E6", "#7584A2"
M_NAVY, M_NAVY_SH, M_NAVY_HI, M_NAVY_LINE = "#2E3F73", "#1F2B54", "#5D74B4", "#111936"
M_SOCK, M_SOCK_SH = "#FFFFFF", "#D3DAE6"
M_SHOE, M_SHOE_SH, M_SHOE_LINE = "#D9483B", "#A72F28", "#5A1410"
M_RAN, M_RAN_SH, M_RAN_HI, M_RAN_LINE = "#C8323A", "#93202B", "#F2717A", "#4E0C14"
M_RIBBON, M_RIBBON_SH, M_RIBBON_LINE = "#FFD43B", "#E2A517", "#8C5A0E"
M_GOLD, M_GOLD_SH = "#F3D27A", "#B98C2E"
M_IRIS = ("#2B140B", "#6E3B1E", "#C98B4F")
M_LASH = "#2A130B"
M_MOUTH_IN, M_TONGUE, M_MOUTH_LINE = "#7A2630", "#F08592", "#7E3036"

# Robo
R_MINT, R_MINT_SH, R_MINT_HI, R_MINT_LINE = "#8FD3C1", "#5FA79A", "#CFF3E8", "#2C5A54"
R_CREAM, R_CREAM_SH, R_CREAM_LINE = "#F3EBD3", "#D3C6A2", "#7C6E50"
R_RUST, R_RUST_DK = "#B8693A", "#7E4122"
R_GLASS, R_GLASS_HI = "#1C2A33", "#3C5664"
R_LED, R_LED_WARM, R_LED_PINK = "#6FF2FF", "#FFB45E", "#FF8FA6"
R_METAL, R_METAL_SH, R_METAL_HI, R_METAL_LINE = "#B7C2C9", "#7D8B96", "#E6EEF2", "#3A4650"
R_TREAD, R_TREAD_SH, R_TREAD_HI, R_TREAD_LINE = "#3E444B", "#2A2E34", "#5E666F", "#121519"
R_BULB, R_BULB_OFF = "#FF4A3D", "#8A2A26"
R_STENCIL = "#A0482A"
CAN_TEAL, CAN_TEAL_SH, CAN_CREAM, CAN_CREAM_SH = "#2FA39A", "#1E7A73", "#F6EDD5", "#D2C4A0"
CAN_LINE = "#124A45"
UMB_RED, UMB_RED_SH, UMB_LINE = "#E8413C", "#B32A28", "#5A1210"

# Kitten
K_WHITE, K_WHITE_SH = "#FFFDF8", "#DCD6CF"
K_ORANGE, K_ORANGE_SH = "#F09A44", "#C8742A"
K_BLACK, K_BLACK_SH = "#3A3330", "#241F1D"
K_LINE, K_PINK, K_EYE = "#5E4638", "#FFAFB8", "#2A2420"

_LW = [2.5]


def line_weight(scale: float, base: float = 2.5) -> float:
    """Outline width in local units for a rig drawn at `scale` (grows sub-linearly)."""
    return base * max(scale, 0.05) ** -0.33


# =============================================================== small helpers
def _c(c):
    return rgb(c) if isinstance(c, str) else c


def _sc(ctx, c, a=1.0):
    c = _c(c)
    ctx.set_source_rgba(c[0], c[1], c[2], a)


def _wet(c, w, k=1.0):
    """Darker, slightly bluer, more saturated 'soaked' version of colour c."""
    c = _c(c)
    if w <= 0:
        return c
    d = (c[0] * 0.70, c[1] * 0.75, c[2] * 0.84)
    return mix(c, d, clamp(w * k))


def _rot(p, a):
    ca, sa = math.cos(a), math.sin(a)
    return (p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca)


def _add(a, b, k=1.0):
    return (a[0] + b[0] * k, a[1] + b[1] * k)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _len(v):
    return math.hypot(v[0], v[1])


def _norm(v):
    l = math.hypot(v[0], v[1]) or 1e-9
    return (v[0] / l, v[1] / l)


def _l2(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _ell(ctx, cx, cy, rx, ry, rot=0.0):
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(rot)
    ctx.scale(max(rx, 1e-3), max(ry, 1e-3))
    ctx.arc(0, 0, 1, 0, TAU)
    ctx.restore()


def _rrect(ctx, x, y, w, h, r):
    r = max(0.0, min(r, w / 2, h / 2))
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -PI / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, PI / 2)
    ctx.arc(x + r, y + h - r, r, PI / 2, PI)
    ctx.arc(x + r, y + r, r, PI, 1.5 * PI)
    ctx.close_path()


def _rpoly(ctx, pts, r):
    """Polygon with rounded corners (radius r, clamped to half edge length)."""
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        d0 = _norm(_sub(p0, p1))
        d2 = _norm(_sub(p2, p1))
        rr = min(r, _len(_sub(p0, p1)) * 0.45, _len(_sub(p2, p1)) * 0.45)
        a = _add(p1, d0, rr)
        b = _add(p1, d2, rr)
        if i == 0:
            ctx.move_to(*a)
        else:
            ctx.line_to(*a)
        ctx.curve_to(lerp(a[0], p1[0], 0.6), lerp(a[1], p1[1], 0.6),
                     lerp(b[0], p1[0], 0.6), lerp(b[1], p1[1], 0.6), *b)
    ctx.close_path()


def _hull(points):
    pts = sorted(set((round(p[0], 2), round(p[1], 2)) for p in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _poly(ctx, pts):
    ctx.move_to(*pts[0][:2])
    for p in pts[1:]:
        ctx.line_to(*p[:2])
    ctx.close_path()


def _stroke(ctx, col, lw, a=1.0):
    _sc(ctx, col, a)
    ctx.set_line_width(lw)
    ctx.stroke()


def _hash(*a):
    n = 0x9E3779B1
    for v in a:
        n = (n ^ (int(v) & 0xFFFFFFFF)) * 0x85EBCA6B & 0xFFFFFFFF
        n ^= n >> 13
    return (n & 0xFFFFFF) / float(0xFFFFFF)


def draw_droplet(ctx, x, y, r, alpha=1.0, stretch=1.0):
    """A small shiny water drop (pointed top) centred at (x, y)."""
    if r <= 0.2 or alpha <= 0:
        return
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(r, r * stretch)
    ctx.move_to(0, -1.7)
    ctx.curve_to(0.35, -1.0, 1.0, -0.4, 1.0, 0.25)
    ctx.curve_to(1.0, 0.85, 0.5, 1.0, 0, 1.0)
    ctx.curve_to(-0.5, 1.0, -1.0, 0.85, -1.0, 0.25)
    ctx.curve_to(-1.0, -0.4, -0.35, -1.0, 0, -1.7)
    ctx.close_path()
    ctx.restore()
    g = cairo.LinearGradient(x, y - r * 1.7, x, y + r)
    g.add_color_stop_rgba(0, 0.86, 0.93, 1.0, 0.25 * alpha)
    g.add_color_stop_rgba(1, 0.70, 0.84, 0.96, 0.75 * alpha)
    ctx.set_source(g)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.30, 0.42, 0.55, 0.55 * alpha)
    ctx.set_line_width(max(0.6, r * 0.18))
    ctx.stroke()
    ctx.arc(x - r * 0.35, y - r * 0.05, r * 0.28, 0, TAU)
    ctx.set_source_rgba(1, 1, 1, 0.95 * alpha)
    ctx.fill()


def _anchor_fn(ctx):
    M0 = ctx.get_matrix()
    inv = cairo.Matrix(*M0)
    inv.invert()

    def anchor(px, py):
        dx, dy = ctx.user_to_device(px, py)
        return inv.transform_point(dx, dy)

    def to_local(px, py):
        dx, dy = M0.transform_point(px, py)
        return ctx.device_to_user(dx, dy)

    return anchor, to_local


# =============================================================== 3D-lite projection
class _P:
    """Body space (l = lateral, + = character's left; f = forward; v = screen-down)
    projected for yaw `ya` (0 front, +pi/2 facing screen-right) with a small camera tilt."""

    def __init__(self, ya, tilt=0.10):
        self.ya = ya
        self.c = math.cos(ya)
        self.s = math.sin(ya)
        self.tilt = tilt

    def p(self, l, f, v):
        return (l * self.c + f * self.s, v + self.tilt * (f * self.c - l * self.s))

    def z(self, l, f):
        return f * self.c - l * self.s

    def p3(self, q):
        return self.p(q[0], q[1], q[2])


_VIEW_YAW = {"front": 0.0, "three_quarter_right": 0.9, "three_quarter_left": -0.9,
             "side_right": PI / 2, "side_left": -PI / 2, "back": PI}


def _resolve_view(view, yaw):
    ya = _VIEW_YAW.get(view, 0.0) if yaw is None else float(yaw)
    ya = (ya + PI) % TAU - PI if abs(ya) > PI + 1e-6 else ya
    mirror = 1
    if -PI + 1e-6 < ya < -1e-6:
        mirror = -1
        ya = -ya
    if abs(ya) >= PI - 1e-6:
        ya = PI
    return ya, mirror


def _ring_ext(P, w, d, f0=0.0):
    """Screen x-centre and half extent of an elliptical body cross-section."""
    return f0 * P.s, math.hypot(w * P.c, d * P.s)


def _interp_rings(rings, v):
    if v <= rings[0][0]:
        return rings[0][1:]
    for a, b in zip(rings, rings[1:]):
        if v <= b[0]:
            u = (v - a[0]) / (b[0] - a[0])
            return tuple(lerp(a[i], b[i], u) for i in range(1, len(a)))
    return rings[-1][1:]


# =============================================================== Mina
_M_EXPR = {
    #            brow_y brow_tilt open lid_tilt lower iris  mouth          smile blush
    "neutral":   (0.00, 0.00, 1.00, 0.00, 0.05, 1.00, "smile_s", 0.40, 0.55),
    "happy":     (0.25, 0.10, 0.92, 0.00, 0.40, 1.00, "smile_open", 1.00, 0.95),
    "laugh":     (0.35, 0.15, 0.00, 0.00, 0.00, 1.00, "laugh", 1.00, 1.00),
    "sad":       (0.15, 1.00, 0.84, 0.35, 0.12, 1.00, "frown", -0.8, 0.50),
    "surprised": (1.00, 0.20, 1.14, 0.00, 0.00, 0.74, "o", 0.00, 0.55),
    "worried":   (0.40, 1.00, 0.96, 0.22, 0.05, 0.92, "wavy", -0.3, 0.55),
    "pout":      (-0.35, -1.00, 0.78, -0.50, 0.30, 1.00, "pout", 0.00, 1.00),
    "amazed":    (0.85, 0.25, 1.12, 0.00, 0.10, 1.10, "amazed", 0.80, 0.90),
    "gentle":    (0.12, 0.30, 0.74, 0.10, 0.36, 1.00, "smile", 0.70, 0.95),
}

_M_VOWEL = {  # half-width, height, roundness, teeth
    "a": (8.0, 11.0, 0.55, 0.35), "i": (8.8, 4.8, 0.25, 0.6), "u": (4.6, 6.0, 1.0, 0.0),
    "e": (7.8, 7.4, 0.4, 0.45), "o": (6.0, 9.6, 0.95, 0.0), "n": (4.6, 1.3, 0.5, 0.0),
    "x": (5.0, 0.0, 0.5, 0.0),
}

M_R = 56.0                 # head radius
M_HEAD_V = -380.0          # head centre (standing), body space
M_HIP_V = -182.0
M_SHOULDER = (31.0, -303.0)
M_UPPER, M_FORE = 62.0, 58.0
M_THIGH, M_SHIN = 84.0, 82.0
M_ANKLE_V = -16.0

_M_TORSO = [(-336, 8.5, 8, 0), (-320, 11, 10, 0), (-313, 23, 15, 0), (-305, 32, 18, 0), (-290, 33.5, 19, 1.0),
            (-272, 31.5, 19.5, 2.0), (-256, 28.5, 18.5, 1.0), (-246, 28.5, 18.5, 0.0)]
_M_SKIRT = [(-256, 30.5, 20, 0.5), (-236, 40, 27, 0.5), (-212, 51, 35, 0.0), (-188, 59, 42, 0.0),
            (-166, 64, 46, 0.0)]


class _MinaHead:
    def __init__(self, ya, pitch, expr, t, blink, look, mouth_open, vowel, over, rim, ls, wet, wind, lw):
        self.ya, self.pitch, self.t = ya, pitch, t
        self.blink, self.look = blink, look
        self.mouth_open, self.vowel = mouth_open, vowel
        self.rim, self.ls, self.wet, self.wind, self.lw = rim, ls, wet, wind, lw
        self.back = math.cos(ya) < -0.45
        self.expr = expr
        e = _M_EXPR.get(expr, _M_EXPR["neutral"])
        (self.brow_y, self.brow_tilt, self.eye_open, self.lid_tilt, self.lower,
         self.iris, self.mouth_kind, self.smile, self.blush) = e
        for k, v in over.items():
            if v is not None:
                setattr(self, k, v)
        self.hair = _wet(M_HAIR, wet, 0.8)
        self.hair_sh = _wet(M_HAIR_SH, wet, 0.8)
        self.eye_pos = []
        self.mouth_xy = (0.0, 0.62 * M_R)
        self.drop_pts = []

    def sp(self, phi, y, rad=1.0):
        R = M_R
        ry = R * rad * math.sqrt(max(0.0, 1 - (y / (1.36 * R)) ** 2))
        a = phi + self.ya
        z = math.cos(a)
        return (ry * math.sin(a), y - self.pitch * 0.2 * R * z), z

    def sway(self, ph=0.0):
        t = self.t
        return self.wind * 6.0 * (0.6 + 0.4 * math.sin(t * 3.0 + ph)) + 0.9 * math.sin(t * 1.4 + ph * 1.7)

    # ------------------------------------------------ silhouettes
    def face_pts(self):
        R = M_R
        F = [(0, -1.0), (0.6, -0.88), (0.93, -0.5), (1.0, -0.05), (1.0, 0.2), (0.97, 0.42), (0.86, 0.66),
             (0.6, 0.88), (0.0, 0.99), (-0.6, 0.88), (-0.86, 0.66), (-0.97, 0.42), (-1.0, 0.2), (-1.0, -0.05),
             (-0.93, -0.5), (-0.6, -0.88)]
        S = [(0.05, -1.0), (0.62, -0.88), (0.88, -0.5), (0.94, -0.1), (1.05, 0.26), (0.96, 0.42), (0.95, 0.6),
             (0.8, 0.86), (0.45, 0.93), (0.1, 0.76), (-0.3, 0.56), (-0.7, 0.42), (-0.9, 0.2), (-0.98, 0.0),
             (-0.95, -0.42), (-0.62, -0.86)]
        w = abs(math.sin(self.ya)) ** 1.3
        pts = []
        puff = 0.15 if self.expr == "pout" else 0.0
        for i, (a, b) in enumerate(zip(F, S)):
            x = lerp(a[0], b[0], w)
            yy = lerp(a[1], b[1], w)
            if puff and i in (4, 5, 6, 10, 11, 12):
                k = 1.0 if i in (5, 11) else 0.6
                x += math.copysign(puff * k, x)
            pts.append((x * R, yy * R + (self.pitch * 0.1 * R if yy > 0 else 0)))
        if w > 0.75:
            pts[4] = (pts[4][0], pts[4][1], True)
        return pts

    def hair_pts(self):
        R = M_R
        FH = [(0, -1.2), (0.72, -1.02), (1.1, -0.55), (1.17, 0.0), (1.1, 0.38), (0.92, 0.6), (0.6, 0.52),
              (0.0, 0.42), (-0.6, 0.52), (-0.92, 0.6), (-1.1, 0.38), (-1.17, 0.0), (-1.1, -0.55), (-0.72, -1.02)]
        SH = [(0.1, -1.2), (0.72, -1.0), (0.98, -0.58), (0.75, -0.3), (0.3, -0.22), (0.12, 0.25), (-0.1, 0.56),
              (-0.45, 0.72), (-0.8, 0.63), (-1.05, 0.36), (-1.17, 0.0), (-1.12, -0.55), (-0.85, -0.95),
              (-0.4, -1.17)]
        BH = [(0, -1.22), (0.72, -1.04), (1.1, -0.55), (1.17, 0.0), (1.1, 0.4), (0.9, 0.62), (0.5, 0.72),
              (0.0, 0.76), (-0.5, 0.72), (-0.9, 0.62), (-1.1, 0.4), (-1.17, 0.0), (-1.1, -0.55), (-0.72, -1.04)]
        if self.back:
            A, B, w = BH, BH, 0.0
        else:
            A, B, w = FH, SH, abs(math.sin(self.ya)) ** 1.1
        pts = []
        for a, b in zip(A, B):
            pts.append((lerp(a[0], b[0], w) * R, lerp(a[1], b[1], w) * R))
        return pts

    # ------------------------------------------------ pigtails
    def pigtail_geom(self, side):
        """side +1 = her left, -1 = her right.  Returns (z, points, base, dir)."""
        R = M_R
        (bx, by), z = self.sp(side * 2.0, 0.36 * R, 1.0)
        c, s, ya = math.cos(self.ya), math.sin(self.ya), self.ya
        wet = self.wet
        dl, df, dv = side * 0.8, -0.42, 0.62 + 0.35 * wet
        dx = dl * c + df * s
        dy = dv + 0.1 * (df * c - dl * s)
        d = _norm((dx, dy))
        if _len((dx, dy)) < 0.5:  # mostly toward/away from camera -> point down-back
            d = _norm((d[0] + (-0.4 if s > 0 else 0.0), d[1] + 0.4))
        L = 0.92 * R * (1 - 0.1 * wet)
        t = self.t
        sw = 0.16 * math.sin(t * 2.6 + side * 1.3) + 0.22 * self.wind * (0.7 + 0.3 * math.sin(t * 4.1 + side))
        sw2 = 0.16 * math.sin(t * 2.6 + side * 1.3 - 1.0) + 0.3 * self.wind * (0.7 + 0.3 * math.sin(t * 4.1 + side - 0.9))
        d1 = _rot(d, sw)
        d2 = _rot(d, sw + sw2 * 1.4 + side * 0.0)
        M = _add((bx, by), d1, L * 0.5)
        T = _add(M, d2, L * 0.55)
        n1 = (-d1[1], d1[0])
        n2 = (-d2[1], d2[0])
        w0, w1 = 0.2 * R, 0.33 * R * (1 - 0.3 * wet)
        tip_w = 0.2 * R * (1 - 0.3 * wet)
        pts = [_add((bx, by), n1, w0), _add(M, n1, w1),
               (*_add(_add(T, n2, tip_w * 0.95), d2, -L * 0.12), True),
               _add(_add(T, n2, tip_w * 0.3), d2, -L * 0.2),
               (*T, True),
               _add(_add(T, n2, -tip_w * 0.4), d2, -L * 0.22),
               (*_add(_add(T, n2, -tip_w * 1.05), d2, -L * 0.18), True),
               _add(M, n1, -w1 * 0.95), _add((bx, by), n1, -w0)]
        if wet > 0.3:  # clumped wet strands: extra pointed tips
            pts.insert(3, (*_add(_add(T, n2, tip_w * 0.6), d2, -L * 0.02), True))
        return z, pts, (bx, by), d1, M, T, n1

    def draw_pigtail(self, ctx, side):
        z, pts, B, d1, M, T, n1 = self.pigtail_geom(side)
        lw = self.lw

        def extra(c):
            _sc(c, self.hair_sh, 0.9)
            _taper(c, [_add(B, n1, 1.5), _add(M, n1, 2.5), _l2(M, T, 0.7)], 1.0, 0.3, 2.4)
            _taper(c, [_add(B, n1, -4), _add(M, n1, -6), _l2(M, T, 0.6)], 0.8, 0.3, 1.8)
            _sc(c, M_HAIR_HI, 0.45 * (1 - 0.5 * self.wet))
            _taper(c, [_add(_l2(B, M, 0.3), n1, 5), _add(M, n1, 6)], 0.5, 0.5, 3.0)

        _cel(ctx, lambda c: _spline(c, pts), self.hair, self.hair_sh, self.rim, M_HAIR_LINE, lw,
             soff=(-4 * self.ls, -4), roff=(-2.2 * self.ls, 0.4), rim_a=0.5, extra=extra)
        if self.wet > 0.2:
            draw_droplet(ctx, T[0], T[1] + 5, 2.6, self.wet)

    def draw_ribbon(self, ctx, side):
        z, pts, B, d1, M, T, n1 = self.pigtail_geom(side)
        lw = self.lw * 0.85
        ang = math.atan2(d1[1], d1[0]) - PI / 2
        rib = _wet(M_RIBBON, self.wet, 0.4)
        ctx.save()
        ctx.translate(*_add(B, d1, 3))
        ctx.rotate(ang + side * 0.15)
        fs = 0.55 + 0.45 * abs(z) if not self.back else 1.0
        ctx.scale(1.0, 1.0)
        tails = [[(-2, 2), (-9, 14), (-5, 17, True), (-1, 12), (1, 4)],
                 [(2, 2), (8, 13), (11, 9, True), (6, 7), (1, 3)]]
        for tl in tails:
            _cel(ctx, lambda c, tl=tl: _spline(c, tl), rib, M_RIBBON_SH, None, M_RIBBON_LINE, lw,
                 soff=(1.5, -1.5), shade_a=0.9)
        for sgn in (-1, 1):
            loop = [(0, 0), (sgn * 7 * fs + sgn * 3, -9), (sgn * 15 * fs, -5), (sgn * 14 * fs, 3), (sgn * 6, 3)]
            _cel(ctx, lambda c, lp=loop: _spline(c, lp), rib, M_RIBBON_SH, "#FFF3B0", M_RIBBON_LINE, lw,
                 soff=(sgn * 2.5, 2.5), roff=(0, 1.2), rim_a=0.9)
        _ell(ctx, 0, -1, 4.2, 4.6)
        _sc(ctx, rib)
        ctx.fill_preserve()
        _stroke(ctx, M_RIBBON_LINE, lw)
        ctx.restore()

    # ------------------------------------------------ main
    def draw(self, ctx):
        lw = self.lw
        R = M_R
        hp = self.hair_pts()
        pig = {s: self.pigtail_geom(s)[0] for s in (-1, 1)}
        behind = [s for s in (-1, 1) if self.back is False and pig[s] < -0.05]
        front = [s for s in (-1, 1) if s not in behind]
        if self.back:
            behind, front = [], [-1, 1]
        behind.sort(key=lambda s: pig[s])
        front.sort(key=lambda s: pig[s])
        for s in behind:
            self.draw_pigtail(ctx, s)
        if self.back:
            self._draw_back(ctx, hp)
            for s in front:
                self.draw_pigtail(ctx, s)
                self.draw_ribbon(ctx, s)
            self._hat(ctx)
            return
        # back hair
        _cel(ctx, lambda c: _spline(c, hp), self.hair, self.hair_sh, self.rim, M_HAIR_LINE, lw,
             soff=(-7 * self.ls, -5), roff=(-2.6 * self.ls, 0.5), rim_a=0.5)
        for s in behind:
            self.draw_ribbon(ctx, s)
        fp = self.face_pts()

        def face_extra(c):
            c.save()
            c.translate(1.5 * self.ls, 6.0)
            self._fringe_path(c)
            _sc(c, M_SKIN_SH, 0.85)
            c.fill()
            c.restore()
            self._blush(c)

        _cel(ctx, lambda c: _spline(c, fp), M_SKIN, M_SKIN_SH, None, M_SKIN_LINE, lw * 0.9,
             soff=(-6 * self.ls, -3.5), shade_a=0.5, extra=face_extra)
        self._eyes(ctx)
        self._nose(ctx)
        self._mouth(ctx)
        self._side_hair(ctx, hp)
        self._ear(ctx)
        self._fringe(ctx, hp)
        self._side_lock(ctx, -1)
        self._side_lock(ctx, 1)
        for s in front:
            self.draw_pigtail(ctx, s)
            self.draw_ribbon(ctx, s)
        self._hat(ctx)
        self._brows(ctx)
        if self.expr == "pout":
            self._puff_marks(ctx)
        if self.expr == "amazed":
            self._amazed_sparkles(ctx)

    # ------------------------------------------------ back view
    def _draw_back(self, ctx, hp):
        R = M_R
        lw = self.lw

        def extra(c):
            _sc(c, self.hair_sh, 1.0)
            # centre part and strands sweeping toward each pigtail
            _taper(c, [(0, -1.2 * R), (0.5, -0.4 * R), (0, 0.72 * R)], 1.6, 0.8, 1.6)
            for side in (-1, 1):
                for k, y0 in enumerate((-0.9, -0.55, -0.2, 0.15, 0.45)):
                    p0 = (side * 0.12 * R, y0 * R)
                    p2 = (side * 0.95 * R, 0.38 * R)
                    p1 = (side * 0.55 * R, lerp(y0, 0.4, 0.4) * R)
                    _taper(c, [p0, p1, p2], 0.4, 0.4, 2.2)
            _sc(c, M_HAIR_HI, 0.4 * (1 - 0.5 * self.wet))
            for side in (-1, 1):
                _taper(c, [(side * 0.3 * R, -0.75 * R), (side * 0.65 * R, -0.5 * R), (side * 0.85 * R, -0.2 * R)],
                       0.6, 0.6, 4.5)

        _cel(ctx, lambda c: _spline(c, hp), self.hair, self.hair_sh, self.rim, M_HAIR_LINE, lw,
             soff=(-8 * self.ls, -6), roff=(-2.6 * self.ls, 0.5), rim_a=0.5, extra=extra)

    def _hairline(self):
        R = M_R
        pts = []
        for i in range(12):
            y = -1.5 * R + 2.3 * R * i / 11
            yy = clamp(y, -1.2 * R, 1.15 * R)
            ph = -1.2 - 0.45 * clamp((y / R - 0.05) / 0.5, 0, 1)
            (x, _), z = self.sp(ph, yy * 0.95, 1.02)
            pts.append((x, y))
        return pts

    def _side_hair(self, ctx, hp):
        if math.sin(self.ya) < 0.25:
            return
        R = M_R
        hl = self._hairline()
        ctx.save()
        ctx.new_path()
        _spline(ctx, hp)
        ctx.clip()
        ctx.new_path()
        ctx.move_to(-4 * R, hl[0][1])
        for p in hl:
            ctx.line_to(*p)
        ctx.line_to(-4 * R, hl[-1][1])
        ctx.close_path()
        ctx.clip()

        def extra(c):
            _sc(c, self.hair_sh, 1.0)
            for off in (0.3, 0.65, 1.0):
                x0 = hl[4][0] - off * R
                _taper(c, [(x0 + 0.15 * R, -0.95 * R), (x0, -0.1 * R), (x0 + 0.25 * R, 0.62 * R)], 0.4, 0.3, 2.4)
            _sc(c, M_HAIR_HI, 0.4 * (1 - 0.5 * self.wet))
            _taper(c, [(hl[3][0] - 0.25 * R, -0.75 * R), (hl[3][0] - 0.75 * R, -0.62 * R),
                       (hl[3][0] - 1.0 * R, -0.3 * R)], 0.6, 0.6, 4.5)

        _cel(ctx, lambda c: _spline(c, hp), self.hair, self.hair_sh, self.rim, None, None,
             soff=(-7 * self.ls, -5), roff=(-2.6 * self.ls, 0.5), rim_a=0.5, extra=extra)
        ctx.restore()
        ctx.save()
        ctx.new_path()
        _spline(ctx, hp)
        ctx.clip()
        ctx.new_path()
        _spline(ctx, hl, False)
        _stroke(ctx, M_HAIR_LINE, self.lw)
        ctx.restore()

    def _ear(self, ctx):
        R = M_R
        if math.sin(self.ya) < 0.3:
            return
        (x, y), z = self.sp(-1.62, 0.2 * R, 1.0)
        if z < 0.1:
            return
        ctx.save()
        ctx.translate(x, y)
        fs = 0.5 + 0.5 * math.sin(self.ya)
        ctx.scale(fs, 1)
        pts = [(4, -11), (-5, -10), (-9, 0), (-6, 10), (3, 11)]
        _cel(ctx, lambda c: _spline(c, pts, True, 0.9), M_SKIN, M_SKIN_SH, None, M_SKIN_LINE, self.lw * 0.8,
             soff=(3, 0), shade_a=0.7)
        _sc(ctx, M_SKIN_LINE, 0.6)
        _taper(ctx, [(0, -6), (-4, 0), (-1, 6)], 0.8, 0.5, 1.5)
        ctx.restore()

    # ------------------------------------------------ fringe (side-swept, parted on her left)
    _FRINGE = [(-1.62, 0.42, 1), (-1.42, -0.05, 0), (-1.2, 0.04, 1), (-1.0, -0.26, 0), (-0.74, -0.08, 1),
               (-0.52, -0.33, 0), (-0.24, -0.13, 1), (0.02, -0.36, 0), (0.24, -0.2, 1), (0.48, -0.46, 0),
               (0.7, -0.24, 1), (0.92, -0.36, 0), (1.18, -0.02, 1), (1.4, -0.12, 0), (1.62, 0.42, 1)]

    def _fringe_line(self):
        R = M_R
        out = []
        sw = self.sway(0.3) * 0.25
        wet = self.wet
        for i, (phi, y, tip) in enumerate(self._FRINGE):
            yy = y * R
            if tip:
                yy += 0.03 * R * wet
                # tips swept toward her right (screen-left in front view): shift phi
                phi2 = phi - 0.14
            else:
                phi2 = phi
            (x, yp), z = self.sp(phi2, yy, 1.06)
            out.append((x + sw * (1 if tip else 0.4), yp, z, bool(tip)))
        return out

    def _fringe_path(self, c):
        R = M_R
        line = [p for p in self._fringe_line() if p[2] > -0.3]
        if len(line) < 2:
            return
        pts = [(p[0], p[1], p[3]) for p in line]
        xr, yr = pts[-1][0], pts[-1][1]
        xl, yl = pts[0][0], pts[0][1]
        top = [(xr + 4, yr - 0.6 * R), (0.1 * R - 0.25 * R * math.sin(self.ya), -1.45 * R), (xl - 4, yl - 0.6 * R)]
        _spline(c, pts + top, True, 0.85)

    def _fringe(self, ctx, hp):
        R = M_R
        lw = self.lw

        def extra(c):
            line = self._fringe_line()
            _sc(c, self.hair_sh, 1.0)
            for i, p in enumerate(line):
                if p[3] or p[2] < 0.05 or i in (0, len(line) - 1):
                    continue
                top = (p[0] * 0.6 + 0.12 * R, p[1] - 0.7 * R)
                mid = (lerp(p[0], top[0], 0.35), lerp(p[1], top[1], 0.5))
                _taper(c, [(p[0], p[1] + 1), mid, top], 2.6, 0.2)
            # glossy band
            _sc(c, M_HAIR_HI, 0.5 * (1 - 0.4 * self.wet))
            up, lo = [], []
            for j in range(11):
                phi = -1.1 + 2.2 * j / 10
                (x, y), z = self.sp(phi, -0.62 * R, 1.08)
                if z < 0.1:
                    continue
                up.append((x, y - 0.02 * R))
                lo.append((x, y + (0.08 if j % 2 else 0.03) * R))
            if len(up) > 2:
                c.new_path()
                _spline(c, up, False)
                for p in reversed(lo):
                    c.line_to(*p)
                c.close_path()
                c.fill()

        _cel(ctx, self._fringe_path, self.hair, self.hair_sh, self.rim, None, lw, soff=(-5 * self.ls, -6),
             roff=(-2.4 * self.ls, 0.5), rim_a=0.5, extra=extra)
        line = [p for p in self._fringe_line() if p[2] > -0.3]
        ctx.new_path()
        _spline(ctx, [(p[0], p[1], p[3]) for p in line], False, 0.85)
        _stroke(ctx, M_HAIR_LINE, lw)
        if self.wet > 0.25:
            for i, p in enumerate(line):
                if p[3] and p[2] > 0.3 and i % 4 == 2:
                    draw_droplet(ctx, p[0], p[1] + 4 + 3 * math.sin(self.t * 2 + i), 2.2, self.wet * 0.9)

    def _side_lock(self, ctx, side):
        R = M_R
        phi = side * 1.4
        (rx, ry), z = self.sp(phi, -0.3 * R, 1.08)
        if z < 0.02:
            return
        (mx, my), _ = self.sp(side * 1.36, 0.3 * R, 1.1)
        (tx, ty), _ = self.sp(side * 1.2, 0.72 * R, 1.04)
        sw = self.sway(side * 2.0) * 0.6
        tx += sw
        mx += sw * 0.5
        ww = 0.22 * R * (0.4 + 0.6 * max(z, 0.15)) * (1 - 0.3 * self.wet)
        o = 1 if math.sin(phi + self.ya) > 0 else -1
        pts = [(rx - o * ww * 0.4, ry - 0.3 * R), (mx - o * ww * 0.7, my), (tx - o * 1.5, ty, True),
               (mx + o * ww * 0.9, my + 0.05 * R), (rx + o * ww * 1.1, ry)]

        def extra(c):
            _sc(c, self.hair_sh, 1)
            _taper(c, [(rx, ry), (mx + o * 1.5, my), (tx - o * 1, ty - 4)], 0.3, 0.2, 1.8)

        _cel(ctx, lambda c: _spline(c, pts), self.hair, None, None, None, None, extra=extra)
        ctx.new_path()
        _spline(ctx, [pts[4], pts[3], (pts[2][0], pts[2][1], True), pts[1]], False)
        _stroke(ctx, M_HAIR_LINE, self.lw)

    # ------------------------------------------------ hat
    def _hat(self, ctx):
        R = M_R
        lw = self.lw
        hat = _wet(M_HAT, self.wet, 0.35)
        hat_sh = _wet(M_HAT_SH, self.wet, 0.35)
        pitch = self.pitch
        yb, rxb, ryb = -0.6 * R - pitch * 0.05 * R, 1.4 * R, (0.24 + 0.12 * pitch) * R
        yd, rxd, ryd, Hd = -0.66 * R, 1.07 * R, (0.17 + 0.1 * pitch) * R, 0.84 * R
        # brim
        brim = lambda c: _ell(c, 0, yb, rxb, max(ryb, 2))

        def brim_extra(c):
            # inner shadow of the dome on the brim
            _ell(c, 0, yb - 0.04 * R, rxd * 1.02, max(ryb * 0.8, 2))
            _sc(c, hat_sh, 0.9)
            c.fill()

        _cel(ctx, brim, hat, hat_sh, M_HAT_HI, M_HAT_LINE, lw, soff=(-8 * self.ls, -4), roff=(0, -2.5),
             shade_a=0.9, rim_a=0.9, extra=brim_extra)
        # dome: front base arc + crown
        def dome(c):
            n = 14
            c.move_to(-rxd, yd)
            for i in range(1, n + 1):
                a = PI - PI * i / n
                c.line_to(rxd * math.cos(a), yd + ryd * math.sin(a))
            c.curve_to(rxd * 1.05, yd - Hd * 0.55, rxd * 0.62, yd - Hd * 1.02, 0, yd - Hd)
            c.curve_to(-rxd * 0.62, yd - Hd * 1.02, -rxd * 1.05, yd - Hd * 0.55, -rxd, yd)
            c.close_path()

        def dome_extra(c):
            # stitched seams (meridians) following the yaw
            for k in range(6):
                ph = -PI / 2 + PI * k / 5 + (0.0 if self.back else 0.0)
                a = ph + self.ya * 0.0 + (self.ya if not self.back else self.ya)
                pts, vis = [], True
                for j in range(9):
                    u = j / 8
                    rr = math.cos(u * PI / 2 * 0.96)
                    aa = PI / 2 * k / 5 * 2 - PI / 2 + self.ya
                    x = rxd * rr * math.sin(aa)
                    zf = math.cos(aa)
                    y = yd - Hd * math.sin(u * PI / 2) * (1.0 - 0.02 * zf) + ryd * rr * zf
                    pts.append((x, y))
                    vis = zf > 0.05
                if vis:
                    c.new_path()
                    _spline(c, pts, False)
                    c.set_dash([2.2, 2.6])
                    _stroke(c, M_HAT_SH, 1.1, 0.9)
                    c.set_dash([])
            # hat band
            c.new_path()
            for i in range(15):
                a = PI - PI * i / 14
                p = (rxd * 1.01 * math.cos(a), yd + ryd * math.sin(a) - 0.17 * R)
                (c.move_to if i == 0 else c.line_to)(*p)
            for i in range(15):
                a = PI * i / 14
                c.line_to(rxd * 1.01 * math.cos(a), yd + ryd * math.sin(a) + 1)
            c.close_path()
            _sc(c, hat_sh, 0.55)
            c.fill()
            # glossy highlight
            _sc(c, M_HAT_HI, 0.75)
            hx = -0.4 * R * self.ls
            _taper(c, [(hx - 0.25 * R, yd - Hd * 0.62), (hx, yd - Hd * 0.86), (hx + 0.3 * R, yd - Hd * 0.9)],
                   1.0, 1.0, 5.5)

        _cel(ctx, dome, hat, hat_sh, M_HAT_HI, M_HAT_LINE, lw, soff=(-9 * self.ls, -3), roff=(-2.8 * self.ls, 0.8),
             rim_a=0.8, extra=dome_extra)
        # top button
        _ell(ctx, 0, yd - Hd + 1, 5.5, 3.2)
        _sc(ctx, hat)
        ctx.fill_preserve()
        _stroke(ctx, M_HAT_LINE, lw * 0.8)
        # front brim edge re-stroke (over dome base)
        ctx.new_path()
        ctx.save()
        ctx.translate(0, yb)
        ctx.scale(rxb, max(ryb, 2))
        ctx.arc(0, 0, 1, 0.05, PI - 0.05)
        ctx.restore()
        _stroke(ctx, M_HAT_LINE, lw)
        self.hat_top = (0, yd - Hd - 3)
        self.drop_pts = [(0, yd - Hd), (-rxb * 0.9, yb), (rxb * 0.9, yb), (-rxd * 0.6, yd - Hd * 0.8),
                         (rxd * 0.6, yd - Hd * 0.8)]
        if self.wet > 0.05:
            for i, fx in enumerate((-0.8, -0.35, 0.2, 0.7)):
                ph = (self.t * 0.6 + i * 0.37) % 1.0
                x = fx * rxb
                y = yb + ryb * math.sqrt(max(0, 1 - fx * fx)) + 1.5
                draw_droplet(ctx, x, y + 3 * ph, 2.0 + 1.6 * ph, self.wet * (1 - 0.3 * ph))
            for i, (fx, fy) in enumerate(((-0.5, -0.5), (0.35, -0.75), (0.1, -0.3), (-0.2, -0.85))):
                draw_droplet(ctx, fx * rxd, yd - Hd * -fy * -1, 2.2, self.wet * 0.85)

    # ------------------------------------------------ face details
    def _blush(self, c):
        R = M_R
        if self.blush <= 0:
            return
        for side in (-1, 1):
            (x, y), z = self.sp(side * 0.68, 0.38 * R, 0.98)
            if z < 0.15:
                continue
            fs = 0.35 + 0.65 * z
            c.save()
            c.translate(x, y)
            c.scale(fs, 0.6)
            g = cairo.RadialGradient(0, 0, 0, 0, 0, 0.24 * R)
            col = _c(M_CHEEK)
            g.add_color_stop_rgba(0, *col, 0.7 * self.blush)
            g.add_color_stop_rgba(0.6, *col, 0.4 * self.blush)
            g.add_color_stop_rgba(1, *col, 0)
            c.set_source(g)
            c.arc(0, 0, 0.24 * R, 0, TAU)
            c.fill()
            _sc(c, "#E86A76", 0.6 * self.blush)
            for k in (-1, 0, 1):
                c.move_to(k * 0.08 * R + 0.02 * R, -0.07 * R)
                c.line_to(k * 0.08 * R - 0.02 * R, 0.07 * R)
            c.set_line_width(1.1)
            c.stroke()
            c.restore()

    def _eyes(self, ctx):
        R = M_R
        self.eye_pos = []
        for side in (-1, 1):
            (x, y), z = self.sp(side * 0.44, 0.15 * R, 0.94)
            if z < 0.1:
                continue
            fs = 0.4 + 0.6 * z
            self.eye_pos.append((x, y))
            self._eye(ctx, x, y, side, fs)

    def _eye(self, ctx, cx, cy, side, fs):
        R = M_R
        a = 0.245 * R
        b = 0.31 * R
        op = self.eye_open * (1 - self.blink)
        lid_t = self.lid_tilt
        low = self.lower
        flip = -1 if side < 0 else 1
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(fs, 1)
        if op < 0.12 or self.expr == "laugh":
            ctx.scale(flip, 1)
            _sc(ctx, M_LASH)
            if self.expr == "laugh" or (self.lower > 0.3 and self.smile > 0.5):
                pts = [(-a * 0.95, b * 0.3), (0, -b * 0.32), (a * 1.0, b * 0.3)]
                _taper(ctx, pts, 1.8, 2.2, 4.2)
            else:
                pts = [(-a * 0.95, b * 0.08), (0, b * 0.42), (a * 1.05, b * 0.02)]
                _taper(ctx, pts, 1.6, 2.4, 3.8)
                _taper(ctx, [(a * 0.95, b * 0.04), (a * 1.25, -b * 0.15)], 2.4, 0.4)
            ctx.restore()
            return
        top_y = -b * (0.35 + 0.65 * op)
        bot_y = b * (1.0 - 0.65 * low)

        def ty(u):
            return top_y * (1 - 0.5 * u * u) - b * 0.05 + lid_t * b * 0.5 * (u + 0.3) * op

        def by(u):
            return bot_y * max(0.0, 1 - 0.8 * u * u) ** 0.8 + b * 0.08 * u

        ctx.scale(flip, 1)
        us = [-1.0, -0.6, -0.1, 0.45, 0.85, 1.0]
        top = [(-a, ty(-1) * 0.6 + b * 0.1)] + [(u * a, ty(u)) for u in us[1:-1]] + [(a * 1.02, ty(1) * 0.7 + b * 0.02)]
        bot = [(0.8 * a, by(0.8) * 0.85), (0.1 * a, by(0.1)), (-0.72 * a, by(-0.72) * 0.9)]
        white = [(top[0][0], top[0][1], True)] + top[1:-1] + [(top[-1][0], top[-1][1], True)] + bot
        ctx.new_path()
        _spline(ctx, white)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.save()
        ctx.clip()
        ctx.scale(flip, 1)
        lx, ly = self.look
        ix = lx * a * 0.42
        iy = b * 0.12 + ly * b * 0.22
        irx, iry = a * 0.8 * self.iris, b * 0.86 * self.iris
        g = cairo.LinearGradient(0, iy - iry, 0, iy + iry)
        for o, col in ((0, M_IRIS[0]), (0.5, M_IRIS[1]), (1, M_IRIS[2])):
            g.add_color_stop_rgb(o, *_c(col))
        _ell(ctx, ix, iy, irx, iry)
        ctx.set_source(g)
        ctx.fill()
        _ell(ctx, ix, iy + iry * 0.5, irx * 0.68, iry * 0.36)
        _sc(ctx, "#E8B070", 0.45)
        ctx.fill()
        _ell(ctx, ix, iy - iry * 0.05, irx * 0.46, iry * 0.5)
        _sc(ctx, "#140804", 0.95)
        ctx.fill()
        _ell(ctx, ix, iy, irx, iry)
        _stroke(ctx, "#1C0C06", 1.2, 0.9)
        ctx.scale(flip, 1)
        ctx.new_path()
        _spline(ctx, [(-a * 1.2, top_y - b), (a * 1.2, top_y - b), (a * 1.2, ty(1) + b * 0.3),
                      (0, ty(0) + b * 0.38), (-a * 1.2, ty(-1) + b * 0.3)], True, 0.6)
        _sc(ctx, "#4A2A20", 0.25)
        ctx.fill()
        ctx.scale(flip, 1)
        hl = -self.ls
        if self.expr == "amazed":
            draw_sparkle(ctx, ix + hl * irx * 0.25, iy - iry * 0.32, irx * 0.62, 1.0, 0.0, "#FFFFFF")
            draw_sparkle(ctx, ix - hl * irx * 0.35, iy + iry * 0.42, irx * 0.32, 0.95, 0.4, "#FFFFFF")
            ctx.arc(ix + hl * irx * 0.55, iy + iry * 0.1, irx * 0.1, 0, TAU)
            ctx.set_source_rgba(1, 1, 1, 0.85)
            ctx.fill()
        else:
            _ell(ctx, ix + hl * irx * 0.3, iy - iry * 0.38, irx * 0.42, iry * 0.3, -0.3 * hl)
            ctx.set_source_rgba(1, 1, 1, 0.97)
            ctx.fill()
            ctx.arc(ix - hl * irx * 0.38, iy + iry * 0.42, irx * 0.17, 0, TAU)
            ctx.set_source_rgba(1, 1, 1, 0.9)
            ctx.fill()
            ctx.arc(ix + hl * irx * 0.52, iy + iry * 0.06, irx * 0.08, 0, TAU)
            ctx.set_source_rgba(1, 1, 1, 0.75)
            ctx.fill()
        ctx.restore()
        ctx.new_path()
        _sc(ctx, M_LASH)
        lash = [(top[0][0] - a * 0.02, top[0][1] + b * 0.05)] + top[1:-1] + [(a * 1.03, ty(1) * 0.7)]
        _taper(ctx, lash, 1.4, 3.8, 3.8)
        _taper(ctx, [(a * 0.72, ty(0.72) - 0.5), (a * 1.08, ty(1) * 0.8), (a * 1.34, ty(1) * 1.05 - b * 0.12)],
               3.0, 0.3)
        _sc(ctx, M_LASH, 0.7)
        _taper(ctx, [(a * 0.98, ty(1) * 0.55 + b * 0.25), (a * 0.82, by(0.8) * 0.8), (a * 0.45, by(0.45) * 0.97)],
               1.5, 0.3)
        _sc(ctx, M_SKIN_LINE, 0.35)
        _taper(ctx, [(-a * 0.3, ty(-0.3) - b * 0.28), (a * 0.4, ty(0.4) - b * 0.3), (a * 0.85, ty(0.85) - b * 0.18)],
               0.3, 0.3, 1.1)
        ctx.restore()

    def _brows(self, ctx):
        R = M_R
        for side in (-1, 1):
            (x, y), z = self.sp(side * 0.44, -0.27 * R - self.brow_y * 0.09 * R, 1.08)
            if z < 0.1:
                continue
            fs = 0.42 + 0.58 * z
            flip = -1 if side < 0 else 1
            ctx.save()
            ctx.translate(x, y)
            ctx.scale(fs * flip, 1)
            tl = self.brow_tilt * 0.08 * R
            pts = [(-0.19 * R, -tl + 0.01 * R), (0.0, -0.035 * R - tl * 0.1), (0.19 * R, tl * 0.6 + 0.02 * R)]
            _sc(ctx, "#4A2618", 0.9)
            _taper(ctx, pts, 2.8, 0.9, 2.8)
            ctx.restore()

    def _nose(self, ctx):
        R = M_R
        (x, y), z = self.sp(0.0, 0.45 * R, 0.99)
        if z < 0.0 or abs(math.sin(self.ya)) > 0.8:
            return
        _sc(ctx, M_SKIN_LINE, 0.55)
        _taper(ctx, [(x - 1.0, y - 2.0), (x + 1.3, y + 1.0)], 0.4, 1.6)

    def _mouth(self, ctx):
        R = M_R
        (x, y), z = self.sp(0.0, 0.66 * R, 0.86)
        if z < -0.1:
            return
        side = abs(math.sin(self.ya))
        fs = 0.55 + 0.45 * max(z, 0.0)
        ctx.save()
        ctx.translate(x - side * 2.5, y)
        ctx.scale(fs, 1)
        self.mouth_xy = (x, y)
        op = clamp(self.mouth_open)
        v = self.vowel if self.vowel in _M_VOWEL else "x"
        if v in ("x", "n") and op < 0.25:
            op = 0.0
        if op < 0.06:
            self._closed_mouth(ctx, v)
            ctx.restore()
            return
        hw, hh, rnd, teeth = _M_VOWEL[v]
        if self.mouth_kind == "laugh":
            hw, hh = max(hw, 9.5), max(hh, 9.0)
        hw *= 0.85 + 0.15 * op
        hh = max(1.6, hh * op * 1.05)
        cu = max(0.0, self.smile) * 1.6
        top = -hh * 0.25 - cu * 0.3
        self._open_mouth(ctx, hw, hh, rnd, teeth, cu, top)
        ctx.restore()

    def _open_mouth(self, ctx, hw, hh, rnd, teeth, cu, top):
        L, Rr = (-hw, -cu), (hw, -cu)
        ctx.new_path()
        ctx.move_to(*L)
        ctx.curve_to(-hw * 0.5, top - hh * 0.15 * rnd, hw * 0.5, top - hh * 0.15 * rnd, *Rr)
        bk = 0.55 + 0.45 * rnd
        ctx.curve_to(hw * (1.0 - 0.1 * rnd), hh * bk, hw * 0.45, hh * 1.05, 0, hh * 1.02)
        ctx.curve_to(-hw * 0.45, hh * 1.05, -hw * (1.0 - 0.1 * rnd), hh * bk, *L)
        ctx.close_path()
        path = ctx.copy_path()
        _sc(ctx, M_MOUTH_IN)
        ctx.fill_preserve()
        ctx.save()
        ctx.clip()
        _ell(ctx, 0, hh * 1.1, hw * 0.75, hh * 0.62)
        _sc(ctx, M_TONGUE)
        ctx.fill()
        if teeth > 0:
            ctx.rectangle(-hw, top - hh, hw * 2, hh * (0.55 + teeth * 0.2))
            ctx.set_source_rgba(1, 1, 1, 0.95)
            ctx.fill()
        ctx.restore()
        ctx.new_path()
        ctx.append_path(path)
        _stroke(ctx, M_MOUTH_LINE, self.lw * 0.7)

    def _closed_mouth(self, ctx, v):
        k = self.mouth_kind
        s = self.smile
        _sc(ctx, M_MOUTH_LINE)
        if v == "n":
            _taper(ctx, [(-4.0, 0.5), (0, 1.2), (4.0, 0.5)], 1.0, 1.0, 2.0)
            return
        if k in ("laugh",):
            self._open_mouth(ctx, 11.0, 10.0, 0.6, 0.3, 2.4, -3.0)
            return
        if k == "amazed":
            ctx.new_path()
            _ell(ctx, 0, 1.5, 5.0, 5.8)
            _sc(ctx, M_MOUTH_IN)
            ctx.fill_preserve()
            ctx.save()
            ctx.clip()
            _ell(ctx, 0, 6, 4, 3)
            _sc(ctx, M_TONGUE)
            ctx.fill()
            ctx.restore()
            _ell(ctx, 0, 1.5, 5.0, 5.8)
            _stroke(ctx, M_MOUTH_LINE, self.lw * 0.6)
            return
        if k == "o":
            _ell(ctx, 0, 0.5, 3.8, 4.8)
            _sc(ctx, M_MOUTH_IN)
            ctx.fill_preserve()
            _stroke(ctx, M_MOUTH_LINE, self.lw * 0.6)
            return
        if k == "pout":
            # pursed puffed lips: tiny "3"-ish pucker
            _taper(ctx, [(-3.5, -1.5), (-1.2, 0.2), (0.6, -0.8), (1.6, 0.6), (-0.4, 2.0), (-3.0, 2.4)],
                   1.0, 0.8, 1.9)
            return
        if k == "wavy":
            _taper(ctx, [(-6, 1), (-3, -0.8), (0, 0.8), (3, -0.6), (6, 1.2)], 0.8, 0.8, 1.8)
            return
        if k == "frown":
            _taper(ctx, [(-5.5, 2.0), (0, -1.2), (5.5, 2.0)], 0.8, 0.8, 2.0)
            return
        if k == "smile_open" or (k == "smile" and s > 0.85):
            ctx.new_path()
            ctx.move_to(-7, -1.5)
            ctx.curve_to(-3, 0, 3, 0, 7, -1.5)
            ctx.curve_to(5.5, 7.5, -5.5, 7.5, -7, -1.5)
            ctx.close_path()
            p = ctx.copy_path()
            _sc(ctx, M_MOUTH_IN)
            ctx.fill_preserve()
            ctx.save()
            ctx.clip()
            _ell(ctx, 0, 7, 4.8, 3.4)
            _sc(ctx, M_TONGUE)
            ctx.fill()
            ctx.restore()
            ctx.new_path()
            ctx.append_path(p)
            _stroke(ctx, M_MOUTH_LINE, self.lw * 0.6)
            return
        w = 4.5 + 2.5 * s
        _taper(ctx, [(-w, -s * 2.2), (0, 0.8 + s * 0.8), (w, -s * 2.2)], 0.8, 0.8, 2.0)

    def _puff_marks(self, ctx):
        """Little anger/puff ticks beside the puffed cheeks + a tiny vein mark."""
        R = M_R
        _sc(ctx, M_SKIN_LINE, 0.8)
        for side in (-1, 1):
            (x, y), z = self.sp(side * 1.0, 0.42 * R, 1.12)
            if z < 0.05:
                continue
            for k in (-1, 0, 1):
                a = side * 0.3 * k
                p0 = (x + side * 3, y + k * 6)
                p1 = (p0[0] + side * 6 * math.cos(a), p0[1] + 6 * math.sin(a))
                _taper(ctx, [p0, p1], 1.6, 0.6)
        # cross-popping vein (怒りマーク) on the hat brim side
        (x, y), z = self.sp(0.95, -0.85 * R, 1.25)
        if z > 0:
            ctx.save()
            ctx.translate(x + 6, y)
            _sc(ctx, "#E2453F", 0.95)
            for ang in (0.0, PI / 2, PI, 1.5 * PI):
                ctx.save()
                ctx.rotate(ang + 0.6)
                _taper(ctx, [(2, -2), (6, -5), (8, -2)], 2.4, 1.0, 2.6)
                ctx.restore()
            ctx.restore()

    def _amazed_sparkles(self, ctx):
        R = M_R
        t = self.t
        for i, (px, py, s) in enumerate(((-1.35, -0.4, 8), (1.4, -0.2, 10), (1.2, 0.6, 6), (-1.25, 0.55, 6))):
            tw = 0.6 + 0.4 * math.sin(t * 6 + i * 1.9)
            draw_sparkle(ctx, px * R, py * R, s * tw, 0.9, 0.2 * i, "#FFF3B0")


def _m_hand(ctx, W, d, size, kind, side_sign, lw, skin=M_SKIN):
    """Small mitten hand at wrist W pointing along d."""
    ang = math.atan2(d[1], d[0])
    ctx.save()
    ctx.translate(*W)
    ctx.rotate(ang)
    s = size
    th = side_sign
    if kind == "fist" or kind == "grip":
        pts = [(-0.2 * s, -0.6 * s), (0.6 * s, -0.75 * s), (1.15 * s, -0.35 * s), (1.15 * s, 0.4 * s),
               (0.6 * s, 0.75 * s), (-0.2 * s, 0.58 * s)]
        _cel(ctx, lambda c: _spline(c, pts), skin, M_SKIN_SH, None, M_SKIN_LINE, lw * 0.8, soff=(1.5, -2.5 * th),
             shade_a=0.8)
        _sc(ctx, M_SKIN_LINE, 0.5)
        ctx.set_line_width(0.9)
        for k in (-0.3, 0.05, 0.38):
            ctx.move_to(0.95 * s, k * s)
            ctx.line_to(1.12 * s, k * s)
        ctx.stroke()
    elif kind == "point":
        pts = [(-0.2 * s, -0.55 * s), (0.6 * s, -0.7 * s), (1.0 * s, -0.4 * s), (2.1 * s, -0.42 * s),
               (2.25 * s, -0.12 * s), (1.05 * s, 0.02 * s), (1.0 * s, 0.4 * s), (0.6 * s, 0.7 * s),
               (-0.2 * s, 0.55 * s)]
        _cel(ctx, lambda c: _spline(c, pts, True, 0.8), skin, M_SKIN_SH, None, M_SKIN_LINE, lw * 0.8,
             soff=(1.5, -2.5 * th), shade_a=0.8)
    else:
        spread = 0.25 if kind == "wave" else 0.0
        thumb = [(0.2 * s, th * 0.45 * s), (0.62 * s, th * (0.98 + spread) * s), (0.98 * s, th * 0.82 * s),
                 (0.75 * s, th * 0.4 * s)]
        _cel(ctx, lambda c: _spline(c, thumb), skin, None, None, M_SKIN_LINE, lw * 0.8)
        pts = [(-0.2 * s, -0.52 * s), (0.6 * s, -0.66 * s - spread * s), (1.38 * s, -0.42 * s - spread * s),
               (1.62 * s, 0.0), (1.38 * s, 0.42 * s), (0.6 * s, 0.62 * s), (-0.2 * s, 0.52 * s)]
        _cel(ctx, lambda c: _spline(c, pts), skin, M_SKIN_SH, None, M_SKIN_LINE, lw * 0.8, soff=(1.5, -2.5 * th),
             shade_a=0.8)
        _sc(ctx, M_SKIN_LINE, 0.45)
        ctx.set_line_width(0.9)
        for k in (-0.18, 0.15):
            ctx.move_to(1.0 * s, k * s * (1 + spread * 2))
            ctx.line_to(1.45 * s, k * s * (1.2 + spread * 3))
        ctx.stroke()
    ctx.restore()


# ---------------------------------------------------------------- randoseru (3D box)
_BOX_FACES = {  # name: (normal (l,f,v), corner order as (sl, sf, sv) for (origin, +u, +w))
    "top": ((0, 0, -1), (-1, 1, -1), (1, 1, -1), (-1, -1, -1)),
    "bottom": ((0, 0, 1), (-1, -1, 1), (1, -1, 1), (-1, 1, 1)),
    "outer": ((0, -1, 0), (1, -1, -1), (-1, -1, -1), (1, -1, 1)),   # the flap face (away from her back)
    "inner": ((0, 1, 0), (-1, 1, -1), (1, 1, -1), (-1, 1, 1)),
    "left": ((1, 0, 0), (1, 1, -1), (1, -1, -1), (1, 1, 1)),
    "right": ((-1, 0, 0), (-1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
}


def _draw_randoseru(ctx, P, center, axes, half, lw, wet=0.0, t=0.0, rim=RIM, ls=1, straps="hidden"):
    """Draw the randoseru as a rounded 3D box.  axes = 3 unit vectors (body space) for the
    box's local l (width), f (depth, + toward her back) and v (height, + down); half = half sizes.
    Returns dict(center=2D, top=2D, corners)."""
    hw, hd, hh = half
    ax, ay, az = axes
    cam = (-P.s, P.c, -P.tilt)

    def corner(sl, sf, sv):
        q = [center[i] + ax[i] * sl * hw + ay[i] * sf * hd + az[i] * sv * hh for i in range(3)]
        return P.p(*q), q

    base = _wet(M_RAN, wet, 0.5)
    sh = _wet(M_RAN_SH, wet, 0.5)
    faces = []
    for name, (n, o, u, w) in _BOX_FACES.items():
        nb = [ax[i] * n[0] + ay[i] * n[1] + az[i] * n[2] for i in range(3)]
        vis = nb[0] * cam[0] + nb[1] * cam[1] + nb[2] * cam[2]
        if vis <= 0.02:
            continue
        p0 = corner(*o)[0]
        pu = corner(*u)[0]
        pw = corner(*w)[0]
        faces.append((vis, name, p0, pu, pw, nb))
    allc = [corner(a, b, c)[0] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    hull = _hull(allc)
    ctx.new_path()
    _rpoly(ctx, hull, 9)
    _sc(ctx, sh)
    ctx.fill_preserve()
    hull_path = ctx.copy_path()
    ctx.new_path()
    ctx.save()
    ctx.append_path(hull_path)
    ctx.clip()
    for vis, name, p0, pu, pw, nb in sorted(faces, key=lambda f: f[0]):
        light = -nb[2] * 0.6 + vis * 0.4
        col = mix(sh, base, clamp(0.25 + light))
        if name == "top":
            col = mix(base, M_RAN_HI, 0.25)
        M = cairo.Matrix(pu[0] - p0[0], pu[1] - p0[1], pw[0] - p0[0], pw[1] - p0[1], p0[0], p0[1])

        def fpath(fn, M=M):
            ctx.save()
            ctx.transform(M)
            ctx.new_path()
            fn(ctx)
            ctx.restore()

        fpath(lambda c: c.rectangle(0.01, 0.01, 0.98, 0.98))
        _sc(ctx, col)
        ctx.fill()
        if name == "outer":
            def flap(c):
                c.move_to(0.04, 0.0)
                c.line_to(0.96, 0.0)
                c.line_to(0.96, 0.6)
                c.curve_to(0.96, 0.82, 0.82, 0.84, 0.5, 0.84)
                c.curve_to(0.18, 0.84, 0.04, 0.82, 0.04, 0.6)
                c.close_path()
            fpath(flap)
            _sc(ctx, mix(base, M_RAN_HI, 0.12))
            ctx.fill()
            # shadow under the flap
            fpath(lambda c: (c.move_to(0.06, 0.86), c.curve_to(0.3, 0.9, 0.7, 0.9, 0.94, 0.86)))
            _stroke(ctx, sh, lw * 1.2, 0.8)

            def stitch(c):
                c.move_to(0.09, 0.02)
                c.line_to(0.09, 0.6)
                c.curve_to(0.09, 0.77, 0.2, 0.79, 0.5, 0.79)
                c.curve_to(0.8, 0.79, 0.91, 0.77, 0.91, 0.6)
                c.line_to(0.91, 0.02)
            fpath(stitch)
            ctx.set_dash([2.0, 2.0])
            _stroke(ctx, M_RAN_HI, 1.0, 0.8)
            ctx.set_dash([])
            fpath(flap)
            _stroke(ctx, M_RAN_LINE, lw * 0.8)
            fpath(lambda c: _rrect(c, 0.12, 0.08, 0.76, 0.07, 0.03))
            _sc(ctx, M_RAN_HI, 0.35)
            ctx.fill()
            fpath(lambda c: _rrect(c, 0.42, 0.72, 0.16, 0.15, 0.04))
            _sc(ctx, M_GOLD)
            ctx.fill_preserve()
            _stroke(ctx, M_GOLD_SH, lw * 0.6)
        elif name in ("left", "right"):
            fpath(lambda c: _rrect(c, 0.3, 0.62, 0.4, 0.18, 0.05))
            _sc(ctx, "#FFC94A")
            ctx.fill_preserve()
            _stroke(ctx, "#A86A10", lw * 0.5)
            fpath(lambda c: (c.move_to(0.08, 0.05), c.line_to(0.08, 0.95), c.move_to(0.92, 0.05),
                             c.line_to(0.92, 0.95)))
            ctx.set_dash([2.0, 2.0])
            _stroke(ctx, M_RAN_HI, 0.9, 0.6)
            ctx.set_dash([])
        elif name == "top":
            fpath(lambda c: (c.move_to(0.35, 0.5), c.curve_to(0.4, 0.0, 0.6, 0.0, 0.65, 0.5)))
            _stroke(ctx, M_RAN_LINE, lw * 1.5)
            fpath(lambda c: (c.move_to(0.36, 0.5), c.curve_to(0.41, 0.06, 0.59, 0.06, 0.64, 0.5)))
            _stroke(ctx, sh, lw * 0.7)
    # wet sheen
    if wet > 0.05:
        cx_ = sum(p[0] for p in allc) / 8
        cy_ = sum(p[1] for p in allc) / 8
        for i in range(4):
            px = cx_ + (_hash(i, 3) - 0.5) * 50
            py = cy_ + (_hash(i, 5) - 0.5) * 70
            draw_droplet(ctx, px, py, 2.4, wet * 0.85)
    ctx.restore()
    # rim highlight + outline
    if rim:
        ctx.save()
        ctx.append_path(hull_path)
        ctx.clip()
        ctx.new_path()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(-1e4, -1e4, 2e4, 2e4)
        ctx.save()
        ctx.translate(-2.4 * ls, 1.0)
        ctx.append_path(hull_path)
        ctx.restore()
        _sc(ctx, rim, 0.5)
        ctx.fill()
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        ctx.restore()
    ctx.new_path()
    ctx.append_path(hull_path)
    _stroke(ctx, M_RAN_LINE, lw)
    ctr = P.p(*center)
    topc = P.p(*[center[i] - az[i] * hh for i in range(3)])
    return dict(center=ctr, top=topc, hull=hull)


# ---------------------------------------------------------------- Mina main
M_ARM_PRESETS = ("rest", "hands_clasped", "hands_to_mouth", "point", "wave", "arms_up",
                 "hold_randoseru_overhead", "shrug")


def draw_mina(ctx, x, y, scale=1.0, *, view="front", yaw=None, expr="neutral", mouth_open=0.0, vowel="x",
              blink=0.0, look=(0.0, 0.0), head_tilt=0.0, head_turn=0.0, head_pitch=0.0, body_lean=0.0,
              bounce=0.0, crouch=0.0, tiptoe=0.0, wet=0.0, wind=0.0, wind_dir=None, t=0.0, arms="rest",
              arm_phase=None, hand=None, hand_l=None, hand_r=None, randoseru="back", hold_target=None,
              hold_offset=(0.0, 0.0), hold_tilt=0.0, hold_flap=0.0, crop=None, rim=RIM, light="left", breathe=1.0,
              eye_open=None, blush=None, brow_tilt=None, smile=None, alpha=1.0):
    """Draw Mina.  (x, y) = ground point between her feet; scale 1 -> ~480 px tall (with hat).

    view : 'front' | 'three_quarter_left' | 'three_quarter_right' | 'side_left' | 'side_right' | 'back'
           (or yaw in radians: 0 front, +pi/2 facing screen-right, -pi/2 screen-left, pi back).
    expr : neutral happy laugh sad surprised worried pout amazed gentle
    mouth_open 0..1 + vowel 'aiueonx' (feed core.mouth('mina', tg)), blink 0..1, look=(dx,dy) -1..1
    head_tilt (rad, + clockwise), head_turn (extra head yaw, rad), head_pitch (-1..1, + looks up)
    body_lean (rad, + clockwise), bounce (+ squash / - stretch), crouch 0..1, tiptoe 0..1
    wet 0..1 (darker fabric/hair, clumped strands, droplets), wind 0..2 + wind_dir (+1 screen-right), t (s)
    arms : rest | hands_clasped | hands_to_mouth | point | wave | arms_up | hold_randoseru_overhead | shrug
           arm_phase (wave cycles; default t*2.2), hand = 'screen_left'|'screen_right' for point/wave.
    hand_l / hand_r : wrist override (l, f, v) in body space (v up-negative from the ground; shoulder
           ~(+-31, 0, -303), head centre v=-380).
    randoseru : 'back' | 'held' | None.  arms='hold_randoseru_overhead' implies 'held'.
    hold_target : (x, y) in caller coords where the randoseru (box centre) should go; it is clamped to
           her reach -- check anchors['randoseru_center'].  hold_offset (dx, dy) in px*scale relative
           to the default spot (above her head, slightly forward).  hold_tilt (rad).
    crop : None | 'bust' (skips legs/shoes).  rim colour (None = off), light 'left'|'right'.
    eye_open / blush / brow_tilt / smile override the expression preset.
    Returns anchors: head, face, eyes, mouth, chin, hat_top, top, neck, chest, hip, feet, hand_l, hand_r,
      hands, randoseru_center, randoseru_top (held: roof top surface), randoseru_span (x0, x1 held),
      drop_points (surfaces rain hits), facing, scale.
    """
    anchor, to_local = _anchor_fn(ctx)
    ya, mirror = _resolve_view(view, yaw)
    ls = (1 if light == "left" else -1) * mirror
    look = (look[0] * mirror, look[1])
    wdir = (wind_dir if wind_dir is not None else 1) * mirror
    head_tilt *= mirror
    body_lean *= mirror
    P = _P(ya)
    back = math.cos(ya) < -0.45
    sy = P.s
    if arms == "hold_randoseru_overhead":
        randoseru = "held"
    if randoseru == "held" and arms != "hold_randoseru_overhead":
        arms = "hold_randoseru_overhead"
    if arm_phase is None:
        arm_phase = t * 2.2
    lw = line_weight(scale)
    _LW[0] = lw
    pal = dict(blouse=_wet(M_BLOUSE, wet, 0.35), blouse_sh=_wet(M_BLOUSE_SH, wet, 0.4),
               navy=_wet(M_NAVY, wet, 0.6), navy_sh=_wet(M_NAVY_SH, wet, 0.6), skin=M_SKIN)

    if alpha < 1.0:
        ctx.push_group()
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale * mirror, scale)
    if bounce:
        sq = clamp(bounce, -0.5, 0.5)
        ctx.scale(1 + sq * 0.45, 1 - sq)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    anchors = {"facing": 0 if abs(sy) < 0.3 else (1 if (math.cos(ya) > -2) else 1) * mirror, "scale": scale}
    if abs(sy) < 0.3:
        anchors["facing"] = 0
    breath = math.sin(t * TAU / 3.2) * breathe
    drop_local = []

    # ------------------------------------------------ legs
    crouch = clamp(crouch)
    hip_v = M_HIP_V + 72 * crouch - 10 * tiptoe
    legs = {}
    for side in (-1, 1):
        H3 = (side * 13.0, -2 * crouch, hip_v)
        A3 = (side * (13.0 + 5 * crouch), -4 * crouch, M_ANKLE_V - 10 * tiptoe)
        K3, A3 = _ik(list(H3), list(A3), M_THIGH, M_SHIN, (side * 0.35 * crouch, 1.0, 0.0))
        legs[side] = (H3, K3, A3)

    def draw_leg(side):
        H3, K3, A3 = legs[side]
        Hh, K, A = P.p3(H3), P.p3(K3), P.p3(A3)
        far = P.z(side * 13, 0) < -2
        sk = M_SKIN if not far else mix(M_SKIN, M_SKIN_SH, 0.35)
        mid1 = _l2(Hh, K, 0.5)
        calf = _add(_l2(K, A, 0.32), (0, 0))
        pts = [Hh, mid1, K, calf, A]
        rad = [13.0, 11.6, 9.4, 10.2, 7.0]
        legp = lambda c: _limb(c, pts, rad)
        sock = _wet(M_SOCK, wet, 0.3)

        def extra(c):
            # sock: below 2/3 of the shin
            c.save()
            sy0 = lerp(K[1], A[1], 0.52)
            c.rectangle(-300, sy0, 600, 300)
            c.clip()
            c.new_path()
            legp(c)
            _sc(c, sock)
            c.fill()
            c.new_path()
            c.move_to(A[0] - 12, sy0 + 5)
            c.line_to(A[0] + 12, sy0 + 5)
            _stroke(c, M_SOCK_SH, 1.0)
            c.restore()
            c.new_path()
            c.move_to(-300, lerp(K[1], A[1], 0.52))
            c.line_to(300, lerp(K[1], A[1], 0.52))
            _stroke(c, M_BLOUSE_LINE, lw * 0.8)

        _cel(ctx, legp, sk, M_SKIN_SH, rim if not far else None, M_SKIN_LINE, lw * 0.9,
             soff=(-4 * ls, 0), roff=(-2.0 * ls, 0), shade_a=0.6, rim_a=0.5, extra=extra)
        # shoe
        heel = P.p(A3[0], A3[1] - 6, -8 + 2 * tiptoe)
        toe = P.p(A3[0], A3[1] + 22, -8 + 8 * tiptoe if tiptoe else -8)
        sp_ = [heel, toe]
        shoe = lambda c: _limb(c, sp_, [11.0, 12.5])
        sc = _wet(M_SHOE, wet, 0.3)

        def sh_extra(c):
            # sole
            c.new_path()
            c.rectangle(-300, max(heel[1], toe[1]) + 5.5, 600, 30)
            _sc(c, M_SHOE_LINE, 0.85)
            c.fill()
            # strap
            mid = _l2(heel, toe, 0.38)
            c.new_path()
            c.move_to(mid[0] - 9, mid[1] - 7)
            c.curve_to(mid[0] - 3, mid[1] - 3, mid[0] + 3, mid[1] - 3, mid[0] + 9, mid[1] - 7)
            _stroke(c, M_SHOE_LINE, 2.0)
            # gloss
            g = _l2(heel, toe, 0.7)
            _ell(c, g[0] - 2, g[1] - 4, 3.8, 2.0, -0.2)
            c.set_source_rgba(1, 1, 1, 0.75)
            c.fill()

        _cel(ctx, shoe, sc, M_SHOE_SH, None, M_SHOE_LINE, lw * 0.9, soff=(-3 * ls, -3), extra=sh_extra)
        anchors_feet.append(anchor(*_l2(heel, toe, 0.5)))

    anchors_feet = []
    if crop != "bust":
        order = sorted((-1, 1), key=lambda s: P.z(s * 13, 0))
        for s_ in order:
            draw_leg(s_)

    # ------------------------------------------------ upper body frame
    hip_pt = P.p(0, 0, hip_v)
    auto_lean = 0.0
    if arms == "hold_randoseru_overhead":
        if hold_target is not None:
            tx_, ty_ = to_local(*hold_target)
        else:
            tx_ = 20.0 + hold_offset[0]
        auto_lean = clamp((tx_ - hip_pt[0]) / 650.0, -0.22, 0.22)
    ctx.save()
    ctx.translate(*hip_pt)
    ctx.rotate(body_lean + auto_lean + 0.12 * crouch * (1 if abs(sy) > 0.3 else 0))
    ctx.translate(-hip_pt[0], -hip_pt[1] + (hip_v - M_HIP_V))
    # (upper body is drawn at standing heights; frame translated down by crouch)

    # arm targets ------------------------------------------------------
    def screen_side_to_anat(h):
        want = 1 if h == "screen_right" else -1
        want *= mirror
        xs = {s: P.p(s * 30, 0, 0)[0] for s in (-1, 1)}
        return 1 if (xs[1] - xs[-1]) * want > 0 else -1

    one = screen_side_to_anat(hand or "screen_right")
    if abs(sy) > 0.3 and hand is None:
        one = -1  # near hand in 3/4 and side views
    hold = None
    if arms == "hold_randoseru_overhead":
        base = (14.0, -492.0)  # default: above her hat, a bit forward (screen x, y in body frame)
        if hold_target is not None:
            lx, ly = to_local(*hold_target)
            base = (lx, ly - (hip_v - M_HIP_V))
        base = (base[0] + hold_offset[0], base[1] + hold_offset[1])
        # clamp to reach: hands 30 px below box centre, each hand within arm reach
        reach = (M_UPPER + M_FORE) * 1.25  # chibi arms may stretch up to 25 % when holding overhead
        sh_c = (0.0, M_SHOULDER[1])
        hc = (base[0], base[1] + 28)
        dvec = _sub(hc, sh_c)
        dl = _len(dvec)
        if dl > reach:
            hc = _add(sh_c, _norm(dvec), reach)
        hold = dict(box=(hc[0], hc[1] - 28), hands=hc)
        anchors["hold_clamped"] = _len(_sub(hc, (base[0], base[1] + 28))) > 1.0

    def target(side):
        s = side
        rest = ((s * 40, 8, -196 + breath * 0.5), (s * 0.4, -1.0, 0.0), "open", "front")
        k = arms
        if k == "rest":
            return rest
        if k == "hands_clasped":
            return ((s * 5, 26, -236), (s * 1, -0.2, 0.4), "clasp", "front")
        if k == "hands_to_mouth":
            return ((s * 9, 46, -338), (s * 0.9, 0.1, 1), "cheek", "over")
        if k == "point":
            if s == one:
                dl_, df_ = P.c, P.s
                return ((s * 31 + dl_ * 112, df_ * 112 + 10, -296), (0, -0.3, 1), "point", "front")
            return rest
        if k == "wave":
            if s == one:
                wv = math.sin(arm_phase * TAU)
                return ((s * (78 + 12 * wv), 14, -396 + 5 * abs(wv)), (s * 1, -0.2, 0.6), "wave", "front")
            return rest
        if k == "arms_up":
            wv = math.sin(arm_phase * TAU) * 6
            return ((s * 92, 16, -436 + wv * s), (s * 1, 0, 0.3), "wave", "front")
        if k == "shrug":
            return ((s * 64, 22, -262), (s * 1, -0.6, 0.6), "palm_up", "front")
        if k == "hold_randoseru_overhead":
            hx, hy = hold["hands"]
            # hand spread along the screen-x direction in body space
            off = s * 40 * (1 if P.c >= 0 else -1)
            lx = hx * P.c + off * P.c
            fx = hx * P.s + off * P.s * 0.4 + 10
            if abs(P.s) > 0.6:  # side views: hands spread in depth so both are visible
                lx = s * 12
                fx = hx * P.s + s * 22
            return ((lx, fx, hy), (s * 1, -0.3, 0.6), "grip", "front")
        return rest

    arm_info = {}
    for side in (-1, 1):
        S = (side * M_SHOULDER[0], 0.0, M_SHOULDER[1] + breath * 0.8)
        ov = hand_l if side == 1 else hand_r
        if ov is not None:
            tg = (tuple(ov), (side, -0.5, 0.3), "open", "front")
        else:
            tg = target(side)
        Wt, pole, kind, layer = tg
        dist = math.sqrt(sum((Wt[i] - S[i]) ** 2 for i in range(3)))
        maxk = 1.25 if (arms in ("hold_randoseru_overhead", "arms_up", "wave") and ov is None) else 1.0
        k_st = clamp(dist / ((M_UPPER + M_FORE) * 0.98), 1.0, maxk)
        E, Wr = _ik(list(S), list(Wt), M_UPPER * k_st, M_FORE * k_st, pole)
        zs = P.z(S[0], S[1])
        zw = P.z(Wr[0], Wr[1])
        if layer != "over":
            if back:
                layer = "front"
            elif zs < -6 and not (kind in ("clasp", "grip") and zw > 0):
                layer = "back"
            elif kind == "grip" and ov is None:
                layer = "under_head"
            else:
                layer = "front"
        arm_info[side] = dict(S=S, E=E, W=Wr, kind=kind, layer=layer, z=zs)

    def draw_arm(side, hand_only=False):
        a = arm_info[side]
        S2, E2, W2 = P.p3(a["S"]), P.p3(a["E"]), P.p3(a["W"])
        if hand_only:
            d = _norm(_sub(W2, E2)) if _len(_sub(W2, E2)) > 1 else (0, 1)
            _m_hand(ctx, W2, _norm((d[0] * 0.5, -1.0)), 11.0, "fist", side, lw, M_SKIN)
            return
        if _len(_sub(E2, S2)) < 6:
            E2 = (E2[0] + side * 5, E2[1] + 4)
        far = a["z"] < -6 and not back
        sk = M_SKIN if not far else mix(M_SKIN, M_SKIN_SH, 0.3)
        d = _norm(_sub(W2, E2)) if _len(_sub(W2, E2)) > 1 else (0, 1)
        _cel(ctx, lambda c: _limb(c, [S2, E2, W2], [7.4, 6.6, 5.6]), sk, M_SKIN_SH, None, M_SKIN_LINE, lw * 0.85,
             soff=(2.0 * ls, -1.5), shade_a=0.6)
        kind = a["kind"]
        hs = 11.0
        th = side * (1 if not back else -1)
        if kind == "clasp":
            _m_hand(ctx, W2, _norm((-side * mirror * 0.0 + d[0] * 0.4, 1.0)), hs, "fist", th, lw, sk)
        elif kind == "cheek":
            _m_hand(ctx, W2, _norm((d[0] * 0.3, -1.0)), hs, "fist", -side, lw, sk)
        elif kind == "palm_up":
            _m_hand(ctx, W2, _norm((d[0], -0.4)), hs, "open", side, lw, sk)
        elif kind == "grip":
            _m_hand(ctx, W2, _norm((d[0] * 0.5, -1.0)), hs, "fist", side, lw, sk)
        else:
            _m_hand(ctx, W2, d if kind != "wave" else _norm((d[0] + 0.3 * math.sin(arm_phase * TAU), d[1])),
                    hs, kind if kind in ("point", "wave", "fist") else "open", th, lw, sk)
        # puff sleeve
        u = _norm(_sub(E2, S2))
        Ls = _len(_sub(E2, S2))
        n = (-u[1], u[0])
        c0 = _add(S2, u, -6)
        c1 = _add(S2, u, Ls * 0.52)
        bl = pal["blouse"] if not far else mix(pal["blouse"], pal["blouse_sh"], 0.5)
        sl = [_add(c0, n, 10), _add(_add(S2, u, Ls * 0.22), n, 13.0), _add(c1, n, 11.0),
              _add(_add(c1, u, 2.5), n, 0), _add(c1, n, -11.0), _add(_add(S2, u, Ls * 0.22), n, -13.0),
              _add(c0, n, -10)]

        def sl_extra(c):
            _sc(c, pal["blouse_sh"], 0.9)
            c.new_path()
            c.move_to(*_add(c1, n, -10.5))
            c.curve_to(*_add(_add(c1, u, -6), n, -4), *_add(_add(c1, u, -6), n, 4), *_add(c1, n, 10.5))
            c.set_line_width(1.0)
            c.stroke()

        _cel(ctx, lambda c: _spline(c, sl), bl, pal["blouse_sh"], rim if not far else None, M_BLOUSE_LINE, lw,
             soff=(-4 * ls, -4), roff=(-2.2 * ls, 0.5), rim_a=0.6, extra=sl_extra)
        # cuff band
        ctx.new_path()
        ctx.move_to(*_add(c1, n, -11.0))
        ctx.curve_to(*_add(_add(c1, u, 3.5), n, -6), *_add(_add(c1, u, 3.5), n, 6), *_add(c1, n, 11.0))
        _stroke(ctx, M_BLOUSE_LINE, lw * 0.9)
        anchors["hand_l" if side == 1 else "hand_r"] = anchor(*W2)
        a["W2"] = W2

    # ------------------------------------------------ randoseru on back
    box_axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    box_center = (0.0, -42.0, -255.0 + breath * 0.6)
    box_half = (47.0, 22.0, 56.0)

    def draw_box_back():
        r = _draw_randoseru(ctx, P, box_center, box_axes, box_half, lw, wet, t, rim, ls)
        anchors["randoseru_center"] = anchor(*r["center"])
        anchors["randoseru_top"] = anchor(*r["top"])

    # ------------------------------------------------ torso / skirt
    def torso():
        L, Rr = [], []
        for v, w, d, f0 in _M_TORSO:
            cx, e = _ring_ext(P, w, d, f0)
            yy = v + P.tilt * f0 * P.c
            L.append((cx - e, yy))
            Rr.append((cx + e, yy))
        pts = Rr + L[::-1]
        tp = lambda c: _spline(c, pts)

        def extra(c):
            # pinafore bib on the front (or crossed straps on the back)
            if not back:
                bib = []
                for i in range(9):
                    ph = -0.62 + 1.24 * i / 8
                    bib.append(surf(_M_TORSO, -296 + 4 * abs(ph), ph))
                for i in range(5):
                    ph = 0.75 - 1.5 * i / 4
                    bib.append(surf(_M_TORSO, -246, ph * 1.1))
                c.new_path()
                _spline(c, [(p[0], p[1]) for p in bib], True, 0.6)
                _sc(c, pal["navy"])
                c.fill_preserve()
                _stroke(c, M_NAVY_LINE, lw * 0.8)
                for ph in (-0.45, 0.45):
                    p = surf(_M_TORSO, -287, ph)
                    if p[2] > 0.1:
                        c.arc(p[0], p[1], 2.6, 0, TAU)
                        _sc(c, M_GOLD)
                        c.fill_preserve()
                        _stroke(c, M_GOLD_SH, 0.8)
                # pinafore straps over the shoulders
                for side in (-1, 1):
                    pts_ = [surf(_M_TORSO, -298, side * 0.5), surf(_M_TORSO, -311, side * 0.62),
                            surf(_M_TORSO, -315, side * 0.9)]
                    if pts_[0][2] > -0.2:
                        _sc(c, pal["navy"])
                        _taper(c, [(p[0], p[1]) for p in pts_], 7.5, 6.0)
            else:
                for side in (-1, 1):
                    a_ = surf(_M_TORSO, -312, PI + side * 0.6)
                    b_ = surf(_M_TORSO, -250, PI - side * 0.55)
                    _sc(c, pal["navy"])
                    _taper(c, [(a_[0], a_[1]), (b_[0], b_[1])], 7.0, 7.5)

        _cel(ctx, tp, pal["blouse"], pal["blouse_sh"], rim, M_BLOUSE_LINE, lw, soff=(-7 * ls, -5),
             roff=(-2.4 * ls, 0.5), rim_a=0.6, extra=extra)

    def surf(rings, v, phi, k=1.0):
        w, d, f0 = _interp_rings(rings, v)
        l = w * math.sin(phi) * k
        f = f0 + d * math.cos(phi) * k
        px, py = P.p(l, f, v)
        return (px, py, P.z(l, f) / max(d, 1))

    def skirt():
        rings = _M_SKIRT
        L, Rr = [], []
        for v, w, d, f0 in rings:
            cx, e = _ring_ext(P, w, d, f0)
            sw = 0.0
            if v > -200:
                sw = (wind * 3 * wdir * math.sin(t * 3) + math.sin(t * 2.1) * 0.8) * (v + 200) / 32
            L.append((cx - e + sw, v))
            Rr.append((cx + e + sw, v))
        hv, hw_, hd_, hf = rings[-1]
        phc = -ya
        hem = []
        n = 18
        sw_h = (wind * 3 * wdir * math.sin(t * 3) + math.sin(t * 2.1) * 0.8)
        for i in range(n + 1):
            ph = phc + PI / 2 - PI * i / n
            l = hw_ * math.sin(ph)
            f = hf + hd_ * math.cos(ph)
            px, py = P.p(l, f, hv)
            py += 2.2 * abs(math.sin(ph * 6))
            hem.append((px + sw_h, py, False))
        pts = Rr[:-1] + hem[1:-1] + L[::-1][1:]
        # ensure hem goes right->left: Rr ends at right side, hem sampled from ph=phc+pi/2 (screen right)
        hem_sorted = sorted(hem, key=lambda p: -p[0])
        pts = Rr + [(p[0], p[1]) for p in hem_sorted[1:-1]] + L[::-1]
        sp = lambda c: _spline(c, pts, True, 0.8)

        def extra(c):
            # pleat lines
            _sc(c, pal["navy_sh"], 0.9)
            for ph in (-1.2, -0.75, -0.3, 0.15, 0.6, 1.05, 1.5):
                for base_ph in ((phc + ph),):
                    a_ = surf(rings, -236, base_ph)
                    b_ = surf(rings, -168, base_ph)
                    if a_[2] > 0.1:
                        _taper(c, [(a_[0], a_[1]), (b_[0] + sw_h, b_[1])], 0.3, 1.6)
            # waist seam
            c.new_path()
            for i in range(13):
                ph = phc + PI / 2 - PI * i / 12
                p = surf(rings, -252, ph)
                (c.move_to if i == 0 else c.line_to)(p[0], p[1] + 2)
            _stroke(c, M_NAVY_LINE, lw * 0.6, 0.7)
            _sc(c, M_NAVY_HI, 0.35)
            hx = -ls * 22
            _taper(c, [(hx, -240), (hx * 1.4, -200), (hx * 1.7, -178)], 2.0, 4.5)

        _cel(ctx, sp, pal["navy"], pal["navy_sh"], rim, M_NAVY_LINE, lw, soff=(-9 * ls, -3), roff=(-2.6 * ls, 0.5),
             rim_a=0.55, extra=extra)
        if wet > 0.15:
            for i in range(3):
                ph = (t * 0.5 + i * 0.33) % 1.0
                p = hem_sorted[3 + i * 5]
                draw_droplet(ctx, p[0], p[1] + 3 + 8 * ph, 2.2, wet * (1 - ph))

    def collar_and_straps():
        if back:
            return
        # round white collar (Peter-Pan) at the neck front
        for side in (-1, 1):
            a_ = surf(_M_TORSO, -318, side * 0.15)
            b_ = surf(_M_TORSO, -314, side * 1.35)
            if a_[2] < -0.3 and b_[2] < -0.3:
                continue
            m_ = surf(_M_TORSO, -302, side * 0.55)
            pts = [(a_[0], a_[1] - 1), (b_[0], b_[1] - 1), (b_[0] + side * 1 * P.c, b_[1] + 3),
                   (m_[0], m_[1] + 2), (a_[0], a_[1] + 9, True)]
            _cel(ctx, lambda c, pts=pts: _spline(c, pts, True, 0.8), pal["blouse"], pal["blouse_sh"], None,
                 M_BLOUSE_LINE, lw * 0.85, soff=(-3 * ls, -3), shade_a=0.8)
        # small yellow neck ribbon to echo the hat
        p = surf(_M_TORSO, -312, 0.0)
        if p[2] > 0.2:
            ctx.save()
            ctx.translate(p[0], p[1] + 4)
            ctx.scale(0.5 + 0.5 * P.c, 1)
            for sgn in (-1, 1):
                ctx.new_path()
                ctx.move_to(0, 0)
                ctx.curve_to(sgn * 5, -6, sgn * 10, -4, sgn * 9, 1)
                ctx.curve_to(sgn * 9, 5, sgn * 4, 4, 0, 0)
                _sc(ctx, M_SHOE)
                ctx.fill_preserve()
                _stroke(ctx, M_SHOE_LINE, lw * 0.6)
            _ell(ctx, 0, 0, 2.6, 2.8)
            _sc(ctx, M_SHOE)
            ctx.fill_preserve()
            _stroke(ctx, M_SHOE_LINE, lw * 0.6)
            ctx.restore()
        # randoseru shoulder straps (red) over the shoulders, down to the armpits
        if randoseru == "back":
            for side in (-1, 1):
                pts3 = [(-311, side * 0.7), (-298, side * 0.86), (-280, side * 0.98), (-262, side * 1.12)]
                pts = [surf(_M_TORSO, v_, ph_, 1.07) for v_, ph_ in pts3]
                if max(p[2] for p in pts) < -0.1:
                    continue
                c0 = P.p(side * 25, -12, -318)
                pts2 = [c0] + [(p[0], p[1]) for p in pts]
                rc = _wet(M_RAN, wet, 0.5)
                _sc(ctx, M_RAN_LINE)
                _taper(ctx, pts2, 9.5, 8.5, 10.0)
                _sc(ctx, rc)
                _taper(ctx, pts2, 6.8, 6.0, 7.4)
                _sc(ctx, M_RAN_HI, 0.5)
                _taper(ctx, [_add(p, (-1.5 * ls, -1)) for p in pts2[:3]], 1.2, 0.8, 2.0)
                # buckle
                bp = pts[2]
                ctx.save()
                ctx.translate(bp[0], bp[1])
                _rrect(ctx, -4, -3, 8, 6, 1.5)
                ctx.restore()
                _sc(ctx, M_GOLD)
                ctx.fill_preserve()
                _stroke(ctx, M_GOLD_SH, 0.8)

    # neck
    def neck():
        n0 = P.p(0, 0, -336)
        n1 = P.p(0, 1, -312)
        _cel(ctx, lambda c: _limb(c, [n0, n1], [8.5, 9.5]), M_SKIN, M_SKIN_SH, None, M_SKIN_LINE, lw * 0.85,
             soff=(-3 * ls, 0), shade_a=0.9)
        # shadow under the chin
        ctx.new_path()
        _ell(ctx, n0[0] + 1.5 * P.s, n0[1] + 3, 9, 5)
        _sc(ctx, M_SKIN_SH)
        ctx.fill()

    # ------------------------------------------------ head
    over = dict(eye_open=eye_open, blush=blush, brow_tilt=brow_tilt, smile=smile)
    ya_h = clamp(ya + head_turn * mirror, 0.0, PI) if not back else PI
    head = _MinaHead(ya_h, head_pitch, expr, t, blink, look, mouth_open, vowel, over, rim, ls, wet,
                     wind * wdir, lw)
    hc = P.p(0, 2.0, M_HEAD_V + breath * 1.0)

    def draw_head():
        ctx.save()
        ctx.translate(hc[0], hc[1] + 46)
        ctx.rotate(head_tilt)
        ctx.translate(0, -46)
        head.draw(ctx)
        R = M_R
        anchors["head"] = anchor(0, 0)
        anchors["face"] = anchor(0.32 * R * math.sin(ya_h), 0.2 * R) if not back else anchor(0, 0)
        anchors["eyes"] = [anchor(*e) for e in head.eye_pos]
        anchors["mouth"] = anchor(*head.mouth_xy)
        anchors["chin"] = anchor(0.6 * R * math.sin(ya_h) if not back else 0, 0.98 * R)
        anchors["hat_top"] = anchor(*getattr(head, "hat_top", (0, -1.7 * R)))
        anchors["top"] = anchors["hat_top"]
        for p in head.drop_pts:
            drop_local.append(anchor(*p))
        ctx.restore()

    # ------------------------------------------------ draw order
    if randoseru == "back" and not back:
        draw_box_back()
    for side in sorted(arm_info, key=lambda s: arm_info[s]["z"]):
        if arm_info[side]["layer"] == "back":
            draw_arm(side)
    if not (back and False):
        pass
    skirt()
    torso()
    neck()
    collar_and_straps()
    if back:
        for side in (-1, 1):
            if arm_info[side]["layer"] == "front":
                draw_arm(side)
        draw_head()
        if randoseru == "back":
            draw_box_back()
    else:
        for side in sorted(arm_info, key=lambda s: arm_info[s]["z"]):
            if arm_info[side]["layer"] == "front":
                draw_arm(side)
        for side in sorted(arm_info, key=lambda s: arm_info[s]["z"]):
            if arm_info[side]["layer"] == "under_head":
                draw_arm(side)
        draw_head()
        for side in sorted(arm_info, key=lambda s: arm_info[s]["z"]):
            if arm_info[side]["layer"] == "over":
                draw_arm(side)

    # ------------------------------------------------ held randoseru overhead
    if randoseru == "held":
        bx, by = hold["box"]
        # long axis along screen-x (in body space), box lying flat: flap face up
        ex = (P.c, P.s, 0.0)
        ef = (-P.s, P.c, 0.0)
        ca, sa = math.cos(hold_tilt), math.sin(hold_tilt)
        # rotate in the screen plane by hold_tilt: long axis gets a v component
        e_long = (ex[0] * ca, ex[1] * ca, sa)
        e_up = (-ex[0] * sa, -ex[1] * sa, ca)  # +v = down
        Pb = _P(ya, tilt=-0.34)
        center3 = (bx * P.c, bx * P.s, by)
        # box local axes: width(l) -> depth direction, depth(f, +toward her back = 'inner' face) -> down,
        # height(v) -> long axis
        axes = (ef, e_up, e_long)
        # straps dangling below
        sw = 0.15 * math.sin(t * 3.0)
        for k in (-1, 1):
            a3 = [center3[i] + e_long[i] * k * 30 + e_up[i] * 22 for i in range(3)]
            a2 = Pb.p3(a3)
            loop = [(a2[0] - 9, a2[1]), (a2[0] - 9 + 14 * sw, a2[1] + 34), (a2[0] + 9 + 14 * sw, a2[1] + 34),
                    (a2[0] + 9, a2[1])]
            ctx.new_path()
            ctx.move_to(*loop[0])
            ctx.curve_to(*loop[1], *loop[2], *loop[3])
            _stroke(ctx, M_RAN_LINE, 9.0)
            ctx.new_path()
            ctx.move_to(*loop[0])
            ctx.curve_to(*loop[1], *loop[2], *loop[3])
            _stroke(ctx, _wet(M_RAN_SH, wet, 0.5), 6.0)
        r = _draw_randoseru(ctx, Pb, center3, axes, (47.0, 22.0, 56.0), lw, wet, t, rim, ls)
        # the lid (kabuse) swings open from the far end and extends the roof
        fo = clamp(hold_flap)
        flap_pts = []
        if fo > 0.01:
            ang = (PI + 1.25) * ease_out_back(fo, 1.2)
            hinge = [center3[i] + ef[i] * 47 - e_up[i] * 22 for i in range(3)]
            dir_ = [ef[i] * -math.cos(ang) - e_up[i] * math.sin(ang) for i in range(3)]
            corners = []
            for wsg, ln in ((-1, 0), (1, 0), (1, 1), (-1, 1)):
                corners.append(Pb.p3([hinge[i] + e_long[i] * wsg * 53 + dir_[i] * 84 * ln for i in range(3)]))
            ctx.new_path()
            _rpoly(ctx, [corners[0], corners[1], _l2(corners[2], corners[1], 0.12), _l2(corners[2], corners[3], 0.2),
                         _l2(corners[3], corners[2], 0.2), _l2(corners[3], corners[0], 0.12)], 6)
            under = math.sin(ang) * 1.0
            _sc(ctx, _wet(M_RAN_HI if ang > PI / 2 else M_RAN, wet, 0.5) if False else _wet(mix(M_RAN, M_RAN_HI, 0.15), wet, 0.5))
            ctx.fill_preserve()
            _stroke(ctx, M_RAN_LINE, lw)
            tipm = _l2(corners[2], corners[3], 0.5)
            ctx.save()
            ctx.translate(*tipm)
            _rrect(ctx, -6, -1, 12, 9, 2.5)
            ctx.restore()
            _sc(ctx, M_GOLD)
            ctx.fill_preserve()
            _stroke(ctx, M_GOLD_SH, lw * 0.6)
            flap_pts = corners
        anchors["randoseru_center"] = anchor(*r["center"])
        xs = [p[0] for p in r["hull"]] + [p[0] for p in flap_pts]
        ys = [p[1] for p in r["hull"]]
        anchors["randoseru_top"] = anchor(r["center"][0], min(ys))
        x0, x1 = anchor(min(xs), min(ys))[0], anchor(max(xs), min(ys))[0]
        anchors["randoseru_span"] = (min(x0, x1), max(x0, x1))
        for side in sorted(arm_info, key=lambda s: arm_info[s]["z"]):
            if arm_info[side]["kind"] == "grip":
                draw_arm(side, hand_only=True)
        drop_local.extend([anchor(r["center"][0] + dx, min(ys) + 2) for dx in (-40, -10, 25, 50)])

    # ------------------------------------------------ anchors
    anchors["neck"] = anchor(*P.p(0, 0, -322))
    anchors["chest"] = anchor(*P.p(0, 18, -282))
    for side in (-1, 1):
        k = "hand_l" if side == 1 else "hand_r"
        if k not in anchors:
            anchors[k] = anchor(*P.p3(arm_info[side]["W"]))
    ctx.restore()  # upper-body frame
    anchors["hip"] = anchor(*P.p(0, 0, hip_v))
    anchors["feet"] = anchor(0, 0)
    if anchors_feet:
        anchors["foot_l"], anchors["foot_r"] = anchors_feet[0], anchors_feet[-1]
    hl_, hr_ = anchors["hand_l"], anchors["hand_r"]
    anchors["hands"] = ((hl_[0] + hr_[0]) / 2, (hl_[1] + hr_[1]) / 2)
    drop_local.extend([anchor(*P.p(s_ * 34, 0, -312)) for s_ in (-1, 1)])
    anchors["drop_points"] = drop_local
    ctx.restore()
    if alpha < 1.0:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(alpha)
    return anchors


# =============================================================== Robo (No.7)
RB_BODY = [(-238, 38, 32), (-228, 62, 52), (-214, 76, 63), (-190, 84, 70), (-140, 88, 73), (-80, 86, 71),
           (-56, 78, 64), (-40, 62, 52), (-30, 34, 30)]
RB_HEAD = dict(hw=98.0, hd=58.0, top=-368.0, bot=-246.0, r=30.0)
RB_SHOULDER = (88.0, -186.0)
RB_UPPER, RB_FORE = 54.0, 54.0
RB_ANT_L = 40.0          # antenna lateral position on the head top
RB_HATCH_L = 6.0       # mini-umbrella hatch lateral position
RB_TREAD = dict(l=50.0, hw=17.0, half=50.0, h=48.0, r=24.0)
CAN_RADIUS = 492.0       # 2.4 x Robo height / 2
CAN_DROP = 230.0         # apex -> rim drop
CAN_K = 0.11             # rim ellipse squash (seen slightly from below)
CAN_APEX_Y = -765.0      # default apex height (local px)
CAN_OFFSET = (470.0, 0.0)

ROBO_FACES = ("dots", "happy", "blink", "question", "loading", "glitch", "warm", "heart", "surprised", "scan")
R_ARM_PRESETS = ("rest", "wave", "point", "hold_handle", "salute", "flail")

_BM = {
    "dot": [" ## ", "####", "####", "####", "####", " ## "],
    "happy": ["  ###  ", " ## ## ", "##   ##", "#     #"],
    "blink": ["#     #", " ##### "],
    "question": [" ### ", "## ##", "   ##", "  ## ", "  #  ", "     ", "  #  "],
    "o": [" #### ", "##  ##", "#    #", "#    #", "##  ##", " #### "],
    "x": ["##   ##", " ## ## ", "  ###  ", " ## ## ", "##   ##"],
    "heart": [" ### ### ", "#########", "#########", "#########", " ####### ", "  #####  ", "   ###   ",
              "    #    "],
    "heart_s": [" ## ## ", "#######", "#######", " ##### ", "  ###  ", "   #   "],
}
SCR_COLS, SCR_ROWS, SCR_CELL = 26, 15, 5.6


def _stamp(cells, bm, cx, cy, col, inten=1.0):
    h = len(bm)
    w = len(bm[0])
    x0 = int(round(cx - (w - 1) / 2))
    y0 = int(round(cy - (h - 1) / 2))
    for j, row in enumerate(bm):
        for i, ch in enumerate(row):
            if ch == "#":
                cells[(x0 + i, y0 + j)] = (col, inten)


def _screen_cells(face, t, speak, look, seed=0):
    """Return ({(col,row): (colour, intensity)}, row_shift dict, extra dict)."""
    cells = {}
    shift = {}
    extra = {}
    lx = clamp(look[0], -1, 1) * 2.0
    ly = clamp(look[1], -1, 1) * 1.5
    ex = [7.0 + lx, 18.0 + lx]
    ey = 6.0 + ly
    col = R_LED
    if face == "dots":
        for e in ex:
            _stamp(cells, _BM["dot"], e, ey, col)
    elif face == "happy":
        for e in ex:
            _stamp(cells, _BM["happy"], e, ey, col)
    elif face == "blink":
        for e in ex:
            _stamp(cells, _BM["blink"], e, ey + 1, col)
    elif face == "question":
        for k, e in enumerate(ex):
            bob = round(0.6 * math.sin(t * 5 + k * 1.6))
            _stamp(cells, _BM["question"], e, ey + bob, col)
    elif face == "surprised":
        for e in ex:
            _stamp(cells, _BM["o"], e, ey, col)
    elif face == "warm":
        col = R_LED_WARM
        for e in ex:
            _stamp(cells, _BM["happy"], e, ey + 0.5, col)
        for e in ex:
            for dx in (-1.5, -0.5, 0.5, 1.5):
                cells[(int(round(e + dx + (-1.0 if e < 12 else 1.0) * 1.0)), 10)] = (R_LED_PINK, 0.85)
    elif face == "heart":
        pulse = 1.0 + 0.08 * math.sin(t * 7)
        _stamp(cells, _BM["heart"], 12.5, 6.5, "#FF6F91", 1.0)
        extra["pulse"] = pulse
    elif face == "loading":
        cx, cy, r = 12.5, 7.0, 4.2
        ph = t * 1.6
        for i in range(8):
            a = TAU * i / 8 - PI / 2
            trail = ((ph * 8 - i) % 8) / 8.0
            inten = 1.0 - trail * 0.85
            px, py = cx + r * math.cos(a), cy + r * math.sin(a)
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                cells[(int(math.floor(px)) + dx, int(math.floor(py)) + dy)] = (col, inten)
    elif face == "scan":
        for e in ex:
            _stamp(cells, _BM["dot"], e, ey, col, 0.75)
        y = (0.5 + 0.5 * math.sin(t * 3.2)) * (SCR_ROWS - 1)
        extra["scan_y"] = y
    elif face == "glitch":
        k = int(t * 14)
        for n, e in enumerate(ex):
            jx = round((_hash(k, n, seed) - 0.5) * 2)
            _stamp(cells, _BM["x"], e + jx, ey, col if _hash(k, n, 7) > 0.2 else "#FF6FD8")
        for i in range(26):
            if _hash(k, i, seed, 3) < 0.55:
                c_ = int(_hash(k, i, 11) * SCR_COLS)
                r_ = int(_hash(k, i, 13) * SCR_ROWS)
                cells[(c_, r_)] = ("#FFFFFF" if _hash(k, i, 17) > 0.6 else col, 0.35 + 0.5 * _hash(k, i, 19))
        for r_ in range(SCR_ROWS):
            if _hash(k, r_, 23) < 0.22:
                shift[r_] = int((_hash(k, r_, 29) - 0.5) * 8)
        extra["tear"] = (_hash(k, 31) * SCR_ROWS, 1 + 2 * _hash(k, 37))
    # speaking: waveform bar mouth
    if speak > 0.02 and face not in ("heart", "loading"):
        mcol = R_LED_WARM if face == "warm" else col
        for i in range(8):
            cc = 9 + i
            amp = speak * (0.55 + 0.45 * math.sin(t * 23 + i * 1.7)) * (1 - abs(i - 3.5) / 5.5)
            hh = max(0, int(round(amp * 3.2)))
            for r_ in range(-hh, hh + 1):
                cells[(cc, 12 + r_)] = (mcol, 0.9)
    return cells, shift, extra


def _draw_screen(ctx, w, h, face, t, glow, speak, look, seed, glow_only=False):
    """LED screen in local screen coords (origin centre, w x h)."""
    cell = SCR_CELL
    gx0 = -SCR_COLS * cell / 2
    gy0 = -SCR_ROWS * cell / 2
    if not glow_only:
        _rrect(ctx, -w / 2, -h / 2, w, h, 18)
        g = linear(0, -h / 2, 0, h / 2, [(0, "#24353F"), (1, R_GLASS)])
        ctx.set_source(g)
        ctx.fill()
    ctx.save()
    _rrect(ctx, -w / 2, -h / 2, w, h, 18)
    ctx.clip()
    if not glow_only:
        # faint unlit LED matrix
        ctx.new_path()
        for j in range(SCR_ROWS):
            for i in range(SCR_COLS):
                ctx.rectangle(gx0 + i * cell + 1.2, gy0 + j * cell + 1.2, cell - 2.4, cell - 2.4)
        _sc(ctx, "#2A3E4A", 0.55)
        ctx.fill()
    cells, shift, extra = _screen_cells(face, t, speak, look, seed)
    g = clamp(glow)
    pulse = extra.get("pulse", 1.0)
    if face in ("glitch",):
        g *= 0.75 + 0.25 * _hash(int(t * 30), 5)
    # halos (soft) then cores
    for (i, j), (col, inten) in cells.items():
        if not (0 <= i < SCR_COLS and 0 <= j < SCR_ROWS):
            continue
        cx = gx0 + (i + shift.get(j, 0)) * cell + cell / 2
        cy = gy0 + j * cell + cell / 2
        if pulse != 1.0:
            cx = cx * pulse
            cy = (cy - (6.5 - SCR_ROWS / 2 + 0.5) * cell) * pulse + (6.5 - SCR_ROWS / 2 + 0.5) * cell
        _sc(ctx, col, 0.16 * inten * g)
        ctx.rectangle(cx - cell * 1.15, cy - cell * 1.15, cell * 2.3, cell * 2.3)
        ctx.fill()
    for (i, j), (col, inten) in cells.items():
        if not (0 <= i < SCR_COLS and 0 <= j < SCR_ROWS):
            continue
        cx = gx0 + (i + shift.get(j, 0)) * cell + cell / 2
        cy = gy0 + j * cell + cell / 2
        if pulse != 1.0:
            cx = cx * pulse
            cy = (cy - (6.5 - SCR_ROWS / 2 + 0.5) * cell) * pulse + (6.5 - SCR_ROWS / 2 + 0.5) * cell
        a = (0.35 + 0.65 * g) * inten
        _sc(ctx, col, a)
        ctx.rectangle(cx - cell * 0.45, cy - cell * 0.45, cell * 0.9, cell * 0.9)
        ctx.fill()
        _sc(ctx, mix(col, "#FFFFFF", 0.6), a * 0.9)
        ctx.rectangle(cx - cell * 0.22, cy - cell * 0.22, cell * 0.44, cell * 0.44)
        ctx.fill()
    if "scan_y" in extra:
        y = gy0 + extra["scan_y"] * cell + cell / 2
        gr = linear(0, y - 18, 0, y + 2, [(0, R_LED, 0.0), (0.85, R_LED, 0.25 * g), (1, R_LED, 0.0)])
        ctx.rectangle(-w / 2, y - 18, w, 20)
        ctx.set_source(gr)
        ctx.fill()
        ctx.rectangle(-w / 2, y - 1.5, w, 3)
        _sc(ctx, mix(R_LED, "#FFFFFF", 0.5), 0.9 * g)
        ctx.fill()
    if face == "glitch" and not glow_only:
        ty_, th_ = extra["tear"]
        y = gy0 + ty_ * cell
        ctx.rectangle(-w / 2, y, w, th_ * cell * 0.6)
        _sc(ctx, "#FFFFFF", 0.25)
        ctx.fill()
        ctx.new_path()
        for j in range(0, int(h), 3):
            ctx.rectangle(-w / 2, -h / 2 + j, w, 1.2)
        _sc(ctx, "#000000", 0.25)
        ctx.fill()
    if not glow_only:
        # glass reflections
        ctx.new_path()
        ctx.move_to(-w * 0.42, -h / 2)
        ctx.line_to(-w * 0.18, -h / 2)
        ctx.line_to(-w * 0.42, h * 0.1)
        ctx.line_to(-w / 2, h * 0.1)
        ctx.line_to(-w / 2, -h / 2)
        ctx.close_path()
        ctx.set_source_rgba(1, 1, 1, 0.07)
        ctx.fill()
        ctx.move_to(-w * 0.12, -h / 2)
        ctx.line_to(-w * 0.06, -h / 2)
        ctx.line_to(-w * 0.34, h * 0.2)
        ctx.line_to(-w * 0.4, h * 0.2)
        ctx.close_path()
        ctx.set_source_rgba(1, 1, 1, 0.06)
        ctx.fill()
    ctx.restore()
    return cells


def linear(x0, y0, x1, y1, stops):
    g = cairo.LinearGradient(x0, y0, x1, y1)
    for s in stops:
        c = _c(s[1])
        g.add_color_stop_rgba(s[0], c[0], c[1], c[2], s[2] if len(s) > 2 else 1.0)
    return g


def _disc_cyl(P, center, axis, r, depth, n=20):
    """Points (2D) of a short cylinder (disc) for hull silhouettes; axis = unit 3D normal."""
    ax = axis
    # two perpendicular vectors
    if abs(ax[2]) < 0.9:
        u = (ax[1] * 0 - ax[2] * 1, ax[2] * 0 - ax[0] * 0, ax[0] * 1 - ax[1] * 0)
        u = (-ax[2], 0.0, ax[0]) if abs(ax[1]) < 0.9 else (1.0, 0.0, 0.0)
    else:
        u = (1.0, 0.0, 0.0)
    ln = math.sqrt(sum(x * x for x in u)) or 1
    u = tuple(x / ln for x in u)
    vv = (ax[1] * u[2] - ax[2] * u[1], ax[2] * u[0] - ax[0] * u[2], ax[0] * u[1] - ax[1] * u[0])
    front, backs = [], []
    for i in range(n):
        a = TAU * i / n
        q = [center[k] + r * (u[k] * math.cos(a) + vv[k] * math.sin(a)) for k in range(3)]
        q2 = [q[k] + ax[k] * depth for k in range(3)]
        backs.append(P.p3(q))
        front.append(P.p3(q2))
    return front, backs


def _r_hand(ctx, W, d, kind, lw, t=0.0, wet=0.0, side=1):
    """Robo's 3-finger clamp hand at wrist W pointing along d (2D unit)."""
    ang = math.atan2(d[1], d[0])
    met = _wet(R_METAL, wet, 0.4)
    ctx.save()
    ctx.translate(*W)
    ctx.rotate(ang)
    if kind == "grip":
        fingers = [(-0.55, 1.6, 0.9), (-0.15, 1.7, 0.95), (0.75, -1.5, 0.8)]
    elif kind == "point":
        fingers = [(-0.05, 0.05, 1.15), (-0.55, 1.6, 0.85), (0.75, -1.4, 0.8)]
    elif kind == "open_up" or kind == "wave":
        fingers = [(-0.6, 0.25, 1.0), (-0.05, 0.12, 1.05), (0.65, -0.25, 0.95)]
    else:  # relaxed open clamp
        fingers = [(-0.42, 0.45, 1.0), (-0.08, 0.35, 1.0), (0.55, -0.55, 0.9)]
    for a1, bend, ln in fingers:
        p0 = (8, 0)
        p1 = _add(p0, (math.cos(a1), math.sin(a1)), 12 * ln)
        p2 = _add(p1, (math.cos(a1 + bend * 0.75), math.sin(a1 + bend * 0.75)), 10 * ln)
        _cel(ctx, lambda c, a=(p0, p1, p2): _limb(c, list(a), [4.6, 4.0, 3.0]), met, R_METAL_SH, None,
             R_METAL_LINE, lw * 0.8, soff=(1.5, -1.5), shade_a=0.8)
        ctx.arc(p1[0], p1[1], 1.6, 0, TAU)
        _sc(ctx, R_METAL_LINE, 0.7)
        ctx.fill()
    # wrist cuff
    _rrect(ctx, -3, -9.5, 13, 19, 4)
    _sc(ctx, _wet(R_MINT, wet, 0.45))
    ctx.fill_preserve()
    _stroke(ctx, R_MINT_LINE, lw * 0.85)
    ctx.rectangle(1, -9.5, 2.2, 19)
    _sc(ctx, R_MINT_SH, 0.9)
    ctx.fill()
    ctx.restore()


def _bez(p0, p1, p2, p3, u):
    v = 1 - u
    return (v ** 3 * p0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u ** 3 * p3[0],
            v ** 3 * p0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u ** 3 * p3[1])


def _spark_bolt(ctx, p, dirv, L, k, lw_scale=1.0, col="#FFF6A0"):
    pts = [p]
    n = 4
    perp = (-dirv[1], dirv[0])
    for i in range(1, n + 1):
        u = i / n
        j = (_hash(k, i, 41) - 0.5) * L * 0.5
        pts.append(_add(_add(p, dirv, L * u), perp, j * (1 - u * 0.4)))
    for w, c, a in ((7.0, "#7FF4FF", 0.25), (3.2, col, 0.95), (1.3, "#FFFFFF", 1.0)):
        ctx.new_path()
        ctx.move_to(*pts[0])
        for q in pts[1:]:
            ctx.line_to(*q)
        _stroke(ctx, c, w * lw_scale, a)
    draw_sparkle(ctx, pts[-1][0], pts[-1][1], 5 * lw_scale, 0.9, 0.3, "#FFFBD0")
    # branch
    b0 = pts[2]
    b1 = _add(b0, _rot(dirv, 0.8 if _hash(k, 5) > 0.5 else -0.8), L * 0.35)
    ctx.move_to(*b0)
    ctx.line_to(*b1)
    _stroke(ctx, col, 1.6 * lw_scale, 0.9)


def _draw_canopy(ctx, base, p, offset, tilt, t, wet, lw, mirror_ls, glint=True):
    """Telescoping pole + 8-rib striped canopy.  base = pole root (local 2D). Returns dict (local)."""
    out = {}
    p = clamp(p)
    ext = ease_in_out(clamp((p - 0.03) / 0.27))
    s2 = clamp((p - 0.3) / 0.4)
    s3 = clamp((p - 0.7) / 0.3)
    apex_f = (offset[0], CAN_APEX_Y + offset[1])
    # pole path: up from Robo's back, bending over to arrive vertically under the apex
    P0 = base
    P1 = (base[0] - 10, lerp(base[1], apex_f[1], 0.75))
    P2 = (apex_f[0], apex_f[1] + 260)
    P3 = apex_f
    n = 26
    curve = [_bez(P0, P1, P2, P3, ext * i / n) for i in range(n + 1)]
    apex = curve[-1]
    tang = _norm(_sub(curve[-1], curve[-2])) if ext > 0.01 else (0.0, -1.0)
    # pole: 3 telescoping sections
    if ext > 0.005:
        segs = [(0.0, 0.42, 12.0, R_METAL_SH), (0.36, 0.74, 9.0, R_METAL), (0.68, 1.0, 6.5, R_METAL_HI)]
        for a, b, wdt, col in segs:
            i0, i1 = int(a * n), max(int(a * n) + 1, int(b * n))
            pts = curve[i0:i1 + 1]
            if len(pts) < 2:
                continue
            ctx.new_path()
            ctx.move_to(*pts[0])
            for q in pts[1:]:
                ctx.line_to(*q)
            _stroke(ctx, R_METAL_LINE, wdt + lw * 1.6)
            ctx.new_path()
            ctx.move_to(*pts[0])
            for q in pts[1:]:
                ctx.line_to(*q)
            _stroke(ctx, _wet(col, wet, 0.3), wdt)
            # collar at the joint
            ctx.arc(pts[0][0], pts[0][1], wdt * 0.75 + 1.5, 0, TAU)
            _sc(ctx, R_MINT)
            ctx.fill_preserve()
            _stroke(ctx, R_MINT_LINE, lw * 0.6)
    out["pole_tip"] = apex
    if p < 0.02:
        return out
    # canopy frame: along pole while folded, -> canopy_tilt when open
    pole_ang = math.atan2(tang[0], -tang[1])
    rot = lerp(pole_ang, tilt, ease_in_out(s2))
    fr = ease_out_back(s3, 2.6) if s3 > 0 else 0.0
    rs = max(1.0, fr)
    hsq = 1.0 - 0.6 * (rs - 1.0)
    u_max = min(1.0, fr)
    R, Hh = CAN_RADIUS * rs, CAN_DROP * hsq
    cr, sr = math.cos(rot), math.sin(rot)

    def to2(X, Y, Z):
        x = X
        y = Y - CAN_K * Z
        return (apex[0] + x * cr - y * sr, apex[1] + x * sr + y * cr)

    ribs = []
    for i in range(8):
        th = PI / 8 + i * PI / 4
        st = clamp((s2 - i * 0.025) / 0.8)
        e = ease_in_out(st)
        alpha_f = math.atan2(R, Hh)
        al = lerp(0.07, alpha_f, e)
        lam = lerp(0.3, 1.0, ease_out(clamp(st * 1.2)))
        Lr = math.hypot(R, Hh) * lam
        dirv = (math.sin(al) * math.cos(th), math.cos(al), math.sin(al) * math.sin(th))
        pw = lerp(1.0, 1.7, e)

        def rib_pt(u, dirv=dirv, Lr=Lr, pw=pw):
            # dome profile: flat crown, steeper toward the rim (no dip at the apex)
            X, Y, Z = (dirv[0] * Lr * u, dirv[1] * Lr * (u ** pw), dirv[2] * Lr * u)
            return to2(X, Y, Z)

        ribs.append((th, rib_pt, st))
    tips = [r[1](1.0) for r in ribs]
    out["rib_tips"] = tips
    teal, cream = _wet(CAN_TEAL, wet, 0.3), _wet(CAN_CREAM, wet, 0.25)
    if s3 > 0:
        # ---- underside (interior), seen from below
        def edge(i, umax=u_max):
            th0, rp0, _ = ribs[i]
            th1, rp1, _ = ribs[(i + 1) % 8]
            a, b = rp0(umax), rp1(umax)
            mid = _l2(a, b, 0.5)
            sag = _l2(mid, apex, 0.05)
            return a, sag, b

        for i in range(8):
            a, sag, b = edge(i)
            ctx.new_path()
            ctx.move_to(*apex)
            ctx.line_to(*a)
            ctx.curve_to(*_l2(a, sag, 0.7), *_l2(b, sag, 0.7), *b)
            ctx.close_path()
            _sc(ctx, CAN_TEAL_SH if i % 2 == 0 else CAN_CREAM_SH)
            ctx.fill_preserve()
            _stroke(ctx, CAN_LINE, lw * 0.6, 0.6)
        # interior ribs (back half visible)
        for i, (th, rp, st) in enumerate(ribs):
            if math.sin(th) < 0:
                pts = [rp(u * u_max) for u in (0, 0.25, 0.5, 0.75, 1.0)]
                ctx.new_path()
                _spline(ctx, pts, False)
                _stroke(ctx, R_METAL_LINE, 2.4, 0.8)
        # shade of underside interior near apex
        # ---- exterior panels (front half), back to front
        order = sorted(range(8), key=lambda i: math.sin(ribs[i][0] + PI / 8))
        for i in order:
            mid_th = ribs[i][0] + PI / 8
            if math.sin(mid_th) < -0.05:
                continue
            a, sag, b = edge(i)
            rp0, rp1 = ribs[i][1], ribs[(i + 1) % 8][1]
            l0 = [rp0(u * u_max) for u in (0, 0.2, 0.45, 0.7, 0.88, 1.0)]
            l1 = [rp1(u * u_max) for u in (1.0, 0.88, 0.7, 0.45, 0.2, 0)]
            col = teal if i % 2 == 0 else cream
            sh = CAN_TEAL_SH if i % 2 == 0 else CAN_CREAM_SH
            # light from upper-left: panels facing left brighter
            facing_left = -math.cos(mid_th) * mirror_ls
            colp = mix(col, sh, clamp(0.35 - 0.45 * facing_left))

            def panel(c, l0=l0, l1=l1, a=a, b=b, sag=sag):
                _spline(c, l0, False)
                c.curve_to(*_l2(a, sag, 0.7), *_l2(b, sag, 0.7), *b)
                _spline(c, l1, False, move=False)
                c.close_path()

            def pextra(c, l0=l0, l1=l1, i=i):
                # glossy rain sheen band
                m0, m1 = l0[2], l1[3]
                c.new_path()
                c.move_to(*_l2(l0[1], l1[4], 0.15))
                c.line_to(*_l2(l0[1], l1[4], 0.45))
                c.line_to(*_l2(m0, m1, 0.5))
                c.line_to(*_l2(m0, m1, 0.15))
                c.close_path()
                c.set_source_rgba(1, 1, 1, 0.16 + 0.1 * wet)
                c.fill()

            _cel(ctx, panel, colp, None, None, CAN_LINE, lw * 0.9, extra=pextra)
        # finial
        top = to2(0, -0.35 * Hh * 0.2, 0)
        ctx.arc(apex[0], apex[1] - 3, 7, 0, TAU)
        _sc(ctx, R_MINT)
        ctx.fill_preserve()
        _stroke(ctx, CAN_LINE, lw * 0.7)
        # rib tip ferrules on visible rim
        for i, (th, rp, st) in enumerate(ribs):
            if math.sin(th) > -0.4:
                q = rp(u_max)
                ctx.arc(q[0], q[1], 4.2, 0, TAU)
                _sc(ctx, CAN_CREAM)
                ctx.fill_preserve()
                _stroke(ctx, CAN_LINE, lw * 0.6)
        rim_pts = []
        for i in range(8):
            a, sag, b = edge(i)
            rim_pts += [a, sag]
        out["rim"] = rim_pts
    else:
        # skeleton: stretchers + ribs (furled bundle while folded)
        for i in sorted(range(8), key=lambda i: math.sin(ribs[i][0])):
            th, rp, st = ribs[i]
            pts = [rp(u) for u in (0, 0.25, 0.5, 0.75, 1.0)]
            runner = _add(apex, (sr * -1 * 0 + -sr * 0, 0), 0)
            run = to2(0, 0.38 * math.hypot(R, Hh) * lerp(0.3, 1, st) * math.cos(0.07 + st), 0)
            ctx.new_path()
            ctx.move_to(*run)
            ctx.line_to(*rp(0.45))
            _stroke(ctx, R_METAL_SH, 2.2)
            ctx.new_path()
            _spline(ctx, pts, False)
            _stroke(ctx, R_METAL_LINE, 5.4)
            ctx.new_path()
            _spline(ctx, pts, False)
            _stroke(ctx, R_METAL_HI if math.sin(th) > 0 else R_METAL, 3.0)
            q = pts[-1]
            ctx.arc(q[0], q[1], 3.6, 0, TAU)
            _sc(ctx, CAN_TEAL)
            ctx.fill_preserve()
            _stroke(ctx, CAN_LINE, lw * 0.5)
        if s2 < 0.35:
            # furled fabric bundle hugging the folded ribs
            k = 1 - s2 / 0.35
            L0 = math.hypot(R, Hh) * 0.3
            a = to2(-16 * k, 8, 0)
            b = to2(16 * k, 8, 0)
            c_ = to2(0, L0 * 0.98, 0)
            ctx.new_path()
            ctx.move_to(*apex)
            ctx.curve_to(*a, *_l2(a, c_, 0.6), *c_)
            ctx.curve_to(*_l2(b, c_, 0.6), *b, *apex)
            ctx.close_path()
            _sc(ctx, teal, k)
            ctx.fill_preserve()
            _stroke(ctx, CAN_LINE, lw * 0.8, k)
            st_ = to2(0, L0 * 0.55, 0)
            ctx.arc(st_[0], st_[1], 5, 0, TAU)
            _sc(ctx, CAN_CREAM, k)
            ctx.fill()
        ctx.arc(apex[0], apex[1], 8, 0, TAU)
        _sc(ctx, R_MINT)
        ctx.fill_preserve()
        _stroke(ctx, R_MINT_LINE, lw * 0.7)
        out["rim"] = tips
    # clamps lock: glints
    if glint and 0.6 < p < 0.82:
        g = math.sin(clamp((p - 0.6) / 0.22) * PI)
        for i, q in enumerate(tips):
            if math.sin(ribs[i][0]) > -0.3:
                draw_sparkle(ctx, q[0], q[1], 16 * g, g, 0.3 * i, "#FFFFFF")
    out["apex"] = apex
    xs = [q[0] for q in tips]
    out["center"] = (sum(xs) / 8, sum(q[1] for q in tips) / 8)
    out["radius"] = R * u_max if s3 > 0 else 0.0
    out["span"] = (min(xs), max(xs))
    return out


def draw_robo(ctx, x, y, scale=1.0, *, view="front", yaw=None, face="dots", screen_glow=1.0, speak=0.0,
              look=(0.0, 0.0), body_tilt=0.0, head_tilt=0.0, bob=0.0, squash=0.0, tread_phase=0.0,
              antenna_blink=True, arms="rest", arm_phase=None, hand=None, hand_l=None, hand_r=None,
              wet=0.0, sparks=0.0, mini_umbrella=0.0, canopy=0.0, canopy_offset=CAN_OFFSET, canopy_tilt=0.0,
              t=0.0, seed=7, rim=RIM, light="left", layer="all", glow_only=False, alpha=1.0):
    """Draw Robo (No.7).  (x, y) = ground contact centre; scale 1 -> ~410 px tall (antenna bulb top).

    view : front | three_quarter_left | three_quarter_right | side_left | side_right (or yaw rad).
    face : dots happy blink question loading glitch warm heart surprised scan  (LED screen, uses t)
    screen_glow 0..1 (LED brightness + spill), speak 0..1 (pixel waveform mouth; feed core.mouth('robo')[0]),
    look (dx, dy) -1..1 moves the LED eyes.
    body_tilt (rad, rocks above the treads), head_tilt (rad), bob (px, + = down), squash (+ squash/- stretch),
    tread_phase (lug scroll, 1 = one lug pitch... use distance/9), antenna_blink: True (auto, uses t) |
      False (off) | float 0..1 (bulb brightness).
    arms : rest | wave | point | hold_handle | salute | flail ; hand = 'screen_left'|'screen_right' for
      one-handed poses; arm_phase (default t); hand_l / hand_r wrist override (l, f, v) body space
      (shoulders at (+-88, 0, -186)).
    wet 0..1 darker paint, drips, streaks.  sparks 0..1 electric sparks from seams (seeded by t, seed).
    mini_umbrella 0..1 hatch opens (0-.2), spring shoots up with overshoot (.12-.62), tiny red umbrella
      opens (.55-1); idle wobble from t.
    canopy 0..1 transformation (0-.3 back panel splits + telescoping pole, .3-.7 8 ribs unfold, .7-1 fabric
      blooms with bounce).  canopy_offset (dx, dy) px*scale of the canopy apex relative to the default
      spot (default (470, 0): toward screen-right/facing direction, leaving Robo half out in the rain).
      canopy_tilt (rad).
    layer : 'all' | 'body' (no canopy/pole) | 'canopy' (only pole + canopy) -- for layering rain between.
    glow_only : draw only emissive parts (LEDs, bulb) -- see robo_glow_layer().
    Returns anchors: screen_center, screen_size, head, head_top, antenna, hatch, umbrella_top (mini),
      umbrella_radius, hand_l, hand_r, hands, handle (grip point for hold_handle), chest, belly, feet,
      tread_l, tread_r, canopy_apex, canopy_center, canopy_radius, canopy_span (x0, x1), canopy_rim_points,
      drop_points, facing, scale.
    """
    anchor, to_local = _anchor_fn(ctx)
    ya, mirror = _resolve_view(view, yaw)
    ya = min(ya, PI / 2)
    ls = (1 if light == "left" else -1) * mirror
    look = (look[0] * mirror, look[1])
    body_tilt *= mirror
    head_tilt *= mirror
    P = _P(ya, 0.10)
    c, s = P.c, P.s
    if arm_phase is None:
        arm_phase = t
    lw = line_weight(scale)
    wet = clamp(wet)
    mint = _wet(R_MINT, wet, 0.55)
    mint_sh = _wet(R_MINT_SH, wet, 0.55)
    cream = _wet(R_CREAM, wet, 0.45)
    anchors = {"scale": scale, "facing": 0 if s < 0.3 else mirror}
    drops = []
    if isinstance(antenna_blink, bool):
        bulb = (0.25 + 0.75 * (1.0 if (t % 1.3) < 0.55 else 0.0)) if antenna_blink else 0.0
    else:
        bulb = clamp(float(antenna_blink))

    if alpha < 1.0:
        ctx.push_group()
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale * mirror, scale)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    # ------------------------------------------------ canopy (behind everything)
    if canopy > 0 and layer in ("all", "canopy") and not glow_only:
        base = P.p(0, -64, -176)
        base = (base[0], base[1] + bob)
        cv = _draw_canopy(ctx, base, canopy, canopy_offset, canopy_tilt * mirror, t, wet, lw, ls)
        anchors["canopy_apex"] = anchor(*cv["apex"])
        if "center" in cv:
            anchors["canopy_center"] = anchor(*cv["center"])
            anchors["canopy_radius"] = cv["radius"] * scale
            x0, x1 = anchor(cv["span"][0], 0)[0], anchor(cv["span"][1], 0)[0]
            anchors["canopy_span"] = (min(x0, x1), max(x0, x1))
            anchors["canopy_rim_points"] = [anchor(*q) for q in cv.get("rim", [])]
    if layer == "canopy":
        ctx.restore()
        if alpha < 1.0:
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(alpha)
        return anchors

    # ------------------------------------------------ treads
    T = RB_TREAD

    def tread_pts(l):
        pts = []
        for i in range(24):
            a = TAU * i / 24
            # stadium: two semicircles at f = +-(half - r)
            fc = (T["half"] - T["r"]) * (1 if math.cos(a) >= 0 else -1)
            pts.append((l, fc + T["r"] * math.cos(a), -T["r"] + T["r"] * math.sin(a) * (T["h"] / (2 * T["r"]))))
        return pts

    def draw_tread(side):
        l0 = side * T["l"]
        inner = tread_pts(l0 - T["hw"])
        outer = tread_pts(l0 + T["hw"])
        hull = _hull([P.p3(q) for q in inner + outer])
        tc = _wet(R_TREAD, wet, 0.3)
        ctx.new_path()
        _rpoly(ctx, hull, 6)
        hp = ctx.copy_path()
        _sc(ctx, tc)
        ctx.fill()
        ctx.save()
        ctx.append_path(hp)
        ctx.clip()
        # visible side face (stadium) -- facing camera when its normal points at us
        nside = side  # outer face normal +side (l)
        face_l = l0 + T["hw"] * (1 if (-side * s) > 0 else -1) * side * side
        vis_out = (-s * side) > 0.05
        fl = l0 + (T["hw"] if vis_out else -T["hw"])
        if s > 0.12:
            fpts = [P.p3(q) for q in tread_pts(fl)]
            ctx.new_path()
            _poly(ctx, fpts)
            _sc(ctx, R_TREAD_SH if not vis_out else mix(tc, R_TREAD_HI, 0.25))
            ctx.fill()
            # wheels
            for fw in (-T["half"] + T["r"], 0.0, T["half"] - T["r"]):
                ww = [P.p(fl, fw + 15 * math.cos(a), -T["r"] + 15 * math.sin(a)) for a in
                      [TAU * k / 16 for k in range(16)]]
                ctx.new_path()
                _poly(ctx, ww)
                _sc(ctx, R_METAL_SH)
                ctx.fill_preserve()
                _stroke(ctx, R_TREAD_LINE, lw * 0.6)
                hub = P.p(fl, fw, -T["r"])
                ctx.arc(hub[0], hub[1], 4.5, 0, TAU)
                _sc(ctx, R_METAL_HI)
                ctx.fill()
                # spokes rotate with tread_phase
                for k in range(3):
                    a = tread_phase * 2.2 + k * TAU / 3
                    q = P.p(fl, fw + 11 * math.cos(a), -T["r"] + 11 * math.sin(a))
                    ctx.move_to(*hub)
                    ctx.line_to(*q)
                _stroke(ctx, R_TREAD_LINE, 1.6)
        # lugs across the belt
        perim = 2 * (2 * (T["half"] - T["r"])) + TAU * T["r"]
        n_l = 22
        for k in range(n_l):
            sp_ = ((k + tread_phase) / n_l) % 1.0 * perim
            # walk the stadium perimeter: bottom straight -> front arc -> top straight -> back arc
            L1 = 2 * (T["half"] - T["r"])
            L2 = PI * T["r"]
            hc = T["h"] / (2 * T["r"])
            if sp_ < L1:
                f, v, nf, nv = -(T["half"] - T["r"]) + sp_, 0.0, 0.0, 1.0
            elif sp_ < L1 + L2:
                a = PI / 2 - (sp_ - L1) / T["r"]
                f, v = (T["half"] - T["r"]) + T["r"] * math.cos(a), -T["r"] + T["r"] * math.sin(a) * hc
                nf, nv = math.cos(a), math.sin(a)
            elif sp_ < 2 * L1 + L2:
                f, v, nf, nv = (T["half"] - T["r"]) - (sp_ - L1 - L2), -T["h"], 0.0, -1.0
            else:
                a = -PI / 2 - (sp_ - 2 * L1 - L2) / T["r"]
                f, v = -(T["half"] - T["r"]) + T["r"] * math.cos(a), -T["r"] + T["r"] * math.sin(a) * hc
                nf, nv = math.cos(a), math.sin(a)
            vis = nf * c - nv * 0.1
            if vis > 0.05:
                a2 = P.p(l0 - T["hw"] + 1, f, v)
                b2 = P.p(l0 + T["hw"] - 1, f, v)
                ctx.move_to(*a2)
                ctx.line_to(*b2)
        _stroke(ctx, R_TREAD_HI, 2.2, 0.9)
        ctx.restore()
        ctx.new_path()
        ctx.append_path(hp)
        _stroke(ctx, R_TREAD_LINE, lw)
        anchors["tread_l" if side == 1 else "tread_r"] = anchor(*P.p(l0, T["half"] * 0.6, 0))

    # upper-body frame: bob / squash / tilt around the body bottom
    pivot = (0.0, -40.0)

    def upper_begin():
        ctx.save()
        ctx.translate(pivot[0], pivot[1] + bob)
        ctx.rotate(body_tilt)
        if squash:
            sq = clamp(squash, -0.5, 0.5)
            ctx.scale(1 + sq * 0.4, 1 - sq * 0.6)
        ctx.translate(-pivot[0], -pivot[1])

    # ------------------------------------------------ arms (IK)
    def screen_side_to_anat(h):
        want = (1 if h == "screen_right" else -1) * mirror
        xs = {sd: P.p(sd * 30, 0, 0)[0] for sd in (-1, 1)}
        return 1 if (xs[1] - xs[-1]) * want > 0 else -1

    one = screen_side_to_anat(hand or "screen_right")
    if s > 0.3 and hand is None:
        one = -1

    def target(sd):
        rest = ((sd * 100, 24, -114 + 2 * math.sin(t * 2 + sd)), (sd * 0.5, -1, 0), "open")
        if arms == "rest":
            return rest
        if arms == "wave":
            if sd == one:
                wv = math.sin(arm_phase * 2.4 * TAU)
                return ((sd * (126 + 16 * wv), 12, -296 + 4 * abs(wv)), (sd, -0.3, 0.6), "wave")
            return rest
        if arms == "point":
            if sd == one:
                dirv = (c, s)
                return ((sd * 70 + dirv[0] * 70, 30 + dirv[1] * 90, -206), (0, 0, 1), "point")
            return rest
        if arms == "hold_handle":
            if sd == one:
                return ((sd * 40, 68, -170), (sd, -0.5, 0.4), "grip")
            return rest
        if arms == "salute":
            if sd == one:
                return ((sd * 92, 46, -318), (sd, 0, 0.5), "open_up")
            return rest
        if arms == "flail":
            ph = arm_phase * 9 + (0 if sd > 0 else 1.7)
            return ((sd * (112 + 18 * math.sin(ph * 1.3)), 20 + 20 * math.cos(ph), -250 + 70 * math.sin(ph)),
                    (sd, 0, 0.4), "wave")
        return rest

    arm_info = {}
    for sd in (-1, 1):
        S = (sd * RB_SHOULDER[0], 0.0, RB_SHOULDER[1])
        ov = hand_l if sd == 1 else hand_r
        if ov is not None:
            tg = (tuple(ov), (sd, -0.5, 0.3), "open")
        else:
            tg = target(sd)
        Wt, pole, kind = tg
        E, Wr = _ik(list(S), list(Wt), RB_UPPER, RB_FORE, pole)
        arm_info[sd] = dict(S=S, E=E, W=Wr, kind=kind, z=P.z(S[0], S[1] + 6) + 0.5 * P.z(Wr[0], Wr[1]))

    def draw_arm(sd):
        a = arm_info[sd]
        S2, E2, W2 = P.p3(a["S"]), P.p3(a["E"]), P.p3(a["W"])
        if _len(_sub(E2, S2)) < 5:
            E2 = (E2[0] + sd * 5, E2[1] + 3)
        pts = _sample_spline([S2, E2, W2], False, 6)
        # accordion bellows: alternating radii
        n = len(pts)
        rad = [10.5 + (2.6 if i % 2 == 0 else 0.0) for i in range(n)]
        met = _wet(R_METAL, wet, 0.4)
        far = a["z"] < -5
        if far:
            met = mix(met, R_METAL_SH, 0.35)
        _cel(ctx, lambda c_: _limb(c_, pts, rad), met, R_METAL_SH, None, R_METAL_LINE, lw * 0.85,
             soff=(2.5 * ls, -2), shade_a=0.8)
        # ridge lines
        _sc(ctx, R_METAL_SH, 0.9)
        for i in range(1, n - 1, 2):
            d = _norm(_sub(pts[min(i + 1, n - 1)], pts[i - 1]))
            nn = (-d[1], d[0])
            ctx.move_to(*_add(pts[i], nn, 10))
            ctx.curve_to(*_add(_add(pts[i], nn, 4), d, 2.5), *_add(_add(pts[i], nn, -4), d, 2.5),
                         *_add(pts[i], nn, -10))
        ctx.set_line_width(1.2)
        ctx.stroke()
        d = _norm(_sub(W2, pts[-3])) if n > 3 else (0, 1)
        kind = a["kind"]
        if kind == "open_up":
            d = _norm((d[0] * 0.3, -1))
        if kind == "wave":
            d = _norm((d[0] + 0.3 * math.sin(arm_phase * 2.4 * TAU), d[1]))
        _r_hand(ctx, W2, d, kind, lw, t, wet, sd)
        # shoulder cap
        ctx.arc(S2[0], S2[1], 16, 0, TAU)
        _sc(ctx, mint if not far else mint_sh)
        ctx.fill_preserve()
        _stroke(ctx, R_MINT_LINE, lw * 0.9)
        ctx.arc(S2[0] - 3 * ls, S2[1] - 3, 3.0, 0, TAU)
        _sc(ctx, R_MINT_HI, 0.9)
        ctx.fill()
        anchors["hand_l" if sd == 1 else "hand_r"] = anchor(*W2)
        if kind == "grip":
            anchors["handle"] = anchor(*_add(W2, d, 14))

    # ------------------------------------------------ body
    def surf(v, phi, k=1.0):
        w, d = _interp_rings(RB_BODY, v)
        l = w * math.sin(phi) * k
        f = d * math.cos(phi) * k
        px, py = P.p(l, f, v)
        return (px, py, P.z(l, f) / max(d, 1))

    def body():
        L, Rr = [], []
        for v, w, d in RB_BODY:
            cx_, e = _ring_ext(P, w, d)
            L.append((cx_ - e, v))
            Rr.append((cx_ + e, v))
        bot = P.p(0, 0, -24)
        topp = P.p(0, 0, -242)
        pts = [topp] + Rr + [bot] + L[::-1]
        bp = lambda c_: _spline(c_, pts, True)

        def extra(c_):
            # cream belly panel
            if c > -0.2:
                pan = []
                for i in range(20):
                    a = TAU * i / 20
                    v = -122 + 50 * math.sin(a)
                    ph = 0.66 * math.cos(a)
                    pan.append(surf(v, ph, 1.0))
                if max(q[2] for q in pan) > 0:
                    c_.new_path()
                    _spline(c_, [(q[0], q[1]) for q in pan], True)
                    _sc(c_, cream)
                    c_.fill_preserve()
                    path = c_.copy_path()
                    c_.save()
                    c_.clip()
                    c_.new_path()
                    _spline(c_, [(q[0] - 6 * ls, q[1] - 5) for q in pan], True)
                    c_.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
                    c_.rectangle(-1e4, -1e4, 2e4, 2e4)
                    _sc(c_, _wet(R_CREAM_SH, wet, 0.45))
                    c_.fill()
                    c_.set_fill_rule(cairo.FILL_RULE_WINDING)
                    # stencilled 7
                    ctr = surf(-120, 0.0)
                    if c > 0.12:
                        c_.save()
                        c_.translate(ctr[0], ctr[1])
                        c_.transform(cairo.Matrix(c, -0.1 * s, 0, 1, 0, 0))
                        c_.scale(mirror, 1)
                        _sc(c_, R_STENCIL, 0.88)
                        c_.new_path()
                        c_.move_to(-17, -28)
                        c_.line_to(17, -28)
                        c_.line_to(17, -20)
                        c_.line_to(-1, 30)
                        c_.line_to(-12, 30)
                        c_.line_to(5, -19)
                        c_.line_to(-17, -19)
                        c_.close_path()
                        c_.fill()
                        # stencil bridges
                        c_.rectangle(-6, -29, 2.6, 11)
                        c_.rectangle(1, 2, 9, 2.6)
                        _sc(c_, cream)
                        c_.fill()
                        c_.restore()
                    c_.restore()
                    c_.new_path()
                    c_.append_path(path)
                    _stroke(c_, R_CREAM_LINE, lw * 0.8)
                    # rivets around the panel
                    for i in range(0, 20, 2):
                        q = pan[i]
                        if q[2] > 0.15:
                            qq = _l2((q[0], q[1]), surf(-122, 0)[:2], -0.09)
                            c_.arc(qq[0], qq[1], 2.3, 0, TAU)
                            _sc(c_, mint_sh)
                            c_.fill()
                            c_.arc(qq[0] - 0.7, qq[1] - 0.7, 0.9, 0, TAU)
                            _sc(c_, R_MINT_HI)
                            c_.fill()
            # side "No.7" in side views
            if s > 0.5:
                sp = surf(-150, -PI / 2 * 1.0)
                c_.save()
                c_.translate(sp[0] + 0.0, sp[1])
                c_.scale((0.35 + 0.65 * s) * mirror, 1)
                _sc(c_, R_STENCIL, 0.75)
                c_.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
                c_.set_font_size(15)
                c_.move_to(-17, 5)
                c_.show_text("No.7")
                c_.restore()
            # horizontal seam near the top + bottom
            for v in (-214, -60):
                c_.new_path()
                first = True
                for i in range(15):
                    ph = -ya + PI / 2 - PI * i / 14
                    q = surf(v, ph)
                    (c_.move_to if first else c_.line_to)(q[0], q[1])
                    first = False
                _stroke(c_, R_MINT_LINE, lw * 0.55, 0.55)
            # rust spots
            for i, (v, ph, rr) in enumerate(((-205, -0.9, 6), (-70, 0.85, 8), (-168, 1.15, 5), (-52, -0.5, 7),
                                               (-110, -1.35, 5), (-196, 0.5, 4))):
                q = surf(v, ph)
                if q[2] < 0.1:
                    continue
                fs = 0.35 + 0.65 * q[2]
                blob = []
                for k in range(9):
                    a = TAU * k / 9
                    r2 = rr * (0.7 + 0.5 * _hash(i, k, 3))
                    blob.append((q[0] + r2 * math.cos(a) * fs, q[1] + r2 * math.sin(a) * 0.8))
                c_.new_path()
                _spline(c_, blob, True)
                _sc(c_, R_RUST, 0.85)
                c_.fill()
                _sc(c_, R_RUST_DK, 0.7)
                c_.arc(q[0] + 1, q[1] + 1, rr * 0.35, 0, TAU)
                c_.fill()
                _sc(c_, R_RUST, 0.5)
                _taper(c_, [(q[0], q[1] + rr * 0.6), (q[0] + 0.5, q[1] + rr * 2.2)], 1.6, 0.4)
            # gloss highlight on the capsule
            hx = -ls * 0.55
            q0 = surf(-200, hx * 1.2)
            q1 = surf(-90, hx * 1.25)
            _sc(c_, R_MINT_HI, 0.55 * (1 - 0.3 * wet))
            _taper(c_, [(q0[0], q0[1]), _l2(q0[:2], q1[:2], 0.5), (q1[0], q1[1])], 3, 2, 7)

        _cel(ctx, bp, mint, mint_sh, rim, R_MINT_LINE, lw, soff=(-14 * ls, -4), roff=(-3 * ls, 0.5), rim_a=0.55,
             extra=extra)

    def back_doors():
        if canopy <= 0:
            return
        op = smoothstep(0.0, 0.1, canopy)
        for sd in (-1, 1):
            hinge_l = sd * 34
            ang = op * 1.7
            pts3 = []
            for v in (-120, -215):
                pts3.append((hinge_l, -66, v))
            outer = [(hinge_l + sd * 40 * math.cos(ang), -66 - 40 * math.sin(ang), v) for v in (-215, -120)]
            quad = [P.p3(q) for q in pts3 + outer]
            ctx.new_path()
            _rpoly(ctx, quad, 4)
            _sc(ctx, mint_sh)
            ctx.fill_preserve()
            _stroke(ctx, R_MINT_LINE, lw * 0.8)

    # ------------------------------------------------ head
    Hd = RB_HEAD

    def rr_pts(hw, top, bot, r, f, n=6):
        """Rounded rect in the l-v plane at depth f (3D points)."""
        pts = []
        cs = [(hw - r, top + r, -PI / 2, 0), (hw - r, bot - r, 0, PI / 2), (-hw + r, bot - r, PI / 2, PI),
              (-hw + r, top + r, PI, 1.5 * PI)]
        for lc, vc, a0, a1 in cs:
            for i in range(n + 1):
                a = lerp(a0, a1, i / n)
                pts.append((lc + r * math.cos(a), f, vc + r * math.sin(a)))
        return pts

    def head():
        hw, hd, top, bot, r = Hd["hw"], Hd["hd"], Hd["top"], Hd["bot"], Hd["r"]
        rd = 15.0
        levels = []
        for k in range(5):
            al = PI / 2 * k / 4
            e = rd * (1 - math.cos(al))
            f = hd - rd + rd * math.sin(al)
            for sg in (1, -1):
                levels += rr_pts(hw - e, top + e, bot - e, max(4.0, r - e), sg * f, 5)
        front = rr_pts(hw - rd, top + rd, bot - rd, r - rd * 0.6, hd)
        hull = _hull([P.p3(q) for q in levels])
        ctx.new_path()
        _poly(ctx, hull)
        hpath = ctx.copy_path()
        _sc(ctx, mint_sh)
        ctx.fill()
        ctx.save()
        ctx.append_path(hpath)
        ctx.clip()
        # top face (rounded rect in l-f plane)
        tp = []
        for lc, fc, a0, a1 in ((hw - r, hd - r, 0, PI / 2), (-hw + r, hd - r, PI / 2, PI),
                               (-hw + r, -hd + r, PI, 1.5 * PI), (hw - r, -hd + r, 1.5 * PI, TAU)):
            for i in range(7):
                a = lerp(a0, a1, i / 6)
                tp.append(P.p(lc + (r - rd) * math.cos(a), fc + (r - rd) * math.sin(a), top))
        ctx.new_path()
        _poly(ctx, tp)
        _sc(ctx, mix(mint, R_MINT_HI, 0.35))
        ctx.fill()
        # side-face shading gradient toward the back
        if c > 0.02:
            fp = [P.p3(q) for q in front]
            ctx.new_path()
            _poly(ctx, fp)
            _sc(ctx, mint)
            ctx.fill_preserve()
            fpath = ctx.copy_path()
            # cel shadow on the front face (light side)
            ctx.save()
            ctx.clip()
            ctx.new_path()
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            ctx.rectangle(-1e4, -1e4, 2e4, 2e4)
            ctx.translate(-10 * ls * c, -6)
            ctx.append_path(fpath)
            _sc(ctx, mint_sh, 0.65)
            ctx.fill()
            ctx.restore()
            ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
            ctx.new_path()
            ctx.append_path(fpath)
            _stroke(ctx, R_MINT_LINE, lw * 0.6, 0.6)
        else:
            fpath = None
        # side face details: ear disc on the visible side
        ctx.restore()
        # rim + outline
        ctx.save()
        ctx.append_path(hpath)
        ctx.clip()
        ctx.new_path()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(-1e4, -1e4, 2e4, 2e4)
        ctx.translate(-3 * ls, 0.5)
        ctx.append_path(hpath)
        if rim:
            _sc(ctx, rim, 0.5)
            ctx.fill()
        ctx.restore()
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        ctx.new_path()
        ctx.append_path(hpath)
        _stroke(ctx, R_MINT_LINE, lw)
        # ear discs
        for sd in (-1, 1):
            nrm = (sd, 0.0, 0.0)
            ctr = (sd * hw, 0.0, (top + bot) / 2 + 4)
            fr_, bk_ = _disc_cyl(P, ctr, nrm, 22, 9)
            vis = -s * sd
            if vis < -0.05 and c > 0.3:
                pass
            hl_ = _hull(fr_ + bk_)
            if vis > -0.05 or c > 0.85:
                ctx.new_path()
                _poly(ctx, hl_)
                _sc(ctx, mint_sh)
                ctx.fill_preserve()
                _stroke(ctx, R_MINT_LINE, lw * 0.8)
                if vis > 0.08:
                    ctx.new_path()
                    _poly(ctx, fr_)
                    _sc(ctx, cream)
                    ctx.fill_preserve()
                    _stroke(ctx, R_MINT_LINE, lw * 0.7)
                    cc = P.p(sd * (hw + 9), 0, ctr[2])
                    ctx.arc(cc[0], cc[1], 6, 0, TAU)
                    _sc(ctx, R_METAL)
                    ctx.fill_preserve()
                    _stroke(ctx, R_METAL_LINE, lw * 0.6)
                    # vent slots
                    for k in (-1, 1):
                        a2 = P.p(sd * (hw + 9), -10 * k * 0 + 12 * k * 0.0 - 2, ctr[2] + 13 * k)
                        b2 = P.p(sd * (hw + 9), 8 - 2, ctr[2] + 13 * k)
                        ctx.move_to(*a2)
                        ctx.line_to(*b2)
                    _stroke(ctx, R_MINT_LINE, 1.6, 0.6)
        # screen
        if c > 0.06:
            ctr = P.p(0, hd, (top + bot) / 2 + 1)
            sw_, sh_ = 150.0, 88.0
            ctx.save()
            ctx.translate(*ctr)
            ctx.transform(cairo.Matrix(c, -0.1 * s, 0, 1, 0, 0))
            # bezel
            _rrect(ctx, -sw_ / 2 - 6, -sh_ / 2 - 6, sw_ + 12, sh_ + 12, 22)
            _sc(ctx, mix(mint_sh, R_MINT_LINE, 0.25))
            ctx.fill()
            _draw_screen(ctx, sw_, sh_, face, t, screen_glow, speak, look, seed)
            _rrect(ctx, -sw_ / 2, -sh_ / 2, sw_, sh_, 18)
            ctx.restore()
            _stroke(ctx, R_MINT_LINE, lw * 0.8)
            # spill glow on the frame
            if screen_glow > 0.05:
                colg = R_LED_WARM if face == "warm" else ("#FF6F91" if face == "heart" else R_LED)
                g = cairo.RadialGradient(ctr[0], ctr[1], 30, ctr[0], ctr[1], 120)
                cc = _c(colg)
                g.add_color_stop_rgba(0, *cc, 0.0)
                g.add_color_stop_rgba(0.55, *cc, 0.10 * screen_glow)
                g.add_color_stop_rgba(1, *cc, 0.0)
                ctx.save()
                ctx.append_path(hpath)
                ctx.clip()
                ctx.set_source(g)
                ctx.paint()
                ctx.restore()
            anchors["screen_center"] = anchor(*ctr)
            anchors["screen_size"] = (150 * c * scale, 88 * scale)
        else:
            anchors["screen_center"] = anchor(*P.p(0, hd, (top + bot) / 2))
            anchors["screen_size"] = (0.0, 94 * scale)
            # edge glow of the screen in pure profile
            e0 = P.p(0, hd + 1, (top + bot) / 2 - 40)
            e1 = P.p(0, hd + 1, (top + bot) / 2 + 40)
            ctx.move_to(*e0)
            ctx.line_to(*e1)
            _stroke(ctx, R_GLASS, 5)
        # rivets on front corners
        if c > 0.2:
            for lr in (-hw + 12, hw - 12):
                for vr in (top + 12, bot - 12):
                    q = P.p(lr, hd, vr)
                    ctx.arc(q[0], q[1], 2.6, 0, TAU)
                    _sc(ctx, mint_sh)
                    ctx.fill()
                    ctx.arc(q[0] - 0.8, q[1] - 0.8, 1.0, 0, TAU)
                    _sc(ctx, R_MINT_HI)
                    ctx.fill()
        # rust streak on the head
        q = P.p(hw * 0.7, hd, bot - 6)
        if c > 0.3:
            _sc(ctx, R_RUST, 0.7)
            _taper(ctx, [(q[0], q[1] - 6), (q[0] + 0.5, q[1] + 3)], 4, 1)
        # hatch lid on top
        hatch_c = (RB_HATCH_L, 4.0, top)
        hop = smoothstep(0.0, 0.2, mini_umbrella)
        hx0, hx1, hf0, hf1 = RB_HATCH_L - 20, RB_HATCH_L + 20, -12.0, 20.0
        ctx.new_path()
        _poly(ctx, [P.p(hx0, hf0, top), P.p(hx1, hf0, top), P.p(hx1, hf1, top), P.p(hx0, hf1, top)])
        _sc(ctx, R_GLASS if hop > 0 else mix(mint, R_MINT_HI, 0.2))
        ctx.fill_preserve()
        _stroke(ctx, R_MINT_LINE, lw * 0.6)
        anchors["hatch"] = anchor(*P.p(*hatch_c))
        # antenna
        ab = P.p(RB_ANT_L, 0, top)
        sway = 0.06 * math.sin(t * 3.3) + 0.04 * wet * math.sin(t * 9)
        tip = (ab[0] + 34 * math.sin(sway), ab[1] - 34 * math.cos(sway))
        _ell(ctx, ab[0], ab[1] - 1, 10, 5)
        _sc(ctx, mint_sh)
        ctx.fill_preserve()
        _stroke(ctx, R_MINT_LINE, lw * 0.7)
        ctx.move_to(*ab)
        ctx.curve_to(ab[0], ab[1] - 15, tip[0], tip[1] + 15, *tip)
        _stroke(ctx, R_METAL_LINE, 5.0)
        ctx.move_to(*ab)
        ctx.curve_to(ab[0], ab[1] - 15, tip[0], tip[1] + 15, *tip)
        _stroke(ctx, R_METAL, 2.6)
        bc = (tip[0], tip[1] - 8)
        if bulb > 0.05:
            g = cairo.RadialGradient(bc[0], bc[1], 2, bc[0], bc[1], 30)
            g.add_color_stop_rgba(0, 1, 0.35, 0.25, 0.55 * bulb)
            g.add_color_stop_rgba(1, 1, 0.3, 0.2, 0)
            ctx.set_source(g)
            ctx.arc(bc[0], bc[1], 30, 0, TAU)
            ctx.fill()
        ctx.arc(bc[0], bc[1], 9, 0, TAU)
        _sc(ctx, mix(R_BULB_OFF, R_BULB, bulb))
        ctx.fill_preserve()
        _stroke(ctx, "#4A0E0A", lw * 0.8)
        ctx.arc(bc[0] - 3, bc[1] - 3, 2.8, 0, TAU)
        ctx.set_source_rgba(1, 1, 1, 0.5 + 0.45 * bulb)
        ctx.fill()
        anchors["antenna"] = anchor(*bc)
        anchors["head_top"] = anchor(*P.p(0, 0, top))
        anchors["head"] = anchor(*P.p(0, 0, (top + bot) / 2))
        drops.extend([anchor(*P.p(lq, 0, top)) for lq in (-70, -30, 20, 70)])
        if wet > 0.05:
            # drips running down the head front and droplets on the top edge
            for i in range(5):
                lq = -80 + 40 * i + 8 * _hash(i, 2)
                ph = (t * (0.35 + 0.2 * _hash(i, 9)) + _hash(i, 4)) % 1.0
                a2 = P.p(lq, hd + 1, top + 4)
                b2 = P.p(lq + 3, hd + 1, lerp(top + 4, bot + 6, ph))
                if c > 0.2 and not (-80 < lq < 80 and False):
                    ctx.move_to(*a2)
                    ctx.line_to(*b2)
                    _stroke(ctx, "#EAF7FF", 2.0, 0.35 * wet)
                    draw_droplet(ctx, b2[0], b2[1], 2.8, wet)
            for i in range(4):
                q = P.p(-75 + 50 * i, hd * 0.6, top)
                draw_droplet(ctx, q[0], q[1] + 1, 3.0, wet * 0.9)

    def neck():
        n0 = P.p(0, 0, -230)
        n1 = P.p(0, 0, -252)
        for k in range(4):
            v = lerp(-232, -250, k / 3)
            q = P.p(0, 0, v)
            _ell(ctx, q[0], q[1], math.hypot(26 * c, 22 * s), 5)
            _sc(ctx, R_METAL if k % 2 == 0 else R_METAL_SH)
            ctx.fill_preserve()
            _stroke(ctx, R_METAL_LINE, lw * 0.6)

    def mini_umb():
        if mini_umbrella <= 0.01:
            return
        p = clamp(mini_umbrella)
        top = Hd["top"]
        base = P.p(RB_HATCH_L, 4, top)
        u = clamp((p - 0.12) / 0.5)
        ext = 1 - (1 - u) ** 2 * math.cos(3.5 * PI * u) if u > 0 else 0.0
        ext = max(0.0, ext)
        Ls = 74 * ext
        lean = 0.36 * smoothstep(0.3, 0.8, p) + 0.07 * math.sin(t * 4.2) * smoothstep(0.6, 1.0, p)
        ang = lean * (1 if True else -1)
        tipp = (base[0] + Ls * math.sin(ang), base[1] - Ls * math.cos(ang))
        # spring coil
        if Ls > 2:
            n = 40
            d = _norm(_sub(tipp, base))
            nn = (-d[1], d[0])
            pts = [_add(_add(base, d, Ls * i / n), nn, 6 * math.sin(i / n * TAU * 6)) for i in range(n + 1)]
            ctx.new_path()
            ctx.move_to(*pts[0])
            for q in pts[1:]:
                ctx.line_to(*q)
            _stroke(ctx, R_METAL_LINE, 3.6)
            ctx.new_path()
            ctx.move_to(*pts[0])
            for q in pts[1:]:
                ctx.line_to(*q)
            _stroke(ctx, R_METAL_HI, 1.8)
        # open lid
        hop = smoothstep(0.0, 0.2, p)
        hx0, hx1, hf1 = RB_HATCH_L - 20, RB_HATCH_L + 20, -12.0
        a = hop * 1.9
        lid = [P.p(hx0, hf1, top), P.p(hx1, hf1, top),
               P.p(hx1, hf1 + 32 * math.cos(a) * -1 * -1 if False else hf1 - 32 * math.cos(PI - a) * -1, top - 32 * math.sin(a)),
               P.p(hx0, hf1 - 32 * math.cos(PI - a) * -1, top - 32 * math.sin(a))]
        lid = [P.p(hx0, hf1, top), P.p(hx1, hf1, top),
               P.p(hx1, hf1 + 32 * math.cos(a), top - 32 * math.sin(a)),
               P.p(hx0, hf1 + 32 * math.cos(a), top - 32 * math.sin(a))]
        ctx.new_path()
        _rpoly(ctx, lid, 3)
        _sc(ctx, mint)
        ctx.fill_preserve()
        _stroke(ctx, R_MINT_LINE, lw * 0.7)
        # the tiny umbrella
        if Ls > 4:
            o = ease_out_back(clamp((p - 0.55) / 0.45), 2.4) if p > 0.55 else 0.0
            rr_ = 2 * Hd["hw"] / 8.0
            ctx.save()
            ctx.translate(*tipp)
            ctx.rotate(ang * 0.6)
            # handle hook
            ctx.move_to(0, 0)
            ctx.line_to(0, -9)
            ctx.move_to(0, 0)
            ctx.curve_to(0, 5, 6, 5, 6, 1)
            _stroke(ctx, "#6A3A20", 2.4)
            shaft_top = -9 - 18
            ctx.move_to(0, -9)
            ctx.line_to(0, shaft_top)
            _stroke(ctx, R_METAL_LINE, 1.6)
            w_ = max(3.0, rr_ * o)
            hgt = lerp(20, 12, clamp(o))
            ytop = shaft_top - 2
            ctx.new_path()
            ctx.move_to(-w_, ytop + hgt)
            ctx.curve_to(-w_, ytop + hgt * 0.1, -w_ * 0.4, ytop - hgt * 0.15, 0, ytop - hgt * 0.15)
            ctx.curve_to(w_ * 0.4, ytop - hgt * 0.15, w_, ytop + hgt * 0.1, w_, ytop + hgt)
            nsc = 4
            for k in range(nsc):
                x0_ = w_ - 2 * w_ * k / nsc
                x1_ = w_ - 2 * w_ * (k + 1) / nsc
                ctx.curve_to(x0_ - (x0_ - x1_) * 0.3, ytop + hgt - 4, x1_ + (x0_ - x1_) * 0.3, ytop + hgt - 4,
                             x1_, ytop + hgt)
            ctx.close_path()
            path = ctx.copy_path()
            _sc(ctx, _wet(UMB_RED, wet, 0.3))
            ctx.fill_preserve()
            ctx.save()
            ctx.clip()
            ctx.new_path()
            ctx.rectangle(-w_ * 1.2 + 4 * ls * 0 + (w_ * 0.25 if ls > 0 else -w_ * 1.2), ytop - hgt, w_ * 0.95,
                          hgt * 3)
            _sc(ctx, UMB_RED_SH, 0.0)
            ctx.fill()
            for k in range(1, nsc):
                xk = w_ - 2 * w_ * k / nsc
                ctx.move_to(0, ytop - hgt * 0.15)
                ctx.curve_to(xk * 0.6, ytop, xk, ytop + hgt * 0.5, xk, ytop + hgt)
            _stroke(ctx, UMB_RED_SH, 1.1)
            _sc(ctx, "#FFFFFF", 0.55)
            _taper(ctx, [(-w_ * 0.55, ytop + hgt * 0.35), (-w_ * 0.3, ytop + 1.0), (0, ytop - hgt * 0.05)],
                   1.0, 0.5, 2.4)
            ctx.restore()
            ctx.new_path()
            ctx.append_path(path)
            _stroke(ctx, UMB_LINE, lw * 0.7)
            ctx.arc(0, ytop - hgt * 0.15 - 2.5, 2.2, 0, TAU)
            _sc(ctx, "#FFF3D0")
            ctx.fill_preserve()
            _stroke(ctx, UMB_LINE, 0.8)
            anchors["umbrella_top"] = anchor(0, ytop - hgt * 0.2)
            anchors["umbrella_radius"] = w_ * scale
            ctx.restore()

    def sparks_fx():
        if sparks <= 0.01:
            return
        k = int(t * 16)
        pts3 = [((-30, 18, -240), (-1, -0.4)), ((30, 18, -240), (1, -0.4)), ((-88, 10, -200), (-1, -0.6)),
                ((88, 10, -200), (1, -0.6)), ((-100, 0, -280), (-1, -0.2)), ((100, 0, -280), (1, -0.3)),
                ((-62, 50, -60), (-1, 0.3)), ((62, 50, -60), (1, 0.2)), ((0, 40, -246), (0.3, -1))]
        for i, (q, dv) in enumerate(pts3):
            if _hash(k, i, seed) > 0.55 * sparks + 0.05:
                continue
            p2 = P.p3(q)
            dirv = _norm((dv[0] + (_hash(k, i, 2) - 0.5) * 1.2, dv[1] + (_hash(k, i, 3) - 0.5) * 1.2))
            _spark_bolt(ctx, p2, dirv, 18 + 26 * _hash(k, i, 4) * sparks, k * 31 + i, 1.0)

    def wet_fx():
        if wet <= 0.05:
            return
        # streaks + drips on the body front
        for i in range(7):
            ph0 = -1.1 + 2.2 * i / 6 + 0.1 * _hash(i, 1)
            q0 = surf(-232, ph0)
            if q0[2] < 0.15:
                continue
            spd = 0.3 + 0.3 * _hash(i, 6)
            u = (t * spd + _hash(i, 8)) % 1.0
            vend = lerp(-225, -40, u)
            q1 = surf(vend, ph0 + 0.02)
            ctx.move_to(q0[0], q0[1])
            ctx.line_to(q1[0], q1[1])
            _stroke(ctx, "#E6F6FF", 2.2, 0.3 * wet)
            draw_droplet(ctx, q1[0], q1[1], 3.0, wet)
        # drops falling off the bottom
        for i in range(3):
            u = (t * 1.6 + i * 0.37) % 1.0
            q = surf(-34, -0.8 + 0.8 * i)
            draw_droplet(ctx, q[0], q[1] + u * 40, 2.4, wet * (1 - u))

    # ------------------------------------------------ draw order
    if glow_only:
        upper_begin()
        top, bot = Hd["top"], Hd["bot"]
        if c > 0.06:
            ctr = P.p(0, Hd["hd"], (top + bot) / 2 + 1)
            ctx.save()
            ctx.translate(*ctr)
            ctx.transform(cairo.Matrix(c, -0.1 * s, 0, 1, 0, 0))
            _draw_screen(ctx, 150, 88, face, t, screen_glow, speak, look, seed, glow_only=True)
            ctx.restore()
        ab = P.p(RB_ANT_L, 0, top)
        sway = 0.06 * math.sin(t * 3.3) + 0.04 * wet * math.sin(t * 9)
        bc = (ab[0] + 34 * math.sin(sway), ab[1] - 34 * math.cos(sway) - 8)
        ctx.arc(bc[0], bc[1], 9, 0, TAU)
        _sc(ctx, R_BULB, bulb)
        ctx.fill()
        sparks_fx()
        ctx.restore()
        ctx.restore()
        return anchors

    upper_begin()
    back_doors()
    for sd in sorted(arm_info, key=lambda q: arm_info[q]["z"]):
        if arm_info[sd]["z"] < -5:
            draw_arm(sd)
    ctx.restore()
    tread_order = sorted((-1, 1), key=lambda sd: P.z(sd * T["l"], 0))
    if s > 0.3:
        draw_tread(tread_order[0])
    upper_begin()
    body()
    ctx.restore()
    if s > 0.3:
        draw_tread(tread_order[1])
    else:
        for sd in tread_order:
            draw_tread(sd)
    upper_begin()
    wet_fx()
    neck()
    ctx.save()
    hcen = P.p(0, 0, Hd["bot"])
    ctx.translate(*hcen)
    ctx.rotate(head_tilt)
    ctx.translate(-hcen[0], -hcen[1])
    head()
    mini_umb()
    ctx.restore()
    for sd in sorted(arm_info, key=lambda q: arm_info[q]["z"]):
        if arm_info[sd]["z"] >= -5:
            draw_arm(sd)
    sparks_fx()
    anchors["chest"] = anchor(*P.p(0, 60, -190))
    anchors["belly"] = anchor(*P.p(0, 70, -122))
    drops.extend([anchor(*P.p(sd * 88, 0, -196)) for sd in (-1, 1)])
    ctx.restore()
    anchors["feet"] = anchor(0, 0)
    hl_, hr_ = anchors.get("hand_l", (0, 0)), anchors.get("hand_r", (0, 0))
    anchors["hands"] = ((hl_[0] + hr_[0]) / 2, (hl_[1] + hr_[1]) / 2)
    anchors["drop_points"] = drops
    ctx.restore()
    if alpha < 1.0:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(alpha)
    return anchors


def robo_glow_layer(x, y, scale=1.0, **pose):
    """RGBA float layer (H, W, 4) with only Robo's emissive parts (LED pixels, bulb, sparks).
    Use: core.add_glow(frame, robo_glow_layer(x, y, s, face=..., t=t), radius=14*s, strength=1.2)."""
    from common.core import new_surface, surface_rgba
    surf, c = new_surface()
    draw_robo(c, x, y, scale, glow_only=True, **pose)
    return surface_rgba(surf)


# =============================================================== kitten (calico)
def draw_kitten(ctx, x, y, scale=1.0, *, pose="sit", expr="neutral", blink=0.0, t=0.0, wet=0.0, facing=1,
                look=(0.0, 0.0), rim=RIM, alpha=1.0):
    """Tiny calico kitten.  (x, y) = ground point under its body; scale 1 -> ~70 px tall (sitting).

    pose : 'sit' | 'curl' (sleeping loaf) | 'nuzzle' (rubbing its head forward, uses t) | 'stand'
    expr : 'neutral' | 'happy' (^ ^) | 'mew' (open mouth)   blink 0..1, look (dx, dy)
    t (s) tail sway / breathing, wet 0..1 (darker spiky fur, droplets), facing +1 right / -1 left.
    Returns anchors: head, nose, mouth, tail_tip, top, feet, scale.
    """
    anchor, _ = _anchor_fn(ctx)
    lw = line_weight(scale, 2.1)
    wet = clamp(wet)
    WH = _wet(K_WHITE, wet, 0.45)
    WH_SH = _wet(K_WHITE_SH, wet, 0.45)
    OR = _wet(K_ORANGE, wet, 0.45)
    BK = _wet(K_BLACK, wet, 0.3)
    ls = 1 if facing > 0 else -1
    if alpha < 1.0:
        ctx.push_group()
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale * (1 if facing > 0 else -1), scale)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    anchors = {"scale": scale}
    breath = math.sin(t * TAU / 1.6) * 0.6
    sway = math.sin(t * 2.4)
    spiky = wet > 0.3

    def fur(pts):
        """Add little fur tufts (sharp points) between outline points when wet/fluffy."""
        if not spiky:
            return pts
        out = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            out.append(a)
            m = _l2(a, b, 0.5)
            ctr = (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
            d = _norm(_sub(m, ctr))
            out.append((m[0] + d[0] * 1.6, m[1] + d[1] * 1.6, True))
        return out

    def patches(c, items):
        for (px, py, rx, ry, rot, col) in items:
            _ell(c, px, py, rx, ry, rot)
            _sc(c, col)
            c.fill()

    def tail(pts, w0=6.5, w1=3.2):
        # orange tail with black tip
        _sc(ctx, K_LINE)
        _taper(ctx, pts, w0 + lw * 1.6, w1 + lw * 1.6)
        _sc(ctx, OR)
        _taper(ctx, pts, w0, w1)
        tip = pts[-2:]
        _sc(ctx, BK)
        _taper(ctx, [_l2(tip[0], tip[1], 0.35), tip[1]], w1 * 1.2, w1)
        anchors["tail_tip"] = anchor(*pts[-1])

    def head(hx, hy, rot=0.0, closed=False):
        ctx.save()
        ctx.translate(hx, hy)
        ctx.rotate(rot)
        droop = 2.5 * wet
        # ears
        for sd in (-1, 1):
            ear = [(sd * 5, -10), (sd * 13 + sd * droop * 0.8, -21 + droop, True), (sd * 16, -6)]
            col = OR if sd < 0 else BK
            _cel(ctx, lambda c, e=ear: _spline(c, e, True, 0.6), col, None, None, K_LINE, lw)
            inner = [(sd * 7, -10), (sd * 12.5 + sd * droop * 0.8, -18 + droop, True), (sd * 14, -8)]
            ctx.new_path()
            _spline(ctx, inner, True, 0.5)
            _sc(ctx, K_PINK, 0.9)
            ctx.fill()
        hp = fur([(0, -13), (10, -11.5), (16, -5), (18.5, 2, True), (14, 6), (17, 8.5, True), (9, 11.5), (0, 12.5),
                  (-9, 11.5), (-17, 8.5, True), (-14, 6), (-18.5, 2, True), (-16, -5), (-10, -11.5)])

        def hx_extra(c):
            patches(c, [(-9, -9, 9, 7, 0.4, OR), (10, -10, 7, 5, -0.4, BK)])
            _ell(c, 0, 6, 9, 6)
            _sc(c, WH)
            c.fill()

        _cel(ctx, lambda c: _spline(c, hp, True, 0.9), WH, WH_SH, rim, K_LINE, lw, soff=(-3 * ls, -2.5),
             roff=(-1.2, 0.3), shade_a=0.8, rim_a=0.6, extra=hx_extra)
        # eyes
        lx, ly = look
        for sd in (-1, 1):
            ex, ey = sd * 6.5, 0.5
            if closed or expr == "happy" or blink > 0.6:
                _sc(ctx, K_EYE)
                if expr == "happy" and not closed:
                    _taper(ctx, [(ex - 3.2, ey + 1), (ex, ey - 2), (ex + 3.2, ey + 1)], 1.3, 1.3, 1.8)
                else:
                    _taper(ctx, [(ex - 3.2, ey), (ex, ey + 2), (ex + 3.2, ey)], 1.3, 1.3, 1.8)
            else:
                _ell(ctx, ex + lx * 1.2, ey + ly * 1.0, 3.6, 4.0 * (1 - blink))
                _sc(ctx, K_EYE)
                ctx.fill()
                ctx.arc(ex + lx * 1.2 - 1.2, ey - 1.5, 1.35, 0, TAU)
                ctx.set_source_rgba(1, 1, 1, 0.95)
                ctx.fill()
                ctx.arc(ex + lx * 1.2 + 1.3, ey + 1.6, 0.6, 0, TAU)
                ctx.fill()
        # blush
        for sd in (-1, 1):
            _ell(ctx, sd * 10, 6.5, 3.2, 1.8)
            _sc(ctx, K_PINK, 0.55)
            ctx.fill()
        # nose + mouth
        ctx.move_to(-1.8, 4.2)
        ctx.line_to(1.8, 4.2)
        ctx.line_to(0, 6.0)
        ctx.close_path()
        _sc(ctx, "#F27C8C")
        ctx.fill()
        if expr == "mew":
            ctx.new_path()
            ctx.move_to(-2.8, 7.2)
            ctx.curve_to(-2.4, 11.5, 2.4, 11.5, 2.8, 7.2)
            ctx.curve_to(1, 8, -1, 8, -2.8, 7.2)
            _sc(ctx, "#7A2630")
            ctx.fill_preserve()
            _stroke(ctx, K_LINE, 0.8)
            _ell(ctx, 0, 9.6, 1.6, 1.0)
            _sc(ctx, "#F08592")
            ctx.fill()
        else:
            _sc(ctx, K_LINE)
            ctx.move_to(0, 6.0)
            ctx.curve_to(0, 8, -2.6, 8.4, -3.2, 7)
            ctx.move_to(0, 6.0)
            ctx.curve_to(0, 8, 2.6, 8.4, 3.2, 7)
            ctx.set_line_width(0.9)
            ctx.stroke()
        # whiskers
        _sc(ctx, K_LINE, 0.55)
        for sd in (-1, 1):
            for k in (-1, 0, 1):
                ctx.move_to(sd * 11, 5 + k * 1.6)
                ctx.line_to(sd * 21, 3.5 + k * 3.0)
        ctx.set_line_width(0.6)
        ctx.stroke()
        if wet > 0.2:
            draw_droplet(ctx, -12, -15, 1.6, wet)
            draw_droplet(ctx, 9, 13, 1.4, wet)
        anchors["head"] = anchor(0, 0)
        anchors["nose"] = anchor(0, 4.5)
        anchors["mouth"] = anchor(0, 7.5)
        anchors["top"] = anchor(0, -20)
        ctx.restore()

    def leg(x0, y0, x1, w=4.6):
        pts = [(x0, y0), (x1, -3)]
        _cel(ctx, lambda c: _limb(c, pts, [w, w * 0.9]), WH, WH_SH, None, K_LINE, lw * 0.9, soff=(-1.5, 0))
        _ell(ctx, x1 + 1.0, -2.4, w * 1.15, 2.6)
        _sc(ctx, WH)
        ctx.fill_preserve()
        _stroke(ctx, K_LINE, lw * 0.9)

    if pose == "sit":
        tail([(-15, -4), (-4, 1.5), (10, 0.5), (18, -2 - 3 * sway), (21 + sway, -7 - 4 * sway)])
        body = fur([(-17, -1), (-21, -13), (-17, -28 - breath), (-7, -38 - breath), (6, -39 - breath),
                    (13, -29), (15, -13), (13, -1)])

        def b_extra(c):
            patches(c, [(-13, -24, 9, 12, 0.3, OR), (-14, -6, 7, 6, 0.1, BK), (-4, -34, 6, 4, 0.2, OR)])

        _cel(ctx, lambda c: _spline(c, body), WH, WH_SH, rim, K_LINE, lw, soff=(-4 * ls, -2), roff=(-1.5, 0.3),
             shade_a=0.7, rim_a=0.6, extra=b_extra)
        leg(2, -24, 2, 4.4)
        leg(10, -24, 11, 4.4)
        head(4, -52 - breath, 0.06 * math.sin(t * 0.9))
    elif pose == "stand":
        tail([(-20, -26), (-27, -36), (-26 - 2 * sway, -50), (-21 - 4 * sway, -56)], 5.5, 3.0)
        for lx in (-15, -8):
            leg(lx, -22, lx - 1)
        body = fur([(-24, -24), (-17, -35 - breath), (2, -36 - breath), (17, -31), (21, -22), (14, -15),
                    (-5, -14), (-20, -16)])

        def b_extra(c):
            patches(c, [(-11, -30, 11, 6, 0.0, OR), (6, -32, 6, 4, 0.2, BK)])

        _cel(ctx, lambda c: _spline(c, body), WH, WH_SH, rim, K_LINE, lw, soff=(-4 * ls, -2), roff=(-1.5, 0.3),
             shade_a=0.7, rim_a=0.6, extra=b_extra)
        for lx in (9, 15):
            leg(lx, -20, lx + 1)
        head(23, -42, 0.05 * math.sin(t))
    elif pose == "curl":
        tail([(-20, -6), (-14, 0.5), (4, 1.0), (18, -1), (24, -5 - sway)], 6.0, 3.0)
        body = fur([(-24, -5), (-20, -17 - breath), (-6, -23 - breath), (10, -22 - breath), (22, -14), (22, -2),
                    (-20, -1)])

        def b_extra(c):
            patches(c, [(-8, -17, 12, 6, 0.1, OR), (8, -18, 6, 4, 0.0, BK)])

        _cel(ctx, lambda c: _spline(c, body), WH, WH_SH, rim, K_LINE, lw, soff=(-4 * ls, -2), roff=(-1.5, 0.3),
             shade_a=0.7, rim_a=0.6, extra=b_extra)
        head(17, -13, 0.35, closed=True)
        # zzz-less: little breathing tuft
    elif pose == "nuzzle":
        rub = math.sin(t * 5.5)
        tail([(-20, -26), (-24, -38), (-22 + 2 * sway, -52), (-17 + 3 * sway, -58)], 5.5, 3.0)
        for lx in (-15, -8):
            leg(lx, -22, lx - 1)
        body = fur([(-24, -24), (-17, -35), (2, -35), (15, -28), (19, -20), (12, -15), (-5, -14), (-20, -16)])

        def b_extra(c):
            patches(c, [(-11, -30, 11, 6, 0.0, OR), (6, -31, 6, 4, 0.2, BK)])

        _cel(ctx, lambda c: _spline(c, body), WH, WH_SH, rim, K_LINE, lw, soff=(-4 * ls, -2), roff=(-1.5, 0.3),
             shade_a=0.7, rim_a=0.6, extra=b_extra)
        for lx in (9, 14):
            leg(lx, -20, lx + 1)
        head(24 + rub * 1.5, -27 + abs(rub) * 1.0, 0.45 + 0.12 * rub, closed=True)
        # little hearts of affection
        if rub > 0.3:
            draw_sparkle(ctx, 30, -46, 3.5 * rub, 0.8, 0.3, "#FFD0DA")
    anchors["feet"] = anchor(0, 0)
    if wet > 0.2:
        draw_droplet(ctx, -10, -18, 1.6, wet)
        draw_droplet(ctx, 6, -8, 1.5, wet)
    ctx.restore()
    if alpha < 1.0:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(alpha)
    return anchors


# =============================================================== rain interaction helpers
def draw_drops_on(ctx, points, t, scale=1.0, seed=0, amount=1.0):
    """Water beading on surfaces: at each point a drop grows, slides a little and falls off.
    points: caller-space (x, y) list, e.g. anchors['drop_points']."""
    if amount <= 0:
        return
    for i, p in enumerate(points):
        per = 0.9 + 0.8 * _hash(i, seed, 1)
        u = ((t + _hash(i, seed, 2) * per) % per) / per
        r = (1.4 + 2.2 * smoothstep(0.0, 0.6, u)) * scale
        if u < 0.65:
            draw_droplet(ctx, p[0], p[1] + r * 0.6 + 4 * scale * u, r, amount)
        else:
            k = (u - 0.65) / 0.35
            dy = 4 * scale * 0.65 + 90 * scale * k * k
            draw_droplet(ctx, p[0], p[1] + dy, r * 0.8, amount * (1 - k), 1.0 + 0.8 * k)


def draw_rim_drips(ctx, rim_points, t, scale=1.0, seed=0, amount=1.0, fall=420.0):
    """Streams of drops falling from an umbrella rim (canopy anchors['canopy_rim_points'])."""
    if amount <= 0:
        return
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i, p in enumerate(rim_points):
        for k in range(3):
            per = 0.55 + 0.25 * _hash(i, k, seed)
            u = ((t + _hash(i, k, seed, 4)) % per) / per
            dy = fall * scale * u * u
            ln = 26 * scale * (0.3 + u)
            a = amount * (1 - 0.6 * u)
            ctx.move_to(p[0], p[1] + dy - ln)
            ctx.line_to(p[0], p[1] + dy)
            ctx.set_source_rgba(0.93, 0.97, 1.0, 0.8 * a)
            ctx.set_line_width(2.8 * scale)
            ctx.stroke()
            if u < 0.15:
                draw_droplet(ctx, p[0], p[1] + 3 * scale, 3.0 * scale, a)
    ctx.restore()
