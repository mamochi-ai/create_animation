"""Character rigs for 「ほしのおとしもの」: Haru, Kira and the paper lantern.

Everything is drawn procedurally with pycairo as flat cel-shaded vector art
(one shadow tone + a cool rim light, coloured outlines).  All functions are
deterministic.  Coordinates are the caller's current cairo user space
(normally 1920x1080 canvas pixels); the rigs respect any transform already on
the context and return anchor points in that same space.

Public API (see docstrings for the full option list)
----------------------------------------------------
draw_haru(ctx, x, y, scale=1.0, **pose) -> anchors dict
draw_kira(ctx, x, y, scale=1.0, **pose) -> anchors dict
draw_kira_glow(ctx, x, y, scale=1.0, glow=1.0, t=0.0)      cheap cairo halo (draw BEFORE draw_kira)
kira_glow_layer(x, y, scale, glow, t=0.0, **pose) -> RGBA array for core.add_glow
draw_lantern(ctx, x, y, scale=1.0, lit=0.0, sway=0.0, t=0.0, kira=0.0) -> anchors
draw_lantern_glow(ctx, x, y, scale=1.0, lit=1.0, sway=0.0)  cheap cairo halo (draw BEFORE the lantern)
draw_sparkle(ctx, x, y, size, alpha=1.0, rot=0.0, color=...)  4-point twinkle for trails
"""
from __future__ import annotations

import math

import cairo

try:  # imported as package (common.characters)
    from .core import clamp, lerp, mix, rgb, set_color, smoothstep, noise1, TAU
except ImportError:  # pragma: no cover
    from common.core import clamp, lerp, mix, rgb, set_color, smoothstep, noise1, TAU

PI = math.pi

# =============================================================== palette
HAIR, HAIR_SH, HAIR_HI, HAIR_LINE = "#1A1426", "#0F0B18", "#4A4274", "#07050C"
SKIN, SKIN_SH, SKIN_LINE = "#FCE3D2", "#EFBFAE", "#A8625A"
CHEEK = "#F59CA0"
YUKATA, YUKATA_SH, YUKATA_LINE = "#2B3A7A", "#1D275A", "#10163A"
PATTERN = "#9EC5FF"
COLLAR = "#8EA6E8"
OBI, OBI_SH, OBI_LINE, OBI_HI = "#F2C14E", "#D39A30", "#8C5A1A", "#FFE9A6"
CORD = "#E2574C"
CLIP, CLIP_SH, CLIP_LINE = "#E8454E", "#B52836", "#6E1422"
STRAW, STRAW_SH, STRAW_LINE = "#DDBB7C", "#B89355", "#72532A"
STRAP = "#D8343E"
IRIS_TOP, IRIS_MID, IRIS_BOT = "#1E1024", "#4A2A4C", "#9A6A8A"
LASH = "#24121C"
MOUTH_IN, TONGUE, MOUTH_LINE = "#6E2233", "#EE8391", "#7A2E36"
BAMBOO, BAMBOO_SH, BAMBOO_LINE = "#CDAA5E", "#9E7E3C", "#5E4620"
RIM_DEFAULT = "#A9BEFF"

KIRA_CORE, KIRA_MID, KIRA_RIM = "#FFF6D6", "#FFE9A8", "#FFC94D"
KIRA_LINE, KIRA_EYE = "#D9912E", "#3A2A1A"

PAPER, PAPER_SH, PAPER_LINE = "#F6F0E4", "#D2CBD2", "#8A7F80"
LANT_RED, LANT_RED_SH, LANT_RED_LINE = "#CC3B3B", "#962430", "#4A1016"

# Global line weight (units at scale 1).  Lines grow sub-linearly with scale so
# close-ups don't get cartoonishly fat outlines.
_LW = [2.4]


def line_weight(scale: float, base: float = 2.4) -> float:
    """Outline width in *local* units for a rig drawn at `scale` (px = lw*scale)."""
    return base * max(scale, 0.05) ** -0.33


# =============================================================== small helpers
def _c(c):
    return rgb(c) if isinstance(c, str) else c


def _sc(ctx, c, a=1.0):
    c = _c(c)
    ctx.set_source_rgba(c[0], c[1], c[2], a)


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


def _lerp2(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _spline(ctx, pts, closed=True, k=1.0, move=True):
    """Smooth Catmull-Rom curve through pts.  A point given as (x, y, True) is a
    sharp corner (cusp) -- used for hair tips, lash flicks, star points."""
    n = len(pts)
    if n == 0:
        return
    P = [(p[0], p[1]) for p in pts]
    sharp = [len(p) > 2 and bool(p[2]) for p in pts]
    if n == 1:
        ctx.move_to(*P[0])
        return

    def tan(i):
        if sharp[i]:
            return (0.0, 0.0)
        if closed:
            a, b = P[(i - 1) % n], P[(i + 1) % n]
        else:
            a, b = P[max(i - 1, 0)], P[min(i + 1, n - 1)]
        return ((b[0] - a[0]) * k / 6.0, (b[1] - a[1]) * k / 6.0)

    if move:
        ctx.move_to(*P[0])
    else:
        ctx.line_to(*P[0])
    T = [tan(i) for i in range(n)]
    segs = n if closed else n - 1
    for i in range(segs):
        j = (i + 1) % n
        ctx.curve_to(P[i][0] + T[i][0], P[i][1] + T[i][1],
                     P[j][0] - T[j][0], P[j][1] - T[j][1], P[j][0], P[j][1])
    if closed:
        ctx.close_path()


def _sample_spline(pts, closed=False, per=8, k=1.0):
    """Polyline samples of the same Catmull-Rom curve (for tapered strokes)."""
    n = len(pts)
    P = [(p[0], p[1]) for p in pts]
    sharp = [len(p) > 2 and bool(p[2]) for p in pts]

    def tan(i):
        if sharp[i]:
            return (0.0, 0.0)
        if closed:
            a, b = P[(i - 1) % n], P[(i + 1) % n]
        else:
            a, b = P[max(i - 1, 0)], P[min(i + 1, n - 1)]
        return ((b[0] - a[0]) * k / 6.0, (b[1] - a[1]) * k / 6.0)

    T = [tan(i) for i in range(n)]
    out = []
    segs = n if closed else n - 1
    for i in range(segs):
        j = (i + 1) % n
        p0, p3 = P[i], P[j]
        p1 = (p0[0] + T[i][0], p0[1] + T[i][1])
        p2 = (p3[0] - T[j][0], p3[1] - T[j][1])
        for s in range(per):
            u = s / per
            v = 1 - u
            out.append((v * v * v * p0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u * u * u * p3[0],
                        v * v * v * p0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u * u * u * p3[1]))
    if not closed:
        out.append(P[-1])
    return out


def _taper(ctx, pts, w0, w1, wmid=None, per=6, k=1.0):
    """Fill a brush-like tapered stroke along a smooth curve through pts.
    widths: w0 at start, w1 at end, optional wmid peak in the middle."""
    c = _sample_spline(pts, False, per, k) if len(pts) > 2 else [pts[0], pts[-1]]
    if len(pts) == 2:
        c = [_lerp2(pts[0], pts[1], i / 8) for i in range(9)]
    n = len(c)
    L, R = [], []
    for i in range(n):
        a = c[max(i - 1, 0)]
        b = c[min(i + 1, n - 1)]
        d = _norm(_sub(b, a))
        nx, ny = -d[1], d[0]
        u = i / (n - 1)
        if wmid is None:
            w = lerp(w0, w1, u)
        else:
            w = lerp(w0, wmid, u / 0.5) if u < 0.5 else lerp(wmid, w1, (u - 0.5) / 0.5)
        w *= 0.5
        L.append((c[i][0] + nx * w, c[i][1] + ny * w))
        R.append((c[i][0] - nx * w, c[i][1] - ny * w))
    ctx.move_to(*L[0])
    for p in L[1:]:
        ctx.line_to(*p)
    for p in reversed(R):
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()


def _cel(ctx, path, base, shade=None, rim=None, line=None, lw=None,
         soff=(-5.0, -4.0), roff=(-2.4, 0.4), shade_a=1.0, rim_a=0.9, extra=None,
         base_src=None):
    """Fill a shape with flat base colour, a cel-shadow crescent (shape minus a
    copy shifted toward the light), a thin rim-light crescent and an outline.
    `path(ctx)` must append a closed path.  `extra(ctx)` runs clipped to the shape."""
    ctx.new_path()
    path(ctx)
    if base_src is not None:
        ctx.set_source(base_src)
    else:
        _sc(ctx, base)
    if shade is None and rim is None and extra is None:
        if line:
            ctx.fill_preserve()
            _sc(ctx, line)
            ctx.set_line_width(lw if lw is not None else _LW[0])
            ctx.stroke()
        else:
            ctx.fill()
        return
    ctx.fill_preserve()
    ctx.save()
    ctx.clip()
    for col, off, a in ((shade, soff, shade_a), (rim, roff, rim_a)):
        if col is None or a <= 0:
            continue
        ctx.new_path()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(-1e4, -1e4, 2e4, 2e4)
        ctx.save()
        ctx.translate(*off)
        path(ctx)
        ctx.restore()
        _sc(ctx, col, a)
        ctx.fill()
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    if extra is not None:
        ctx.new_path()
        extra(ctx)
    ctx.restore()
    ctx.new_path()
    if line:
        path(ctx)
        _sc(ctx, line)
        ctx.set_line_width(lw if lw is not None else _LW[0])
        ctx.stroke()


def _hull(points):
    pts = sorted(set((round(p[0], 3), round(p[1], 3)) for p in points))
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


def _resample_closed(pts, spacing):
    """Resample a closed polygon so vertices are roughly evenly spaced (for
    smoothing hulls without long straight runs producing wobbles)."""
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        L = _len(_sub(b, a))
        m = max(1, int(L / spacing))
        for j in range(m):
            out.append(_lerp2(a, b, j / m))
    return out


def _limb(ctx, pts, radii):
    """Closed smooth outline around a polyline with per-joint radii (round caps)."""
    n = len(pts)
    dirs = []
    for i in range(n):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, n - 1)]
        d = _sub(b, a)
        if _len(d) < 1e-3:
            d = dirs[-1] if dirs else (0.0, 1.0)
        dirs.append(_norm(d))
    L = [(pts[i][0] - dirs[i][1] * radii[i], pts[i][1] + dirs[i][0] * radii[i]) for i in range(n)]
    R = [(pts[i][0] + dirs[i][1] * radii[i], pts[i][1] - dirs[i][0] * radii[i]) for i in range(n)]
    # end caps (3 points each)
    de, ds = dirs[-1], dirs[0]
    cap_e = [(pts[-1][0] + de[0] * radii[-1] * 0.95, pts[-1][1] + de[1] * radii[-1] * 0.95)]
    cap_s = [(pts[0][0] - ds[0] * radii[0] * 0.95, pts[0][1] - ds[1] * radii[0] * 0.95)]
    ring = L + cap_e + list(reversed(R)) + cap_s
    _spline(ctx, ring, True, 1.0)


def _star_points(cx, cy, r_out, r_in, n=5, rot=-PI / 2):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * PI / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _rounded_star(ctx, cx, cy, r_out, r_in, rot=-PI / 2, rnd=0.35):
    """Star with rounded tips and soft inner corners."""
    pts = _star_points(cx, cy, r_out, r_in, 5, rot)
    out = []
    n = len(pts)
    for i, p in enumerate(pts):
        prv, nxt = pts[i - 1], pts[(i + 1) % n]
        k = rnd if i % 2 == 0 else rnd * 0.4
        out.append(_lerp2(p, prv, k * 0.5))
        out.append(p if i % 2 else _lerp2(p, (cx, cy), 0.02))
        out.append(_lerp2(p, nxt, k * 0.5))
    # quadratic-ish rounding: corners as curve_to through control point p
    ctx.move_to(*out[0])
    for i in range(0, len(out), 3):
        a, p, b = out[i], out[i + 1], out[(i + 2) % len(out)]
        if i:
            ctx.line_to(*a)
        ctx.curve_to(lerp(a[0], p[0], 0.66), lerp(a[1], p[1], 0.66),
                     lerp(b[0], p[0], 0.66), lerp(b[1], p[1], 0.66), b[0], b[1])
    ctx.close_path()


def draw_sparkle(ctx, x, y, size, alpha=1.0, rot=0.0, color="#FFF6D6"):
    """A soft 4-point twinkle (for Kira's trail, fireflies, the final star)."""
    if alpha <= 0 or size <= 0:
        return
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    c = _c(color)
    g = cairo.RadialGradient(0, 0, 0, 0, 0, size * 1.2)
    g.add_color_stop_rgba(0, c[0], c[1], c[2], 0.55 * alpha)
    g.add_color_stop_rgba(1, c[0], c[1], c[2], 0)
    ctx.set_source(g)
    ctx.arc(0, 0, size * 1.2, 0, TAU)
    ctx.fill()
    s, w = size, size * 0.18
    ctx.move_to(0, -s)
    ctx.curve_to(w, -w, w, -w, s, 0)
    ctx.curve_to(w, w, w, w, 0, s)
    ctx.curve_to(-w, w, -w, w, -s, 0)
    ctx.curve_to(-w, -w, -w, -w, 0, -s)
    ctx.close_path()
    ctx.set_source_rgba(1, 1, 1, alpha)
    ctx.fill()
    ctx.restore()


# =============================================================== 3D-lite body projection
CAM_TILT = 0.13  # we look very slightly down on characters: front points sit lower


class _Proj:
    """Projects body-space (l=lateral(+ her left), f=forward, v=down) to 2D for a yaw."""

    def __init__(self, ya):
        self.ya = ya
        self.c = math.cos(ya)
        self.s = math.sin(ya)

    def p(self, l, f, v):
        x = l * self.c + f * self.s
        z = f * self.c - l * self.s
        return (x, v + CAM_TILT * z)

    def z(self, l, f):
        return f * self.c - l * self.s

    def p3(self, q):
        return self.p(q[0], q[1], q[2])


# torso rings: (v rel hip, half-width, half-depth, forward offset)
_TORSO = [(-166, 10.5, 10.0, 0.0), (-156, 22, 15, 0.0), (-146, 36, 20, 0.0), (-134, 39, 22, 0.5),
          (-116, 36, 23, 2.0), (-84, 32, 21, 1.0), (-48, 33, 22, 0.0)]
_HEM_V, _HEM_W, _HEM_D = 74.0, 55.0, 42.0
_SKIRT_TOP = (-50.0, 33.5, 22.5)


def _ring_at(v):
    """Interpolated torso/skirt cross-section at height v (rel hip)."""
    T = _TORSO
    if v <= T[0][0]:
        return T[0][1:]
    for a, b in zip(T, T[1:]):
        if v <= b[0]:
            u = (v - a[0]) / (b[0] - a[0])
            return tuple(lerp(a[i], b[i], u) for i in (1, 2, 3))
    u = clamp((v - _SKIRT_TOP[0]) / (_HEM_V - _SKIRT_TOP[0]))
    return (lerp(_SKIRT_TOP[1], _HEM_W, u ** 0.9), lerp(_SKIRT_TOP[2], _HEM_D, u ** 0.9), 0.0)


def _ik(S, H, La, Lb, pole):
    """3-D two-bone IK. Returns (elbow, wrist)."""
    d = [H[i] - S[i] for i in range(3)]
    dl = math.sqrt(sum(x * x for x in d)) or 1e-6
    reach = (La + Lb) * 0.999
    if dl > reach:
        d = [x * reach / dl for x in d]
        H = [S[i] + d[i] for i in range(3)]
        dl = reach
    dn = [x / dl for x in d]
    a = (La * La - Lb * Lb + dl * dl) / (2 * dl)
    h = math.sqrt(max(La * La - a * a, 0.0))
    pd = sum(pole[i] * dn[i] for i in range(3))
    pp = [pole[i] - pd * dn[i] for i in range(3)]
    pl = math.sqrt(sum(x * x for x in pp)) or 1e-6
    pp = [x / pl for x in pp]
    E = [S[i] + dn[i] * a + pp[i] * h for i in range(3)]
    return E, list(H)


# =============================================================== Haru — expressions
_EXPR = {
    #            brow_y brow_tilt open  lid_tilt lower iris  mouth       smile blush
    "neutral":   (0.00, 0.00, 1.00, 0.00, 0.05, 1.00, "smile_s", 0.35, 0.55),
    "happy":     (0.25, 0.10, 0.92, 0.00, 0.38, 1.00, "smile", 1.00, 0.95),
    "surprised": (1.00, 0.20, 1.12, 0.00, 0.00, 0.78, "o", 0.00, 0.60),
    "determined": (-0.45, -0.85, 0.92, -0.35, 0.18, 1.00, "firm", 0.25, 0.55),
    "sad":       (0.20, 1.00, 0.86, 0.35, 0.10, 1.00, "frown", -0.8, 0.55),
    "gentle":    (0.12, 0.30, 0.78, 0.10, 0.30, 1.00, "smile", 0.65, 0.95),
    "sleepy":    (-0.10, 0.35, 0.42, 0.20, 0.05, 1.00, "small", 0.05, 0.50),
    "worried":   (0.40, 1.00, 0.97, 0.20, 0.05, 0.92, "wavy", -0.3, 0.55),
    "laugh":     (0.30, 0.10, 0.00, 0.00, 0.00, 1.00, "smile", 1.00, 1.00),
}

# vowel -> (half-width, height, roundness, teeth)
_VOWEL = {
    "a": (7.6, 10.0, 0.55, 0.35),
    "i": (8.4, 4.6, 0.25, 0.60),
    "u": (4.4, 5.6, 1.00, 0.00),
    "e": (7.4, 7.0, 0.40, 0.45),
    "o": (5.6, 9.0, 0.95, 0.00),
    "n": (4.5, 1.2, 0.50, 0.00),
    "x": (5.0, 0.0, 0.50, 0.00),
}

R_HEAD = 58.0
HEAD_V = -230.0          # head centre above hip (rel), body space
HIP_H = 230.0            # standing hip height above ground
L_THIGH, L_SHIN, FOOT_H = 96.0, 96.0, 13.0
L_UPPER, L_FORE = 66.0, 61.0
SHOULDER = (35.0, -136.0)


# ----------------------------------------------------------------- head
class _Head:
    def __init__(self, ya, pitch, expr, t, wind, blink, look, mouth_open, vowel,
                 overrides, rim, light_sign):
        self.ya = ya
        self.pitch = pitch
        self.t = t
        self.wind = wind
        self.blink = blink
        self.look = look
        self.mouth_open = mouth_open
        self.vowel = vowel
        self.rim = rim
        self.ls = light_sign
        self.back = math.cos(ya) < -0.35
        e = list(_EXPR.get(expr, _EXPR["neutral"]))
        self.expr = expr
        (self.brow_y, self.brow_tilt, self.eye_open, self.lid_tilt, self.lower,
         self.iris, self.mouth_kind, self.smile, self.blush) = e
        for k, v in overrides.items():
            if v is not None:
                setattr(self, k, v)

    # sphere projection of a head-surface feature
    def sp(self, phi, y, rad=1.0):
        R = R_HEAD
        ry = R * rad * math.sqrt(max(0.0, 1 - (y / (1.3 * R)) ** 2))
        a = phi + self.ya
        z = math.cos(a)
        return (ry * math.sin(a), y - self.pitch * 0.32 * R * z), z

    # ---------------------------------------------------- outlines
    def face_pts(self):
        R = R_HEAD
        F = [(0, -1.0), (0.62, -0.86), (0.95, -0.42), (1.0, 0.05), (0.99, 0.2), (0.96, 0.36),
             (0.84, 0.62), (0.56, 0.86), (0.0, 0.99), (-0.56, 0.86), (-0.84, 0.62), (-0.96, 0.36),
             (-0.99, 0.2), (-1.0, 0.05), (-0.95, -0.42), (-0.62, -0.86)]
        S = [(0.05, -1.0), (0.6, -0.88), (0.9, -0.5), (0.97, -0.08), (1.05, 0.25), (0.97, 0.42),
             (0.93, 0.66), (0.76, 0.88), (0.42, 0.9), (0.1, 0.72), (-0.3, 0.55), (-0.7, 0.4),
             (-0.9, 0.2), (-0.98, 0.0), (-0.95, -0.42), (-0.62, -0.86)]
        w = abs(math.sin(self.ya)) ** 1.3
        pts = []
        for i, (a, b) in enumerate(zip(F, S)):
            x = lerp(a[0], b[0], w) * R
            y = lerp(a[1], b[1], w) * R
            pts.append((x, y + self.pitch * 0.1 * R * (1 if y > 0 else 0)))
        # nose tip sharp only when near profile
        if w > 0.75:
            pts[4] = (pts[4][0], pts[4][1], True)
        return pts

    def hair_pts(self):
        """Back-hair / bob silhouette (includes the crown)."""
        R = R_HEAD
        wn = self.wind_dx()
        F = [(0, -1.24), (0.72, -1.06), (1.12, -0.52), (1.24, 0.1), (1.24, 0.56), (1.2, 0.86, True),
             (1.06, 0.78), (0.98, 0.9, True), (0.86, 0.72), (0.5, 0.6), (0, 0.55), (-0.5, 0.6),
             (-0.86, 0.72), (-0.98, 0.9, True), (-1.06, 0.78), (-1.2, 0.86, True), (-1.24, 0.56),
             (-1.24, 0.1), (-1.12, -0.52), (-0.72, -1.06)]
        S = [(-0.05, -1.24), (0.62, -1.08), (1.05, -0.6), (1.02, -0.2), (0.6, 0.1), (0.52, 0.5, True),
             (0.36, 0.44), (0.2, 0.62, True), (0.0, 0.52), (-0.2, 0.7), (-0.45, 0.75), (-0.7, 0.82),
             (-0.95, 0.78), (-1.08, 0.94, True), (-1.16, 0.8), (-1.3, 0.9, True), (-1.38, 0.5),
             (-1.4, 0.0), (-1.26, -0.6), (-0.78, -1.08)]
        B = [(0, -1.24), (0.72, -1.06), (1.12, -0.52), (1.24, 0.1), (1.24, 0.56), (1.18, 0.9, True),
             (1.02, 0.82), (0.86, 0.96, True), (0.68, 0.84), (0.4, 0.98, True), (0.15, 0.86), (-0.12, 0.99, True),
             (-0.38, 0.86), (-0.62, 0.97, True), (-0.84, 0.84), (-1.0, 0.95, True), (-1.2, 0.86),
             (-1.24, 0.1), (-1.12, -0.52), (-0.72, -1.06)]
        if self.back:
            src = B
            w = 0.0
            A = B
        else:
            w = abs(math.sin(self.ya)) ** 1.1
            A = F
            src = S
        pts = []
        for a, b in zip(A, src):
            x = lerp(a[0], b[0], w) * R
            y = lerp(a[1], b[1], w) * R
            sharp = len(a) > 2 or len(b) > 2
            # sway: lower points move with wind
            k = clamp((y / R + 0.2) / 1.1) ** 1.5
            x += wn * k
            pts.append((x, y, True) if sharp else (x, y))
        return pts

    def wind_dx(self, ph=0.0):
        t = self.t
        base = self.wind * 7.0
        return base * (0.55 + 0.45 * math.sin(t * 3.1 + ph)) + 1.2 * math.sin(t * 1.3 + ph * 1.7)

    # ---------------------------------------------------- drawing
    def draw(self, ctx):
        lw = _LW[0]
        R = R_HEAD
        if self.back:
            self._draw_back(ctx)
            return
        # --- back hair mass
        hp = self.hair_pts()
        _cel(ctx, lambda c: _spline(c, hp), HAIR, HAIR_SH, self.rim, HAIR_LINE, lw,
             soff=(-7 * self.ls, -5), roff=(-2.6 * self.ls, 0.5), rim_a=0.55)
        # far side lock behind face
        self._side_lock(ctx, -1, behind=True)
        self._side_lock(ctx, 1, behind=True)
        # --- face
        fp = self.face_pts()

        def face_path(c):
            _spline(c, fp)

        def face_extra(c):
            # shadow cast by the fringe / hair onto the forehead
            c.save()
            c.translate(1.5 * self.ls, 7.0)
            self._fringe_path(c)
            _sc(c, SKIN_SH, 0.9)
            c.fill()
            c.restore()
            # neck shadow under jaw is drawn on the neck
            self._blush(c)

        _cel(ctx, face_path, SKIN, SKIN_SH, None, SKIN_LINE, lw * 0.9,
             soff=(-6 * self.ls, -3.5), shade_a=0.55, extra=face_extra)
        # --- features
        self._eyes(ctx)
        self._nose(ctx)
        self._mouth(ctx)
        # --- hair covering the side/back of the skull (3/4 & profile)
        self._side_hair(ctx, hp)
        # --- fringe, locks, cowlick, clip
        self._fringe(ctx)
        self._side_lock(ctx, -1, behind=False)
        self._side_lock(ctx, 1, behind=False)
        self._cowlick(ctx)
        self._clip(ctx)
        self._brows(ctx)

    # ---------- back view
    def _draw_back(self, ctx):
        lw = _LW[0]
        R = R_HEAD
        hp = self.hair_pts()

        def extra(c):
            # strand lines radiating from the crown whorl
            _sc(c, HAIR_SH, 0.9)
            crown = (0.1 * R, -0.62 * R)
            for i, a in enumerate([-1.1, -0.65, -0.25, 0.2, 0.6, 1.05]):
                end = (math.sin(a) * 1.12 * R + self.wind_dx(i) * 0.8, 0.84 * R)
                mid = (lerp(crown[0], end[0], 0.5) + math.sin(a) * 8, lerp(crown[1], end[1], 0.55))
                _taper(c, [crown, mid, end], 0.5, 0.4, 3.0)
            self._angel_ring(c, back=True)

        _cel(ctx, lambda c: _spline(c, hp), HAIR, HAIR_SH, self.rim, HAIR_LINE, lw,
             soff=(-8 * self.ls, -6), roff=(-2.6 * self.ls, 0.5), rim_a=0.55, extra=extra)
        self._cowlick(ctx)
        self._clip(ctx)

    def _hairline(self):
        R = R_HEAD
        pts = []
        for i in range(12):
            y = -1.5 * R + 2.4 * R * i / 11
            yy = clamp(y, -1.2 * R, 1.2 * R)
            ph = -1.3 + 0.12 * clamp((y / R - 0.1), 0, 1)
            (x, y2), z = self.sp(ph, yy * 0.95, 1.02)
            pts.append((x, y))
        return pts

    def _side_hair(self, ctx, hp):
        if math.sin(self.ya) < 0.25:
            return
        R = R_HEAD
        hl = self._hairline()
        wn = self.wind_dx(1.0)

        def region(c):
            c.move_to(-4 * R, hl[0][1])
            for p in hl:
                c.line_to(*p)
            c.line_to(-4 * R, hl[-1][1])
            c.close_path()

        ctx.save()
        ctx.new_path()
        _spline(ctx, hp)
        ctx.clip()
        ctx.new_path()
        region(ctx)
        ctx.clip()

        def extra(c):
            _sc(c, HAIR_SH, 1.0)
            for k, off in enumerate((0.25, 0.6, 0.95)):
                x0 = hl[4][0] - off * R
                _taper(c, [(x0 + 0.1 * R, -0.9 * R), (x0 - 0.05 * R, 0.0), (x0 + wn * 0.5, 0.8 * R)], 0.4, 0.3, 2.6)
            self._angel_ring(c)

        _cel(ctx, lambda c: _spline(c, hp), HAIR, HAIR_SH, self.rim, None, None,
             soff=(-7 * self.ls, -5), roff=(-2.6 * self.ls, 0.5), rim_a=0.55, extra=extra)
        ctx.restore()
        # outline along the hairline (only inside the hair silhouette)
        ctx.save()
        ctx.new_path()
        _spline(ctx, hp)
        ctx.clip()
        ctx.new_path()
        _spline(ctx, hl, False)
        _sc(ctx, HAIR_LINE)
        ctx.set_line_width(_LW[0])
        ctx.stroke()
        ctx.restore()

    # ---------- fringe
    def _fringe_line(self):
        R = R_HEAD
        pts = []
        N = 13
        span = 1.62
        for j in range(N):
            phi = -span + 2 * span * j / (N - 1)
            u = abs(phi) / span
            y = -0.33 * R + 0.30 * R * u ** 2.2
            tip = j % 2 == 1
            if tip:
                y += 0.07 * R
            if j in (4, 8):
                y += 0.03 * R
            (x, yy), z = self.sp(phi, y, 1.06 + 0.1 * u ** 3)
            sway = self.wind_dx(j * 0.7) * 0.25
            pts.append((x + sway, yy, z, tip))
        return pts

    def _fringe_path(self, c):
        R = R_HEAD
        line = [p for p in self._fringe_line() if p[2] > -0.35]
        if len(line) < 2:
            return
        # closed polygon: scalloped bottom (left->right), then up over the crown
        pts = [(p[0], p[1], p[3]) for p in line]
        x_r, y_r = pts[-1][0], pts[-1][1]
        x_l, y_l = pts[0][0], pts[0][1]
        top = [(x_r + 6, y_r - 0.5 * R), (0.2 * R - 0.25 * R * math.sin(self.ya), -1.6 * R),
               (x_l - 6, y_l - 0.5 * R)]
        _spline(c, pts + top, True, 0.9)

    def _fringe(self, ctx):
        lw = _LW[0]
        R = R_HEAD
        hp = self.hair_pts()

        def clipped(c):
            # fringe = hair silhouette ∩ fringe polygon
            pass

        ctx.save()
        ctx.new_path()
        _spline(ctx, hp)
        # a slightly larger cap so the fringe fills to the silhouette
        ctx.clip()
        ctx.new_path()

        def fpath(c):
            self._fringe_path(c)

        def extra(c):
            # strand separation lines from each notch upward
            line = self._fringe_line()
            _sc(c, HAIR_SH, 1.0)
            for i, p in enumerate(line):
                if p[3] or p[2] < 0.05 or i in (0, len(line) - 1):
                    continue
                top = (p[0] * 0.55, p[1] - 0.72 * R)
                mid = (lerp(p[0], top[0], 0.4), lerp(p[1], top[1], 0.45))
                _taper(c, [(p[0], p[1] + 1), mid, top], 2.4, 0.2)
            self._angel_ring(c)

        _cel(ctx, fpath, HAIR, HAIR_SH, self.rim, None, lw, soff=(-6 * self.ls, -6),
             roff=(-2.6 * self.ls, 0.5), rim_a=0.55, extra=extra)
        ctx.restore()
        # outline only along the scalloped bottom edge
        line = [p for p in self._fringe_line() if p[2] > -0.35]
        pts = [(p[0], p[1], p[3]) for p in line]
        ctx.new_path()
        _spline(ctx, pts, False, 0.9)
        _sc(ctx, HAIR_LINE)
        ctx.set_line_width(lw)
        ctx.stroke()
        # re-stroke silhouette top so fringe/back hair read as one mass
        ctx.save()
        ctx.rectangle(-3 * R, -3 * R, 6 * R, 3 * R - 0.05 * R)
        ctx.clip()
        ctx.new_path()
        _spline(ctx, hp)
        _sc(ctx, HAIR_LINE)
        ctx.set_line_width(lw)
        ctx.stroke()
        ctx.restore()

    def _angel_ring(self, c, back=False):
        """The glossy highlight band ("angel ring") on the crown: a thin band
        with a jagged lower edge broken by strands."""
        R = R_HEAD
        y0 = -0.7 * R
        n = 15
        up, lo = [], []
        for j in range(n):
            phi = -1.3 + 2.6 * j / (n - 1)
            if back:
                phi += PI
            (x, y), z = self.sp(phi, y0, 1.2)
            if z < 0.05:
                continue
            k = min(1.0, z * 2.5)
            dip = (0.11 if j % 2 else 0.035) * R * k
            up.append((x, y - 0.035 * R * k))
            lo.append((x, y + dip, True) if j % 2 else (x, y + dip))
        if len(up) < 3:
            return
        c.new_path()
        _spline(c, up, False, 1.0)
        for p in reversed(lo):
            c.line_to(p[0], p[1])
        c.close_path()
        _sc(c, HAIR_HI, 0.75)
        c.fill()

    def _side_lock(self, ctx, side, behind):
        """Bob side locks framing the face (side=-1 her right, +1 her left)."""
        R = R_HEAD
        phi = side * 1.38
        (rx, ry), z = self.sp(phi, -0.25 * R, 1.08)
        front = z > 0.02
        if front == behind:
            return
        (mx, my), _ = self.sp(side * 1.3, 0.35 * R, 1.1)
        (tx, ty), _ = self.sp(side * 1.12, 0.84 * R, 1.02)
        sw = self.wind_dx(side * 2.0)
        tx += sw
        mx += sw * 0.5
        ww = 0.3 * R * (0.4 + 0.6 * max(z, 0.15))
        # direction across the lock (screen): outward from the face centre
        o = 1 if math.sin(phi + self.ya) > 0 else -1
        pts = [(rx - o * ww * 0.3, ry - 0.3 * R), (mx - o * ww * 0.8, my), (tx - o * 2, ty, True),
               (mx + o * ww * 0.9, my + 0.05 * R), (rx + o * ww * 1.0, ry)]

        def extra(c):
            _sc(c, HAIR_SH, 1)
            _taper(c, [(rx, ry), (mx + o * 2, my), (tx - o * 1, ty - 4)], 0.3, 0.2, 2.0)

        path = lambda c: _spline(c, pts)
        if behind:
            _cel(ctx, path, HAIR, HAIR_SH, None, HAIR_LINE, _LW[0], soff=(-4 * self.ls, -3))
            return
        _cel(ctx, path, HAIR, None, None, None, None, extra=extra)
        # outline only the face-side edge + tip
        ctx.new_path()
        _spline(ctx, [pts[3], (pts[2][0], pts[2][1]), pts[1]][::-1] if False else [pts[4], pts[3], (pts[2][0], pts[2][1], True), pts[1]], False)
        _sc(ctx, HAIR_LINE)
        ctx.set_line_width(_LW[0])
        ctx.stroke()

    def _cowlick(self, ctx):
        R = R_HEAD
        sgn = 1.0
        base_x = 0.12 * R - 0.35 * R * math.sin(self.ya) * (0 if self.back else 1)
        if self.back:
            base_x = -0.1 * R
        b = (base_x, -1.2 * R)
        sw = math.sin(self.t * 2.3) * 2.5 + self.wind_dx(3.0) * 0.6
        p1 = (b[0] + 0.12 * R * sgn + sw * 0.5, b[1] - 0.26 * R)
        p2 = (b[0] + 0.3 * R * sgn + sw, b[1] - 0.3 * R)
        p3 = (b[0] + 0.36 * R * sgn + sw * 1.2, b[1] - 0.18 * R)
        pts = [(b[0] - 5, b[1] + 3), (b[0] + 0.02 * R, b[1] - 0.2 * R), p1, p2, (p3[0], p3[1], True),
               (p2[0] - 1, p2[1] + 3.5), (p1[0] + 1, p1[1] + 4.5), (b[0] + 5, b[1] + 3)]
        _cel(ctx, lambda c: _spline(c, pts), HAIR, None, self.rim, HAIR_LINE, _LW[0] * 0.85,
             roff=(-1.6 * self.ls, 0.4), rim_a=0.6)

    def _clip(self, ctx):
        """Red star hair clip on her right side of the fringe."""
        R = R_HEAD
        (x, y), z = self.sp(-0.95, -0.52 * R, 1.16)
        if z < -0.62 and not self.back:
            return
        s = 0.2 * R
        fs = 0.55 + 0.45 * abs(z)
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(-0.25)
        ctx.scale(fs, 1)
        path = lambda c: _rounded_star(c, 0, 0, s, s * 0.5, -PI / 2, 0.45)
        _cel(ctx, path, CLIP, CLIP_SH, "#FF9AA0", CLIP_LINE, _LW[0] * 0.8,
             soff=(-2.5, -2.5), roff=(1.2, 1.2), rim_a=0.8)
        ctx.arc(-s * 0.25, -s * 0.28, s * 0.14, 0, TAU)
        ctx.set_source_rgba(1, 1, 1, 0.85)
        ctx.fill()
        ctx.restore()

    # ---------- blush
    def _blush(self, c):
        R = R_HEAD
        if self.blush <= 0:
            return
        for side in (-1, 1):
            (x, y), z = self.sp(side * 0.72, 0.36 * R, 0.98)
            if z < 0.15:
                continue
            fs = 0.35 + 0.65 * z
            c.save()
            c.translate(x, y)
            c.scale(fs, 0.62)
            g = cairo.RadialGradient(0, 0, 0, 0, 0, 0.22 * R)
            col = _c(CHEEK)
            g.add_color_stop_rgba(0, *col, 0.75 * self.blush)
            g.add_color_stop_rgba(0.55, *col, 0.45 * self.blush)
            g.add_color_stop_rgba(1, *col, 0)
            c.set_source(g)
            c.arc(0, 0, 0.22 * R, 0, TAU)
            c.fill()
            # little hatch lines
            _sc(c, "#E0707A", 0.55 * self.blush)
            for k in (-1, 0, 1):
                c.save()
                c.translate(k * 0.075 * R, 0)
                c.move_to(0.02 * R, -0.07 * R)
                c.line_to(-0.02 * R, 0.07 * R)
                c.set_line_width(1.1)
                c.stroke()
                c.restore()
            c.restore()

    # ---------- eyes
    def _eyes(self, ctx):
        R = R_HEAD
        self.eye_pos = []
        for side in (-1, 1):
            (x, y), z = self.sp(side * 0.46, 0.13 * R, 0.93)
            if z < 0.08:
                continue
            fs = 0.42 + 0.58 * z
            # keep far eye inside the silhouette
            self.eye_pos.append((x, y))
            self._eye(ctx, x, y, side, fs)

    def _eye(self, ctx, cx, cy, side, fs):
        R = R_HEAD
        a = 0.215 * R      # half width
        b = 0.285 * R      # half height
        op = self.eye_open * (1 - self.blink)
        lid_t = self.lid_tilt
        low = self.lower
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(fs, 1)
        # outer corner toward +x in local space: her right eye (side -1) outer is screen-left
        flip = -1 if side < 0 else 1
        # ---- closed / happy-arc eyes
        if op < 0.12 or self.expr == "laugh":
            ctx.scale(flip, 1)
            _sc(ctx, LASH)
            if self.expr == "laugh" or (self.lower > 0.3 and op < 0.12 and self.smile > 0.5):
                pts = [(-a * 0.95, b * 0.25), (0, -b * 0.35), (a * 1.0, b * 0.25)]
            else:
                pts = [(-a * 0.95, b * 0.05), (0, b * 0.42), (a * 1.05, b * 0.0)]
            _taper(ctx, pts, 1.6, 2.6, 3.8)
            if not self.expr == "laugh":
                _taper(ctx, [(a * 0.95, b * 0.02), (a * 1.25, -b * 0.15)], 2.4, 0.4)
            ctx.restore()
            return
        top_y = -b * (0.35 + 0.65 * op)  # upper lid top at centre
        bot_y = b * (1.0 - 0.7 * low)

        def ty(u):  # top lid y at u in [-1,1] (-1 inner, +1 outer)
            base = top_y * (1 - 0.55 * u * u) - b * 0.05
            return base + lid_t * b * 0.45 * (u + 0.3) * op

        def by(u):
            return bot_y * (1 - 0.8 * u * u) ** 0.8 + b * 0.1 * u

        ctx.scale(flip, 1)
        us = [-1.0, -0.6, -0.1, 0.45, 0.85, 1.0]
        top = [(-a, ty(-1) * 0.6 + b * 0.1)] + [(u * a, ty(u)) for u in us[1:-1]] + [(a * 1.02, ty(1) * 0.7 + b * 0.02)]
        bot = [(0.78 * a, by(0.78) * 0.85), (0.1 * a, by(0.1)), (-0.7 * a, by(-0.7) * 0.9)]
        white = [(top[0][0], top[0][1], True)] + top[1:-1] + [(top[-1][0], top[-1][1], True)] + bot
        ctx.new_path()
        _spline(ctx, white)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.save()
        ctx.clip()
        # ---- iris (undo flip so highlights share world light direction)
        ctx.scale(flip, 1)
        lx, ly = self.look
        ix = lx * a * 0.42 * flip * flip
        iy = b * 0.1 + ly * b * 0.22
        irx, iry = a * 0.78 * self.iris, b * 0.86 * self.iris
        g = cairo.LinearGradient(0, iy - iry, 0, iy + iry)
        for o, col in ((0, IRIS_TOP), (0.45, IRIS_MID), (1, IRIS_BOT)):
            g.add_color_stop_rgb(o, *_c(col))
        ctx.save()
        ctx.translate(ix, iy)
        ctx.scale(irx, iry)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        ctx.set_source(g)
        ctx.fill()
        # lower iris glow
        ctx.save()
        ctx.translate(ix, iy + iry * 0.45)
        ctx.scale(irx * 0.7, iry * 0.38)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        _sc(ctx, "#C99AB6", 0.45)
        ctx.fill()
        # pupil
        ctx.save()
        ctx.translate(ix, iy - iry * 0.05)
        ctx.scale(irx * 0.48, iry * 0.52)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        _sc(ctx, "#0C0610", 0.95)
        ctx.fill()
        # iris ring
        ctx.save()
        ctx.translate(ix, iy)
        ctx.scale(irx, iry)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        _sc(ctx, "#140A18", 0.9)
        ctx.set_line_width(1.2)
        ctx.stroke()
        # lid shadow across the top of the eye
        ctx.scale(flip, 1)
        ctx.new_path()
        _spline(ctx, [(-a * 1.2, top_y - b), (a * 1.2, top_y - b), (a * 1.2, ty(1) + b * 0.28),
                      (0, ty(0) + b * 0.36), (-a * 1.2, ty(-1) + b * 0.3)], True, 0.6)
        _sc(ctx, "#3A2A5A", 0.28)
        ctx.fill()
        ctx.scale(flip, 1)
        # catch-lights (world-fixed: upper toward the light)
        hl = -self.ls
        ctx.save()
        ctx.translate(ix + hl * irx * 0.3, iy - iry * 0.38)
        ctx.rotate(-0.3 * hl)
        ctx.scale(irx * 0.42, iry * 0.3)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        ctx.set_source_rgba(1, 1, 1, 0.97)
        ctx.fill()
        ctx.arc(ix - hl * irx * 0.38, iy + iry * 0.42, irx * 0.17, 0, TAU)
        ctx.set_source_rgba(1, 1, 1, 0.9)
        ctx.fill()
        ctx.arc(ix + hl * irx * 0.52, iy + iry * 0.05, irx * 0.08, 0, TAU)
        ctx.set_source_rgba(1, 1, 1, 0.75)
        ctx.fill()
        ctx.restore()  # unclip
        ctx.new_path()
        # ---- upper lash line (thick, flick at the outer corner)
        _sc(ctx, LASH)
        lash = [(top[0][0] - a * 0.02, top[0][1] + b * 0.05)] + top[1:-1] + [(a * 1.03, ty(1) * 0.7)]
        _taper(ctx, lash, 1.4, 3.6, 3.6)
        _taper(ctx, [(a * 0.72, ty(0.72) - 0.5), (a * 1.08, ty(1) * 0.8), (a * 1.32, ty(1) * 1.05 - b * 0.1)],
               2.8, 0.3)
        # lower lash hint
        _sc(ctx, LASH, 0.7)
        _taper(ctx, [(a * 0.98, ty(1) * 0.55 + b * 0.25), (a * 0.82, by(0.8) * 0.8), (a * 0.45, by(0.45) * 0.97)],
               1.5, 0.3)
        # thin crease line above the lid
        _sc(ctx, SKIN_LINE, 0.35)
        _taper(ctx, [(-a * 0.3, ty(-0.3) - b * 0.28), (a * 0.4, ty(0.4) - b * 0.3), (a * 0.85, ty(0.85) - b * 0.18)],
               0.3, 0.3, 1.1)
        ctx.restore()

    def _brows(self, ctx):
        R = R_HEAD
        for side in (-1, 1):
            (x, y), z = self.sp(side * 0.44, -0.3 * R - self.brow_y * 0.1 * R, 1.08)
            if z < 0.1:
                continue
            fs = 0.42 + 0.58 * z
            flip = -1 if side < 0 else 1
            ctx.save()
            ctx.translate(x, y)
            ctx.scale(fs * flip, 1)
            tl = self.brow_tilt * 0.08 * R
            pts = [(-0.2 * R, tl * -1.0 + 0.01 * R), (0.0, -0.035 * R - tl * 0.1), (0.2 * R, tl * 0.7 + 0.02 * R)]
            _sc(ctx, "#3A2230", 0.85)
            _taper(ctx, pts, 2.6, 0.8, 2.6)
            ctx.restore()

    def _nose(self, ctx):
        R = R_HEAD
        (x, y), z = self.sp(0.0, 0.43 * R, 0.99)
        if z < 0.0:
            return
        side = abs(math.sin(self.ya))
        if side > 0.8:
            return  # profile nose is part of the face outline
        _sc(ctx, SKIN_LINE, 0.55)
        o = 1 if self.ya >= 0 else -1
        _taper(ctx, [(x - 1.2 * o, y - 2.5), (x + 1.4 * o, y + 1.0)], 0.4, 1.6)

    def _mouth(self, ctx):
        R = R_HEAD
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
        v = self.vowel if self.vowel in _VOWEL else "x"
        if v in ("x", "n") and op < 0.25:
            op = 0.0
        smile = self.smile
        if op < 0.06:
            self._closed_mouth(ctx, v)
            ctx.restore()
            return
        hw, hh, rnd, teeth = _VOWEL[v]
        hw = hw * (0.85 + 0.15 * op)
        hh = max(1.6, hh * op * 1.05)
        cu = smile * 1.6  # corner lift for happy talking
        top = -hh * 0.25 - cu * 0.3
        # outline: corners, top lip (flatter), bottom (rounder)
        L, Rr = (-hw, -cu), (hw, -cu)
        ctx.new_path()
        ctx.move_to(*L)
        ctx.curve_to(-hw * 0.5, top - hh * 0.15 * rnd, hw * 0.5, top - hh * 0.15 * rnd, *Rr)
        bk = 0.55 + 0.45 * rnd
        ctx.curve_to(hw * (1.0 - 0.1 * rnd), hh * bk, hw * 0.45, hh * 1.05, 0, hh * 1.02)
        ctx.curve_to(-hw * 0.45, hh * 1.05, -hw * (1.0 - 0.1 * rnd), hh * bk, *L)
        ctx.close_path()
        path = ctx.copy_path()
        _sc(ctx, MOUTH_IN)
        ctx.fill_preserve()
        ctx.save()
        ctx.clip()
        # tongue
        ctx.save()
        ctx.translate(0, hh * 1.1)
        ctx.scale(hw * 0.75, hh * 0.62)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        _sc(ctx, TONGUE)
        ctx.fill()
        if teeth > 0:
            ctx.rectangle(-hw, top - hh, hw * 2, hh * (0.55 + teeth * 0.2))
            ctx.set_source_rgba(1, 1, 1, 0.95)
            ctx.fill()
        ctx.restore()
        ctx.new_path()
        ctx.append_path(path)
        _sc(ctx, MOUTH_LINE)
        ctx.set_line_width(_LW[0] * 0.7)
        ctx.stroke()
        ctx.restore()

    def _closed_mouth(self, ctx, v):
        R = R_HEAD
        k = self.mouth_kind
        _sc(ctx, MOUTH_LINE)
        s = self.smile
        if v == "n":
            _taper(ctx, [(-4.0, 0.5), (0, 1.2), (4.0, 0.5)], 1.0, 1.0, 2.0)
            return
        if k == "o":
            ctx.save()
            ctx.scale(3.6, 4.6)
            ctx.arc(0, 0.2, 1, 0, TAU)
            ctx.restore()
            _sc(ctx, MOUTH_IN)
            ctx.fill_preserve()
            _sc(ctx, MOUTH_LINE)
            ctx.set_line_width(_LW[0] * 0.6)
            ctx.stroke()
            return
        if k == "wavy":
            _taper(ctx, [(-6, 1), (-3, -0.8), (0, 0.8), (3, -0.6), (6, 1.2)], 0.8, 0.8, 1.8)
            return
        if k == "frown":
            _taper(ctx, [(-5.5, 1.8), (0, -1.0), (5.5, 1.8)], 0.8, 0.8, 2.0)
            return
        if k == "firm":
            _taper(ctx, [(-6, -0.5), (0, 0.6), (5, -0.8)], 0.8, 0.8, 2.2)
            return
        if k == "small":
            _taper(ctx, [(-3, 0), (0, 0.9), (3, 0)], 0.8, 0.8, 1.8)
            return
        if k == "smile" and s > 0.8:
            # open little happy smile ("D" shape)
            ctx.new_path()
            ctx.move_to(-6.5, -1.5)
            ctx.curve_to(-3, 0, 3, 0, 6.5, -1.5)
            ctx.curve_to(5, 6.5, -5, 6.5, -6.5, -1.5)
            ctx.close_path()
            p = ctx.copy_path()
            _sc(ctx, MOUTH_IN)
            ctx.fill_preserve()
            ctx.save()
            ctx.clip()
            ctx.save()
            ctx.translate(0, 6)
            ctx.scale(4.5, 3.2)
            ctx.arc(0, 0, 1, 0, TAU)
            ctx.restore()
            _sc(ctx, TONGUE)
            ctx.fill()
            ctx.restore()
            ctx.new_path()
            ctx.append_path(p)
            _sc(ctx, MOUTH_LINE)
            ctx.set_line_width(_LW[0] * 0.6)
            ctx.stroke()
            return
        w = 4.5 + 2.5 * s
        _taper(ctx, [(-w, -s * 2.2), (0, 0.8 + s * 0.8), (w, -s * 2.2)], 0.8, 0.8, 2.0)


# =============================================================== Haru — body parts
def _pattern_arcs(ctx, pts, color=PATTERN, alpha=0.55, lw=1.2):
    """Seigaiha-like wave motif: (x, y, size, squash) list."""
    _sc(ctx, color, alpha)
    ctx.set_line_width(lw)
    for (x, y, s, fx) in pts:
        for r in (1.0, 0.62, 0.28):
            ctx.save()
            ctx.translate(x, y)
            ctx.scale(fx, 1)
            ctx.new_path()
            ctx.arc(0, 0, s * r, PI * 1.05, PI * 1.95)
            ctx.restore()
            ctx.stroke()


# fixed motif positions on the garment surface (phi, v)
_MOTIFS = [(-2.6, -120), (-1.7, -104), (-0.85, -126), (0.1, -104), (0.95, -124), (1.8, -104), (2.7, -122),
           (-2.2, -64 + 50), (-1.3, -2), (-0.35, 12), (0.6, -4), (1.5, 14), (2.4, -2),
           (-2.7, 44), (-1.8, 58), (-0.9, 40), (0.05, 60), (1.0, 42), (1.9, 60), (2.9, 46), (3.3, 10)]


def _hand(ctx, W, d, size, kind, side_sign, rim, ls):
    """A small mitten-like hand at wrist W pointing along d (unit 2D)."""
    ang = math.atan2(d[1], d[0])
    ctx.save()
    ctx.translate(*W)
    ctx.rotate(ang)
    s = size
    if kind == "fist":
        pts = [(-0.2 * s, -0.55 * s), (0.55 * s, -0.7 * s), (1.1 * s, -0.35 * s), (1.15 * s, 0.35 * s),
               (0.6 * s, 0.72 * s), (-0.2 * s, 0.55 * s)]
    else:
        pts = [(-0.2 * s, -0.5 * s), (0.6 * s, -0.62 * s), (1.35 * s, -0.4 * s), (1.6 * s, 0.0),
               (1.35 * s, 0.42 * s), (0.6 * s, 0.6 * s), (-0.2 * s, 0.5 * s)]
    th = side_sign
    thumb = [(0.2 * s, th * 0.45 * s), (0.6 * s, th * 0.95 * s), (0.95 * s, th * 0.8 * s), (0.75 * s, th * 0.4 * s)]
    _cel(ctx, lambda c: _spline(c, thumb), SKIN, None, None, SKIN_LINE, _LW[0] * 0.8)
    _cel(ctx, lambda c: _spline(c, pts), SKIN, SKIN_SH, None, SKIN_LINE, _LW[0] * 0.8,
         soff=(1.5, -2.5 * th), shade_a=0.8)
    if kind != "fist":
        _sc(ctx, SKIN_LINE, 0.45)
        ctx.set_line_width(0.9)
        for k in (-0.18, 0.15):
            ctx.move_to(1.0 * s, k * s)
            ctx.line_to(1.45 * s, k * s * 1.2)
            ctx.stroke()
    ctx.restore()


def _sleeve_path(S, E, W, hang, flutter):
    """Furisode-style yukata sleeve around the upper arm, pouch hanging by gravity."""
    u = _norm(_sub(E, S))
    L = _len(_sub(E, S))
    n = (-u[1], u[0])
    if n[1] < 0:  # n points 'down' (toward gravity)
        n = (-n[0], -n[1])
    hw = 19.0
    cuff = _add(S, u, L * 1.05)
    down = (flutter, 1.0)
    # gravity pouch: longer when the arm is horizontal, short when hanging/raised
    hang_k = 1.0 - 0.55 * abs(u[1])
    H = hang * hang_k
    P = [
        _add(_add(S, n, -hw * 0.9), u, -9),
        _add(_add(S, n, -hw * 1.05), u, L * 0.5),
        _add(cuff, n, -hw * 1.05),
        (_add(cuff, n, hw * 0.9)[0], _add(cuff, n, hw * 0.9)[1], True),
    ]
    pouch_o = _add(_add(_add(S, u, L * 0.92), n, hw), down, H)
    pouch_i = _add(_add(_add(S, u, L * 0.25), n, hw), down, H * 0.75)
    P += [pouch_o, _lerp2(pouch_o, pouch_i, 0.5), pouch_i, _add(_add(S, n, hw * 0.8), u, 4)]
    return P, cuff, n, u


# =============================================================== Haru — main
_VIEW_YAW = {"front": 0.0, "three_quarter_right": 0.92, "three_quarter_left": -0.92,
             "side_right": PI / 2, "side_left": -PI / 2, "back": PI}

ARM_PRESETS = ("rest", "chin_on_hands", "cup_hands", "hold_lantern", "raise_lantern", "wave",
               "reach_up", "run", "hands_to_mouth", "hips")


def draw_haru(ctx, x, y, scale=1.0, *, view="front", yaw=None, expr="neutral", mouth_open=0.0,
              vowel="x", blink=0.0, look=(0.0, 0.0), head_tilt=0.0, head_turn=0.0, head_pitch=0.0,
              body_lean=0.0, crouch=0.0, bounce=0.0, arms="rest", arm_phase=0.0, hand=None,
              run_phase=None, wind=0.0, wind_dir=None, t=0.0, lantern=False, lantern_lit=0.0,
              lantern_kira=0.0, lantern_scale=0.95, held=None, crop=None, rim=RIM_DEFAULT, light="left",
              hand_l=None, hand_r=None, breathe=1.0, eye_open=None, blush=None, brow_tilt=None,
              smile=None, alpha=1.0):
    """Draw Haru.  (x, y) = ground point between her feet; scale 1 -> ~520 px tall.

    view   : 'front' | 'three_quarter_left' | 'three_quarter_right' | 'side_left' |
             'side_right' | 'back'   (or pass `yaw` radians: 0 front, +pi/2 facing screen-right,
             -pi/2 facing screen-left, pi back).  Left-facing views are mirrored right-facing ones.
    expr   : neutral happy surprised determined sad gentle sleepy worried laugh
    mouth_open 0..1, vowel in 'aiueonx' (lip sync, e.g. from core.mouth()).
    blink 0..1, look (dx, dy) in -1..1, head_tilt (rad, + = clockwise on screen),
    head_turn (extra yaw for the head only, rad), head_pitch (-1..1, + = looking up).
    body_lean (rad, + = clockwise/forward for a right-facing run), crouch 0..1, bounce (squash +/stretch -).
    arms   : 'rest' 'chin_on_hands' 'cup_hands' 'hold_lantern' 'raise_lantern' 'wave' 'reach_up'
             'run' 'hands_to_mouth' 'hips';  arm_phase animates wave/run (cycles per unit).
    hand   : for one-handed poses (wave, hold_lantern, reach_up): 'screen_left' | 'screen_right' |
             'near' | 'far' | 'both' (reach_up only).  Default: the screen-right hand in front view,
             the near hand in side views.
    run_phase 0..1: full side-view run cycle (legs, arms if arms='run', hair/sleeves).  Works in
             any view (front view becomes an in-place jog).  None = standing.
    wind   : hair/sleeve sway amount (0 calm, 1 breezy, 2 strong); wind_dir +1 blows to screen-right.
    t      : seconds, drives idle breathing and hair sway.
    lantern: False | True (in hand for hold_lantern / raise_lantern) | 'back' (hanging on her back).
    lantern_lit 0..1, lantern_kira 0..1 (Kira glowing inside the lantern).
    held   : callback held(ctx, x, y, s) drawn in caller coordinates at the cupped hands
             (between palms and fingers for 'cup_hands'); s = scale.
    crop   : None | 'bust' (skips legs/feet; scene draws a sill/frame over the waist).
    rim    : rim-light colour (default cool moonlight; e.g. '#FFC27A' for lantern light), None = off.
    light  : 'left' | 'right' key-light side (moves cel shadows and catch-lights).
    hand_l / hand_r : override wrist target (l, f, v) in body space (her left / right hand),
             v relative to hip (hip is 230 above feet, head centre at v=-226).
    eye_open, blush, brow_tilt, smile: override the expression preset values.

    Returns anchors in caller coordinates:
      head, face (centre of the face), eyes [..], mouth, chin, top (crown), neck, chest, hip, feet,
      hand_l, hand_r, hands (midpoint), cup (top of cupped hands), lantern (hang point),
      lantern_body (paper centre), lantern_radius, facing (+1 right / -1 left / 0 front|back),
      scale.
    """
    M0 = ctx.get_matrix()
    M0inv = cairo.Matrix(*M0)
    M0inv.invert()

    def anchor(px, py):
        dx, dy = ctx.user_to_device(px, py)
        return M0inv.transform_point(dx, dy)

    # ------------------------------------------------ view / mirroring
    ya = _VIEW_YAW.get(view, 0.0) if yaw is None else float(yaw)
    ya = (ya + PI) % TAU - PI if abs(ya) > PI + 1e-6 else ya
    mirror = 1
    if ya < -1e-6 and ya > -PI + 1e-6:
        mirror = -1
        ya = -ya
    if abs(ya) >= PI - 1e-6:
        ya = PI
    ls = 1 if light == "left" else -1
    ls *= mirror
    look = (look[0] * mirror, look[1])
    wdir = (wind_dir if wind_dir is not None else (-1 if (0.2 < ya < 2.9) else 1) * mirror) * mirror
    head_tilt *= mirror
    body_lean *= mirror
    sy, cyw = math.sin(ya), math.cos(ya)
    back = cyw < -0.35
    P = _Proj(ya)
    running = run_phase is not None
    rp = (run_phase or 0.0) % 1.0

    # map one-handed 'hand' requests to anatomical side: +1 her left, -1 her right
    def which_hand(default="screen_right"):
        h = hand or default
        if h == "both":
            return 0
        if h in ("near", "far"):
            if abs(sy) < 0.2:
                h = "screen_right"
            else:
                near = -1  # her right is nearer when facing right (after mirroring)
                if back:
                    near = 1
                return near if h == "near" else -near
        # screen side -> anatomical (after mirroring, screen right is +x in local space)
        want = 1 if h == "screen_right" else -1
        want *= mirror
        xs = {s: P.p(s * 30, 0, 0)[0] for s in (-1, 1)}
        return 1 if (xs[1] - xs[-1]) * want > 0 else -1

    _LW[0] = line_weight(scale)
    if alpha < 1.0:
        ctx.push_group()
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale * mirror, scale)
    if bounce:
        sq = clamp(bounce, -0.5, 0.5)
        ctx.scale(1 + sq * 0.5, 1 - sq)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    # ------------------------------------------------ legs (angles in sagittal plane)
    legs = {}
    for side in (-1, 1):
        if running:
            ph = rp + (0.0 if side == -1 else 0.5)
            s2 = math.sin(TAU * ph)
            alpha_t = math.radians(38) * s2 + math.radians(6)
            kb = math.radians(22 + 88 * max(0.0, math.cos(TAU * (ph + 0.03))) ** 1.4)
            if s2 < 0:
                kb += math.radians(10) * -s2
        else:
            alpha_t = math.radians(2 + 84 * crouch) + side * 0.0
            kb = math.radians(4 + 110 * crouch)
        legs[side] = [alpha_t, kb]
    # crouch: widen knees a bit
    knee_out = 6 + 14 * crouch

    def leg_geom(side, hip_pt):
        a, kb = legs[side]
        lat = side * (12.5 + (3 if running else 0))
        K = (lat + side * knee_out * (1 - 0.6 * (1 if running else 0)), hip_pt[1] + L_THIGH * math.sin(a),
             hip_pt[2] + L_THIGH * math.cos(a))
        sa = a - kb
        A = (lat + side * (2 + 4 * crouch), K[1] + L_SHIN * math.sin(sa), K[2] + L_SHIN * math.cos(sa))
        return K, A, sa

    hip0 = (0.0, 0.0, 0.0)
    low = -1e9
    f_shift = 0.0
    tmp = {}
    for side in (-1, 1):
        K, A, sa = leg_geom(side, hip0)
        tmp[side] = (K, A)
        low = max(low, A[2] + FOOT_H)
        if not running:
            f_shift += A[1] * 0.5
    hip_h = low
    if running:
        hip_h += 9.0 * max(0.0, math.sin(TAU * (2 * rp - 0.18)))
        f_shift = 0.0
    hip_f = -f_shift if not running else 0.0
    breath = math.sin(t * TAU / 3.4) * breathe
    hip = (0.0, hip_f, -hip_h)

    # ------------------------------------------------ upper-body transform
    lean = body_lean + (0.14 if running else 0.0) + 0.1 * crouch * (1 if abs(sy) > 0.3 else 0)
    hip2 = P.p(*hip)

    def to_root(pt2):  # upper-body local -> root (feet) space
        r = _rot(pt2, lean)
        return (hip2[0] + r[0], hip2[1] + r[1])

    # ------------------------------------------------ arms
    arm_info = {}
    hl = which_hand("screen_right")
    hands_up = {}

    def target(side):
        """Returns (wrist (l,f,v), pole, hand_kind, layer) for her side (+1 left, -1 right)."""
        s = side
        k = arms
        ph = arm_phase
        rest = ((s * 40, 14, -22), (s * 0.35, -1.0, 0.0), "open", "front")
        if k == "rest":
            return rest
        if k == "hips":
            return ((s * 44, -4, -66), (s * 1, -0.4, -0.3), "fist", "front")
        if k == "chin_on_hands":
            return ((s * 12, 48, -168), (s * 0.6, 0.3, 1), "cheek", "over")
        if k == "cup_hands":
            return ((s * 9, 34, -98), (s * 1, -0.3, 0.45), "cup", "front")
        if k == "hold_lantern":
            if s == hl:
                return ((s * 40, 26, -52), (s * 1, -0.6, 0.3), "fist", "front")
            return rest
        if k == "raise_lantern":
            return ((s * 38, 18, -262), (s * 1, 0.1, 0.3), "up", "front")
        if k == "wave":
            if s == hl:
                wv = math.sin(ph * TAU)
                return ((s * (66 + 16 * wv), 16, -250 + 5 * abs(wv)), (s * 1, -0.2, 0.6), "wave", "front")
            return rest
        if k == "reach_up":
            if hl == 0 or s == hl:
                return ((s * 24, 24, -300), (s * 1, 0, 0.2), "up", "front")
            return ((s * 46, 12, -70), (s * 1, -0.4, 0.2), "open", "front")
        if k == "hands_to_mouth":
            return ((s * 11, 58, -186), (s * 0.9, 0.2, 1), "cheek", "over")
        if k == "run":
            return None
        return rest

    run_arm = arms == "run"
    for side in (-1, 1):
        S = (side * SHOULDER[0], 0.0, SHOULDER[1] + breath * 0.8)
        if side == -1 and hand_r is not None:
            tg = (tuple(hand_r), (side, -0.5, 0.3), "open", "front")
        elif side == 1 and hand_l is not None:
            tg = (tuple(hand_l), (side, -0.5, 0.3), "open", "front")
        else:
            tg = target(side)
        if tg is None:  # FK run swing (opposite to same-side leg)
            ph = (run_phase if running else arm_phase) or 0.0
            ph = ph + (0.5 if side == -1 else 0.0)
            sw = math.sin(TAU * ph)
            th = math.radians(55) * sw - math.radians(8)
            bend = math.radians(95 + 10 * sw)
            E = [S[0] + side * 4, S[1] + L_UPPER * math.sin(th), S[2] + L_UPPER * math.cos(th)]
            W = [E[0] - side * 5, E[1] + L_FORE * math.sin(th + bend), E[2] + L_FORE * math.cos(th + bend)]
            kind, layer = "fist", "front"
        else:
            Wt, pole, kind, layer = tg
            E, W = _ik(S, Wt, L_UPPER, L_FORE, pole)
        arm_info[side] = dict(S=S, E=E, W=W, kind=kind, layer=layer)

    # layer resolution by depth
    for side, a in arm_info.items():
        zs = P.z(a["S"][0], a["S"][1])
        zw = P.z(a["W"][0], a["W"][1])
        if a["layer"] != "over":
            if back:
                a["layer"] = "back" if zw < -4 else "front"
            elif zs < -8:
                a["layer"] = "back" if not (arms in ("cup_hands",) and zw > 0) else "back"
            else:
                a["layer"] = "front"
        a["z"] = zs

    # ------------------------------------------------ head state
    ya_h = clamp(ya + head_turn * mirror, 0.0, PI) if not back else PI
    over = dict(eye_open=eye_open, blush=blush, brow_tilt=brow_tilt, smile=smile)
    head = _Head(ya_h, head_pitch, expr, t, wind * wdir + (1.3 if running else 0.0) * wdir * 1.0,
                 blink, look, mouth_open, vowel, over, rim, ls)
    if running:
        head.wind = (wind + 1.4) * wdir
    else:
        head.wind = wind * wdir
    head_pos3 = (0.0, 4.0 + 10 * crouch, HEAD_V + breath * 1.1)
    if arms == "chin_on_hands":
        head_pos3 = (0.0, 12.0, HEAD_V + 12)

    anchors = {}

    # ------------------------------------------------ drawing helpers
    def draw_arm(side):
        a = arm_info[side]
        S2 = P.p3(a["S"])
        E2 = P.p3(a["E"])
        W2 = P.p3(a["W"])
        if _len(_sub(E2, S2)) < 8:
            E2 = (E2[0] + side * 6, E2[1] + 4)
        far = a["z"] < -8
        sk = SKIN if not far else mix(SKIN, SKIN_SH, 0.35)
        d = _norm(_sub(W2, E2)) if _len(_sub(W2, E2)) > 1 else (0, 1)
        # forearm
        _cel(ctx, lambda c: _limb(c, [E2, W2], [8.2, 6.4]), sk, SKIN_SH, None, SKIN_LINE, _LW[0] * 0.85,
             soff=(2.0 * ls, -1.5), shade_a=0.7)
        kind = a["kind"]
        hs = 11.5
        th_side = side * (1 if not back else -1) * (1 if cyw >= 0 else -1)
        if kind == "cup":
            pass  # drawn jointly later
        elif kind == "cheek":
            _hand(ctx, W2, _norm((d[0] * 0.3, -1.0)), hs, "open", -side, rim, ls)
        elif kind == "up":
            _hand(ctx, W2, d, hs, "open", side, rim, ls)
        elif kind == "wave":
            _hand(ctx, W2, _norm((d[0] + 0.3 * math.sin(arm_phase * TAU), d[1])), hs * 1.05, "open", side, rim, ls)
        else:
            _hand(ctx, W2, d, hs, kind, th_side, rim, ls)
        # sleeve
        fl = -wdir * 0.0 + (0.35 * wdir if running else 0.0) + 0.12 * wind * wdir * math.sin(t * 2.2 + side)
        pts, cuff, n, u = _sleeve_path(S2, E2, W2, 44.0, fl)
        base = YUKATA if not far else mix(YUKATA, YUKATA_SH, 0.45)

        def sleeve_extra(c):
            mot = []
            for i, (tt, nn) in enumerate(((0.35, 0.2), (0.8, -0.3), (0.65, 1.4), (0.2, 1.6))):
                p = _add(_add(S2, u, _len(_sub(E2, S2)) * tt), n, 14 * nn)
                mot.append((p[0], p[1] + 6, 6.5, 1.0))
            _pattern_arcs(c, mot, PATTERN, 0.45 if not far else 0.3)

        _cel(ctx, lambda c: _spline(c, pts, True, 0.9), base, YUKATA_SH, rim if not far else None, YUKATA_LINE,
             _LW[0], soff=(-5 * ls, -5), roff=(-2.6 * ls, 0.5), rim_a=0.7, extra=sleeve_extra)
        # cuff opening
        c1 = _add(cuff, n, -17)
        c2 = _add(cuff, n, 16)
        _sc(ctx, YUKATA_LINE, 0.9)
        _taper(ctx, [c1, _add(_lerp2(c1, c2, 0.5), u, 2.5), c2], 1.4, 1.4, 2.6)
        anchors["hand_l" if side == 1 else "hand_r"] = anchor(*W2)
        hands_up[side] = W2

    def draw_cup_hands():
        W = [P.p3(arm_info[s]["W"]) for s in (-1, 1)]
        cx = (W[0][0] + W[1][0]) / 2
        cy = (W[0][1] + W[1][1]) / 2 + 6
        return cx, cy

    def cup_back(cx, cy):
        # palms (back layer): a shallow bowl
        pts = [(cx - 25, cy - 8), (cx - 15, cy + 10), (cx, cy + 14), (cx + 15, cy + 10), (cx + 25, cy - 8),
               (cx + 12, cy - 2), (cx - 12, cy - 2)]
        _cel(ctx, lambda c: _spline(c, pts), SKIN, SKIN_SH, None, SKIN_LINE, _LW[0] * 0.85,
             soff=(0, -4), shade_a=0.7)

    def cup_front(cx, cy):
        for s in (-1, 1):
            pts = [(cx + s * 27, cy - 11), (cx + s * 22, cy + 4), (cx + s * 8, cy + 13), (cx + s * 1, cy + 9),
                   (cx + s * 6, cy + 1), (cx + s * 16, cy - 7)]
            _cel(ctx, lambda c: _spline(c, pts), SKIN, SKIN_SH, None, SKIN_LINE, _LW[0] * 0.85,
                 soff=(-1.5 * s, -3), shade_a=0.6)
            _sc(ctx, SKIN_LINE, 0.5)
            ctx.set_line_width(0.9)
            for k in (0.35, 0.6):
                p0 = (lerp(cx + s * 26, cx + s * 6, k), lerp(cy - 9, cy + 9, k))
                ctx.move_to(*p0)
                ctx.line_to(p0[0] + s * 4, p0[1] - 3)
                ctx.stroke()
            # thumbs over the rim
            _cel(ctx, lambda c: _spline(c, [(cx + s * 22, cy - 10), (cx + s * 15, cy - 13), (cx + s * 11, cy - 10),
                                             (cx + s * 16, cy - 6)]), SKIN, None, None, SKIN_LINE, _LW[0] * 0.8)

    def call_held(px, py):
        if held is None:
            return
        gx, gy = anchor(px, py)
        ctx.save()
        ctx.set_matrix(M0)
        held(ctx, gx, gy, scale)
        ctx.restore()

    def lantern_for_hand():
        """Stick + lantern hang point for hold_lantern (in upper-body 2D coords)."""
        a = arm_info[hl]
        W2 = P.p3(a["W"])
        E2 = P.p3(a["E"])
        sway_k = math.sin(t * 2.0) * 0.08 + (0.25 * math.sin(TAU * rp) if running else 0.0)
        d3 = (hl * 0.45, 0.75, -0.5)
        tip3 = (a["W"][0] + d3[0] * 70, a["W"][1] + d3[1] * 70, a["W"][2] + d3[2] * 70)
        tip = P.p3(tip3)
        return W2, tip, sway_k

    def draw_stick(W2, tip):
        d = _norm(_sub(tip, W2))
        n = (-d[1], d[0])
        base = _add(W2, d, -10)
        pts = [_add(base, n, 2.4), _add(tip, n, 1.6), _add(tip, n, -1.6), _add(base, n, -2.4)]
        _cel(ctx, lambda c: (c.move_to(*pts[0]), [c.line_to(*p) for p in pts[1:]], c.close_path()), BAMBOO,
             BAMBOO_SH, None, BAMBOO_LINE, _LW[0] * 0.7, soff=(0, -1.5))
        for k in (0.35, 0.7):
            q = _lerp2(base, tip, k)
            _sc(ctx, BAMBOO_LINE, 0.8)
            ctx.set_line_width(1.1)
            ctx.move_to(*_add(q, n, 2.2))
            ctx.line_to(*_add(q, n, -2.2))
            ctx.stroke()

    def draw_lantern_at(hang_local, sway, ls_scale):
        gx, gy = anchor(*hang_local)
        ctx.save()
        ctx.set_matrix(M0)
        res = draw_lantern(ctx, gx, gy, scale * ls_scale, lit=lantern_lit, sway=sway, t=t, kira=lantern_kira,
                           _mirror=1)
        ctx.restore()
        anchors["lantern"] = (gx, gy)
        anchors["lantern_body"] = res["body_center"]
        anchors["lantern_radius"] = res["radius"]

    # ------------------------------------------------ DRAW
    hipl = (-1, 1)
    near_leg = -1 if not back else 1  # her right is nearer when facing right (ya>0)
    order_legs = [s for s in (1, -1)] if ya > 0.05 else [-1, 1]
    if back:
        order_legs = [-1, 1]

    # lantern on back (behind everything unless back view)
    def lantern_back_hang():
        v = SHOULDER[1] + 10
        f = -30.0
        p = P.p(-8.0, f, v)
        return to_root(p)

    if lantern == "back" and not back:
        sw = math.sin(t * 2.3) * 0.1 + (0.35 * math.sin(TAU * rp + 1.0) if running else 0.0)
        draw_lantern_at(lantern_back_hang(), sw * 0.6 - (0.25 if running else 0.0), lantern_scale * 0.9)

    # back arms
    ctx.save()
    ctx.translate(*hip2)
    ctx.rotate(lean)
    for side in (-1, 1):
        if arm_info[side]["layer"] == "back":
            draw_arm(side)
    ctx.restore()

    # obi bow behind?
    def draw_bow():
        bw = 33.0
        (bx, by) = P.p(0, -22.0 - 1.0, -66.0)
        zb = -cyw
        fw = max(abs(cyw), 0.28)
        ctx.save()
        ctx.translate(bx, by)
        ctx.scale(fw, 1)
        fl = math.sin(t * 2.4) * 1.5 + (4 * math.sin(TAU * rp * 2) if running else 0)
        for s in (-1, 1):
            lobe = [(0, -4), (s * 20, -24 + fl * 0.3), (s * 46, -22), (s * 52, -4), (s * 42, 12),
                    (s * 20, 8), (0, 4)]
            _cel(ctx, lambda c: _spline(c, lobe), OBI, OBI_SH, OBI_HI, OBI_LINE, _LW[0],
                 soff=(-4 * ls, -5), roff=(2, 2), rim_a=0.5)
            _sc(ctx, OBI_SH, 0.9)
            _taper(ctx, [(s * 8, -1), (s * 26, -8), (s * 42, -12)], 1.6, 0.4)
            _taper(ctx, [(s * 8, 2), (s * 28, 4), (s * 40, 4)], 1.4, 0.3)
        for s in (-1, 1):
            tail = [(s * 3, 0), (s * 14, 20), (s * 20 + fl, 62), (s * 10 + fl, 66, True), (s * 2 + fl * 0.5, 30)]
            _cel(ctx, lambda c: _spline(c, tail), OBI, OBI_SH, None, OBI_LINE, _LW[0], soff=(-3 * ls, -3))
        knot = [(-9, -9), (9, -9), (11, 8), (-11, 8)]
        _cel(ctx, lambda c: _spline(c, knot), OBI, OBI_SH, None, OBI_LINE, _LW[0], soff=(-2 * ls, -3))
        ctx.restore()

    ctx.save()
    ctx.translate(*hip2)
    ctx.rotate(lean)
    if -cyw < 0.05:
        draw_bow()
    ctx.restore()

    # legs & feet
    if crop != "bust":
        for side in order_legs:
            K, A, sa = leg_geom(side, hip)
            Hj = (side * 12.5, hip[1], hip[2])
            K3 = (K[0], K[1] + hip[1], K[2] + hip[2]) if False else K
            # leg_geom already used hip (l,f,v) relative -> recompute absolute
            K, A, sa = leg_geom(side, hip)
            h2 = P.p(side * 11, hip[1], hip[2] + 20)
            k2 = P.p(*K)
            a2 = P.p(*A)
            farl = P.z(side * 10, 0) < -3
            sk = SKIN if not farl else mix(SKIN, SKIN_SH, 0.4)
            _cel(ctx, lambda c: _limb(c, [h2, k2, a2], [13.5, 9.5, 6.6]), sk, SKIN_SH, None, SKIN_LINE,
                 _LW[0] * 0.9, soff=(-3 * ls, 0), shade_a=0.8)
            # foot
            if running:
                fa = max(0.0, -sa - 0.15) * 1.1 - (0.25 if legs[side][0] > 0.35 else 0.0)
            else:
                fa = 0.0
            yaw_out = side * 0.28
            fdir = (math.sin(yaw_out), math.cos(yaw_out))

            def fp(fwd, down, lat):
                # foot-local: fwd along foot, down (+), lat across; rotate by pitch fa about ankle
                ff = fwd * math.cos(fa) - down * math.sin(fa)
                dd = fwd * math.sin(fa) + down * math.cos(fa)
                l = A[0] + fdir[0] * ff + fdir[1] * lat
                f = A[1] + fdir[1] * ff - fdir[0] * lat
                return P.p(l, f, A[2] + dd)

            sole = [fp(26 * math.cos(q), FOOT_H - 1, 9.5 * math.sin(q)) for q in [i * TAU / 16 for i in range(16)]]
            sole = [(p[0] + 4 * 0, p[1]) for p in sole]
            sole = [fp(6 + 22 * math.cos(q), FOOT_H - 0.5, 9.5 * math.sin(q)) for q in [i * TAU / 16 for i in range(16)]]
            sole_lo = [fp(6 + 22 * math.cos(q), FOOT_H + 3.5, 9.5 * math.sin(q)) for q in [i * TAU / 16 for i in range(16)]]
            hull = _hull(sole + sole_lo)
            _cel(ctx, lambda c: _spline(c, hull, True, 0.8), STRAW if not farl else STRAW_SH, STRAW_SH, None,
                 STRAW_LINE, _LW[0] * 0.8, soff=(0, -2.5))
            foot3 = [fp(0, -3, 0)] + [fp(-6, 6, s2 * 6) for s2 in (-1, 1)] + [fp(0, FOOT_H - 1, s2 * 7) for s2 in (-1, 1)] \
                + [fp(20, FOOT_H - 1, s2 * 8) for s2 in (-1, 1)] + [fp(24, 6, s2 * 6) for s2 in (-1, 1)] + \
                [fp(26, FOOT_H - 2, s2 * 4) for s2 in (-1, 1)] + [fp(10, 1, 0)]
            fh = _hull(foot3)
            _cel(ctx, lambda c: _spline(c, fh, True, 0.9), sk, SKIN_SH, None, SKIN_LINE, _LW[0] * 0.85,
                 soff=(-2 * ls, -2), shade_a=0.8)
            # strap (hanao)
            _sc(ctx, STRAP)
            ctx.set_line_width(2.6)
            p_toe = fp(19, 3.0, 0)
            ctx.move_to(*fp(6, 5.5, -8.5))
            ctx.curve_to(*fp(12, 1.0, -5), *fp(16, 1.5, -1), *p_toe)
            ctx.curve_to(*fp(16, 1.5, 1), *fp(12, 1.0, 5), *fp(6, 5.5, 8.5))
            ctx.stroke()
            ctx.arc(*p_toe, 2.0, 0, TAU)
            ctx.fill()

    # skirt + torso in upper-body frame
    ctx.save()
    ctx.translate(*hip2)
    ctx.rotate(lean)
    rel = lambda l, f, v: P.p(l, f, v)  # already relative to hip
    # knees relative for hem deformation
    knees = []
    for side in (-1, 1):
        K, A, sa = leg_geom(side, (0.0, 0.0, 0.0))
        knees.append(K)
    hem_v = _HEM_V + 2 * math.sin(t * 1.7)
    kmin = min(k[2] for k in knees)
    if crop != "bust":
        hem_v = min(hem_v, kmin + 18)
    ring_pts = []
    N = 28
    fl_amp = (3.5 if running else 0) + 1.5 * wind
    for i in range(N):
        phi = i * TAU / N
        w, d = _HEM_W * (1 + 0.25 * crouch), _HEM_D * (1 + 0.3 * crouch)
        l, f = w * math.sin(phi), d * math.cos(phi)
        v = hem_v + fl_amp * math.sin(3 * phi + t * 7 + rp * TAU * 2)
        # push the hem outward toward knees (running / crouching)
        for K in knees:
            kr = math.hypot(K[0], K[1])
            if kr < 1:
                continue
            kph = math.atan2(K[0], K[1])
            dd = math.atan2(math.sin(phi - kph), math.cos(phi - kph))
            er = math.hypot(l, f)
            need = kr + 16 - er
            if need > 0 and abs(dd) < 1.2:
                wgt = math.cos(dd / 1.2 * PI / 2) ** 2
                l += math.sin(phi) * need * wgt
                f += math.cos(phi) * need * wgt
                v = min(v, lerp(v, K[2] + 16, wgt))
        l += wdir * (4 if running else 0) + wdir * wind * 2.5
        ring_pts.append(P.p(l, f, v))
    top_ring = [P.p(_SKIRT_TOP[1] * math.sin(q), _SKIRT_TOP[2] * math.cos(q), _SKIRT_TOP[0])
                for q in [i * TAU / 20 for i in range(20)]]
    skirt_h = _resample_closed(_hull(ring_pts + top_ring), 14)

    def skirt_path(c):
        _spline(c, skirt_h, True, 0.8)

    def skirt_extra(c):
        # overlap line of the front panel (okumi) & motifs
        pts = []
        for vv in [-48 + (hem_v + 48) * k / 6 for k in range(7)]:
            w, d, cf = _ring_at(vv)
            phi = -0.32
            z = P.z(w * math.sin(phi), d * math.cos(phi))
            pts.append((P.p(w * math.sin(phi), d * math.cos(phi) + cf, vv), z))
        if all(z > 0 for _, z in pts):
            _sc(c, YUKATA_LINE, 0.9)
            _taper(c, [p for p, _ in pts], 1.8, 2.2)
            # shade the under-panel slightly
        mot = []
        for (phi, vv) in _MOTIFS:
            if vv < -40:
                continue
            w, d, cf = _ring_at(min(vv, hem_v - 8))
            z = P.z(w * math.sin(phi), d * math.cos(phi))
            if z < 3:
                continue
            p = P.p(w * math.sin(phi), d * math.cos(phi) + cf, min(vv, hem_v - 8))
            mot.append((p[0], p[1], 8, clamp(z / d, 0.25, 1)))
        _pattern_arcs(c, mot, PATTERN, 0.5)
        # soft darker band just above hem (inside fold)
        _sc(c, YUKATA_SH, 0.5)
        ctx_pts = []

    _cel(ctx, skirt_path, YUKATA, YUKATA_SH, rim, YUKATA_LINE, _LW[0], soff=(-8 * ls, -3),
         roff=(-2.8 * ls, 0.3), rim_a=0.7, extra=skirt_extra)
    # hem inner edge line
    hem_front = sorted([p for i, p in enumerate(ring_pts)
                        if P.z(_HEM_W * math.sin(i * TAU / N), _HEM_D * math.cos(i * TAU / N)) > 0],
                       key=lambda p: p[0])

    # torso
    tpts = []
    for (v, w, d, cf) in _TORSO:
        vv = v + (breath * 0.8 if v < -120 else 0)
        for i in range(20):
            q = i * TAU / 20
            tpts.append(P.p(w * math.sin(q), d * math.cos(q) + cf, vv))
    torso_h = _resample_closed(_hull(tpts), 12)

    def torso_path(c):
        _spline(c, torso_h, True, 0.8)

    def surf(phi, v):
        w, d, cf = _ring_at(v)
        l, f = w * math.sin(phi), d * math.cos(phi) + cf
        return P.p(l, f, v), P.z(l, f)

    def torso_extra(c):
        # motifs on the chest
        mot = []
        for (phi, vv) in _MOTIFS:
            if vv >= -90 or vv < -140:
                continue
            p, z = surf(phi, vv)
            w, d, cf = _ring_at(vv)
            if z < 3:
                continue
            mot.append((p[0], p[1], 7.5, clamp(z / d, 0.25, 1)))
        _pattern_arcs(c, mot, PATTERN, 0.5)
        if not back:
            # --- V neck (skin) and collar bands
            over = [(0.46, -165), (0.28, -151), (0.08, -134), (-0.1, -112), (-0.22, -90)]
            under = [(-0.46, -165), (-0.28, -151), (-0.1, -137), (0.02, -128)]
            op = [surf(*q)[0] for q in over]
            up = [surf(*q)[0] for q in under]
            neck_top = [surf(ph, -170)[0] for ph in (0.6, 0.3, 0.0, -0.3, -0.6)]
            zs = [surf(*q)[1] for q in over + under]
            c.new_path()
            _spline(c, [up[0]] + up[1:] + [op[2]] + op[1::-1] + neck_top[1:4][::-1], True, 0.7)
            c.new_path()
            vpts = up + [op[3], op[2], op[1], op[0]]
            c.move_to(*neck_top[0])
            for p in neck_top[1:]:
                c.line_to(*p)
            c.line_to(*up[0])
            for p in up[1:]:
                c.line_to(*p)
            for p in reversed(op[:3]):
                c.line_to(*p)
            c.close_path()
            _sc(c, SKIN)
            c.fill_preserve()
            _sc(c, SKIN_SH, 0.7)
            c.fill()
            # collar bands
            for pts, wdt in ((up, 7.5), (op, 8.0)):
                vis = [p for p, q in zip(pts, (over if pts is op else under)) if surf(*q)[1] > -6]
                if len(vis) < 2:
                    continue
                _sc(c, COLLAR)
                _taper(c, vis, wdt, wdt * 0.8)
                _sc(c, "#9FB4DE", 0.9)
                _taper(c, [(p[0] + 1.5 * ls, p[1] + 1) for p in vis], wdt * 0.35, wdt * 0.25)
                _sc(c, YUKATA_LINE)
                _taper(c, [(p[0] - 3.2, p[1]) if pts is op else (p[0] + 3.2, p[1]) for p in vis], 1.2, 1.2)
        # --- obi band
        top_v, bot_v = -86, -46
        tl = [surf(q, top_v) for q in [i * TAU / 36 for i in range(36)]]
        bl = [surf(q, bot_v) for q in [i * TAU / 36 for i in range(36)]]
        tvis = sorted([p for p, z in tl if z > -1], key=lambda p: p[0])
        bvis = sorted([p for p, z in bl if z > -1], key=lambda p: p[0])
        if tvis and bvis:
            poly = [(tvis[0][0] - 30, tvis[0][1])] + tvis + [(tvis[-1][0] + 30, tvis[-1][1]),
                                                             (bvis[-1][0] + 30, bvis[-1][1])] + bvis[::-1] + \
                   [(bvis[0][0] - 30, bvis[0][1])]

            def obi_path(cc):
                cc.move_to(*poly[0])
                for p in poly[1:]:
                    cc.line_to(*p)
                cc.close_path()

            def obi_extra(cc):
                # soft folds + cord
                mid = sorted([p for p, z in [surf(q, -65) for q in [i * TAU / 36 for i in range(36)]] if z > -1],
                             key=lambda p: p[0])
                if len(mid) > 2:
                    _sc(cc, CORD)
                    _taper(cc, mid, 3.2, 3.2)
                    _sc(cc, "#FFB0A0", 0.7)
                    _taper(cc, [(p[0], p[1] - 1) for p in mid], 0.9, 0.9)
                f1 = sorted([p for p, z in [surf(q, -78) for q in [i * TAU / 36 for i in range(36)]] if z > 2],
                            key=lambda p: p[0])
                if len(f1) > 2:
                    _sc(cc, OBI_SH, 0.8)
                    _taper(cc, f1[1:-1], 0.3, 0.3, 1.5)
                _sc(cc, OBI_LINE)
                _taper(cc, tvis, 1.4, 1.4)
                _taper(cc, bvis, 1.4, 1.4)

            _cel(c, obi_path, OBI, OBI_SH, OBI_HI, None, None, soff=(-7 * ls, -2), roff=(-2.5 * ls, 0),
                 rim_a=0.6, extra=obi_extra)

    # neck (under the torso / collar)
    neck_c = P.p(0, 2, -176)
    nb = P.p(0, 2, -158)

    def neck_path(c):
        _limb(c, [neck_c, nb], [9.5, 10.5])

    def neck_extra(c):
        hx, hy = P.p(*head_pos3)
        c.save()
        c.translate(hx, hy + 10)
        c.rotate(head_tilt)
        c.scale(1.0, 0.8)
        c.arc(0, 0, R_HEAD * 0.95, 0, TAU)
        c.restore()
        _sc(c, SKIN_SH)
        c.fill()

    _cel(ctx, neck_path, SKIN, None, None, SKIN_LINE, _LW[0] * 0.85, extra=neck_extra)
    _cel(ctx, torso_path, YUKATA, YUKATA_SH, rim, YUKATA_LINE, _LW[0], soff=(-7 * ls, -4),
         roff=(-2.8 * ls, 0.4), rim_a=0.7, extra=torso_extra)
    if -cyw >= 0.05:
        draw_bow()
    chest = P.p(0, 22, -112)
    anchors["chest"] = anchor(*chest)
    anchors["hip"] = anchor(0, 0)

    # front arms (+ lanterns / cup)
    cup_c = None
    if arms == "cup_hands":
        cup_c = draw_cup_hands()
    for side in (-1, 1):
        if arm_info[side]["layer"] == "front":
            draw_arm(side)
    if cup_c is not None:
        cup_back(*cup_c)
        call_held(cup_c[0], cup_c[1] - 4)
        cup_front(*cup_c)
        anchors["cup"] = anchor(cup_c[0], cup_c[1] - 4)
    if arms == "hold_lantern" and lantern and lantern != "back":
        W2, tip, swk = lantern_for_hand()
        draw_stick(W2, tip)
        draw_lantern_at(tip, swk - 0.05 * wdir * wind, lantern_scale)

    # head
    hx, hy = P.p(*head_pos3)
    if back:
        hy += 5 * head_pitch
    ctx.save()
    ctx.translate(hx, hy)
    ctx.rotate(head_tilt)
    head.draw(ctx)
    anchors["head"] = anchor(0, 0)
    anchors["top"] = anchor(0, -1.25 * R_HEAD)
    anchors["chin"] = anchor(*head.face_pts()[8][:2]) if not back else anchor(0, R_HEAD)
    anchors["face"] = anchor(*head.sp(0, 0.1 * R_HEAD)[0])
    anchors["eyes"] = [anchor(*e) for e in getattr(head, "eye_pos", [])]
    mx, my = getattr(head, "mouth_xy", head.sp(0, 0.66 * R_HEAD)[0])
    anchors["mouth"] = anchor(mx, my)
    ctx.restore()
    anchors["neck"] = anchor(*P.p(0, 2, -170))

    # arms over the head (hands at face)
    for side in (-1, 1):
        if arm_info[side]["layer"] == "over":
            draw_arm(side)

    if arms == "raise_lantern" and lantern and lantern != "back":
        W = [P.p3(arm_info[s]["W"]) for s in (-1, 1)]
        cx = (W[0][0] + W[1][0]) / 2
        cy = (W[0][1] + W[1][1]) / 2
        g = _lantern_geom(1.0)
        hang = (cx, cy - (g["body_cy"]) * lantern_scale - 16)
        draw_lantern_at(hang, math.sin(t * 1.7) * 0.03, lantern_scale)
        # fingers over the lantern
        for s, Wp in zip((-1, 1), W):
            _hand(ctx, Wp, (0, -1), 11.5, "open", s, rim, ls)
    if lantern == "back" and back:
        draw_lantern_at(lantern_back_hang(), math.sin(t * 2.3) * 0.1, lantern_scale * 0.9)
    ctx.restore()  # upper-body

    # hands anchors
    hl_, hr_ = anchors.get("hand_l"), anchors.get("hand_r")
    if hl_ and hr_:
        anchors["hands"] = ((hl_[0] + hr_[0]) / 2, (hl_[1] + hr_[1]) / 2)
    anchors["feet"] = anchor(0, 0)
    anchors["facing"] = 0 if (ya < 0.2 or back) else mirror
    anchors["scale"] = scale
    ctx.restore()
    if alpha < 1.0:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(alpha)
    return anchors


# =============================================================== Kira
def _kira_radius(theta, R, r_in, pts_ang, pts_len, p=1.7, width=None):
    """Soft 5-lobed star radius at angle theta (0 = up, clockwise)."""
    r = r_in
    for a0, L in zip(pts_ang, pts_len):
        d = math.atan2(math.sin(theta - a0), math.cos(theta - a0))
        W = width or math.radians(37)
        if abs(d) < W:
            r += (L - r_in) * math.cos(d / W * PI / 2) ** p
    return r


def _kira_outline(R, arm_l=0.0, arm_r=0.0, arm_l_len=1.0, arm_r_len=1.0, n=150):
    angs = [0.0, math.radians(72) - arm_r, math.radians(144), math.radians(216), math.radians(288) + arm_l]
    lens = [R, R * 0.93 * arm_r_len, R * 0.9, R * 0.9, R * 0.93 * arm_l_len]
    r_in = R * 0.56
    pts = []
    for i in range(n):
        th = i * TAU / n
        r = _kira_radius(th, R, r_in, angs, lens)
        pts.append((r * math.sin(th), -r * math.cos(th)))
    return pts


def _kira_colors(glow):
    """Body colours for a given life level: pale/desaturated when dim."""
    g = clamp(glow)
    pale = ("#E9E6DA", "#D9D5C6", "#BDB7A6")
    k = smoothstep(0.0, 0.85, g)
    core = mix(pale[0], KIRA_CORE, k)
    mid = mix(pale[1], KIRA_MID, k)
    rimc = mix(pale[2], KIRA_RIM, k)
    line = mix("#9C9282", KIRA_LINE, k)
    return core, mid, rimc, line


def draw_kira_glow(ctx, x, y, scale=1.0, glow=1.0, t=0.0):
    """Soft additive-looking halo around Kira (call BEFORE draw_kira).  Cheap: pure cairo."""
    g = clamp(glow)
    if g <= 0.01:
        return
    flick = 1.0 + 0.04 * math.sin(t * 9.0) + 0.03 * noise1(t * 6.0, 5)
    R = 60 * scale * (1.1 + 1.6 * g) * flick
    ctx.save()
    gr = cairo.RadialGradient(x, y, 0, x, y, R)
    c1, c2 = _c("#FFE7A0"), _c("#FF9A3D")
    gr.add_color_stop_rgba(0.0, *c1, 0.55 * g)
    gr.add_color_stop_rgba(0.25, *c1, 0.32 * g)
    gr.add_color_stop_rgba(0.55, *c2, 0.12 * g)
    gr.add_color_stop_rgba(1.0, *c2, 0.0)
    ctx.set_source(gr)
    ctx.arc(x, y, R, 0, TAU)
    ctx.fill()
    ctx.restore()


def kira_glow_layer(x, y, scale=1.0, glow=1.0, t=0.0, **pose):
    """RGBA float layer of Kira alone (1920x1080) to feed core.add_glow for real bloom:
        lay = kira_glow_layer(x, y, s, g, t)
        add_glow(frame, lay, radius=28*s, strength=1.2*g)"""
    from .core import draw_layer
    return draw_layer(lambda ctx: draw_kira(ctx, x, y, scale, glow=glow, t=t, **pose))


def draw_kira(ctx, x, y, scale=1.0, *, expr="neutral", mouth_open=0.0, vowel="x", blink=0.0,
              look=(0.0, 0.0), rot=0.0, squash=0.0, glow=1.0, t=0.0, pose="none", pose_phase=None,
              blush=None, alpha=1.0):
    """Draw Kira, the fallen star.  (x, y) = star centre; scale 1 -> ~120 px across.

    expr : 'neutral' 'happy' 'sad' 'hurt' (>_<) 'surprised' 'sleepy'
    mouth_open 0..1 + vowel for lip sync; blink 0..1; look (dx, dy) -1..1.
    rot (rad), squash -1..1 (+ squash wide, - stretch tall).
    glow 0..1 = life: ~0.1 pale/desaturated/translucent, 1 blazing (also drives flicker).
    pose : 'none' | 'rub_head' (left point rubs its head) | 'wave' (right point waves) |
           'cheer' (both up) | 'droop' (points sag -- tired).  pose_phase animates (defaults to t).
    Returns {'center', 'glow_radius', 'eyes', 'mouth', 'top'} in caller coordinates.
    """
    M0 = ctx.get_matrix()
    M0inv = cairo.Matrix(*M0)
    M0inv.invert()

    def anchor(px, py):
        dx, dy = ctx.user_to_device(px, py)
        return M0inv.transform_point(dx, dy)

    g = clamp(glow)
    R = 60.0
    ph = t if pose_phase is None else pose_phase
    arm_l = arm_r = 0.0
    ll = lr = 1.0
    if pose == "rub_head":
        arm_l = math.radians(52) + math.radians(9) * math.sin(ph * TAU * 2.2)
        ll = 0.86
    elif pose == "wave":
        arm_r = math.radians(40) + math.radians(22) * math.sin(ph * TAU * 1.6)
        lr = 1.04
    elif pose == "cheer":
        arm_l = arm_r = math.radians(40) + math.radians(6) * math.sin(ph * TAU * 2)
    elif pose == "droop":
        arm_l = arm_r = -math.radians(22)
    alpha_eff = alpha * (0.55 + 0.45 * smoothstep(0.0, 0.6, g))
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale, scale)
    ctx.rotate(rot)
    sq = clamp(squash, -1, 1)
    ctx.scale(1 + 0.22 * sq, 1 - 0.22 * sq)
    lw = line_weight(scale, 2.2)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.push_group()
    pts = _kira_outline(R, arm_l, arm_r, ll, lr)
    core, mid, rimc, line = _kira_colors(g)

    def body(c):
        c.move_to(*pts[0])
        for p in pts[1:]:
            c.line_to(*p)
        c.close_path()

    gr = cairo.RadialGradient(-6, -8, 2, 0, 0, R * 1.02)
    gr.add_color_stop_rgb(0.0, *core)
    gr.add_color_stop_rgb(0.42, *mid)
    gr.add_color_stop_rgb(1.0, *rimc)

    def extra(c):
        # inner soft shadow bottom-right & glossy highlight top-left
        c.save()
        c.translate(-5, -6)
        body(c)
        c.restore()
        c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        c.rectangle(-500, -500, 1000, 1000)
        _sc(c, mix(rimc, "#E08A2A", 0.35), 0.55)
        c.fill()
        c.set_fill_rule(cairo.FILL_RULE_WINDING)
        c.save()
        c.translate(-20, -24)
        c.rotate(-0.6)
        c.scale(11, 5)
        c.arc(0, 0, 1, 0, TAU)
        c.restore()
        c.set_source_rgba(1, 1, 1, 0.75 * (0.4 + 0.6 * g))
        c.fill()
        c.arc(-6, -33, 3, 0, TAU)
        c.set_source_rgba(1, 1, 1, 0.7 * (0.4 + 0.6 * g))
        c.fill()
        # blush
        bl = 0.8 if blush is None else blush
        if expr in ("happy",):
            bl = 1.0
        for s in (-1, 1):
            c.save()
            c.translate(s * 21, 11)
            c.scale(1, 0.6)
            grd = cairo.RadialGradient(0, 0, 0, 0, 0, 9)
            cc = _c("#FF9E8C")
            grd.add_color_stop_rgba(0, *cc, 0.7 * bl * (0.4 + 0.6 * g))
            grd.add_color_stop_rgba(1, *cc, 0)
            c.set_source(grd)
            c.arc(0, 0, 9, 0, TAU)
            c.fill()
            c.restore()

    _cel(ctx, body, None, None, None, line, lw, extra=extra, base_src=gr)

    # ---- eyes
    lx, ly = look
    eye_pos = []
    ex0, ey0 = 11.5, -4.0
    op = 1 - clamp(blink)
    if expr == "sleepy":
        op *= 0.35
    if expr == "surprised":
        op = min(1.0, op * 1.1)
    eye_col = mix("#6E6558", KIRA_EYE, smoothstep(0, 0.7, g))
    for s in (-1, 1):
        cx, cy = s * ex0 + lx * 2.5, ey0 + ly * 2.5
        eye_pos.append((cx, cy))
        ctx.save()
        ctx.translate(cx, cy)
        if expr == "hurt":
            _sc(ctx, eye_col)
            _taper(ctx, [(-s * 6, -6), (s * 4, -0.5), (-s * 6, 5)], 2.6, 2.6, 3.0)
            ctx.restore()
            continue
        if expr == "happy" and op > 0.5:
            _sc(ctx, eye_col)
            _taper(ctx, [(-6, 2.5), (0, -4.5), (6, 2.5)], 2.4, 2.4, 3.2)
            ctx.restore()
            continue
        rx, ry = 6.6, 10.5 * (1.08 if expr == "surprised" else 1.0)
        if op < 0.15:
            _sc(ctx, eye_col)
            _taper(ctx, [(-6, 0), (0, 3), (6, 0)], 2.2, 2.2, 3.0)
            ctx.restore()
            continue
        ctx.save()
        ctx.scale(rx, ry)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        # eyelid: clip top by openness
        ctx.save()
        ctx.clip()
        top = -ry + 2 * ry * (1 - op)
        ctx.rectangle(-rx - 2, top, 2 * rx + 4, 2 * ry + 4)
        gr2 = cairo.LinearGradient(0, -ry, 0, ry)
        gr2.add_color_stop_rgb(0, *_c(eye_col))
        gr2.add_color_stop_rgb(1, *mix(eye_col, "#8A5A30", 0.55))
        ctx.set_source(gr2)
        ctx.fill()
        if expr == "sad":
            ctx.arc(0, ry * 0.55, rx * 0.9, 0, TAU)
            _sc(ctx, "#9ED0FF", 0.35)
            ctx.fill()
        # catch-lights
        ctx.save()
        ctx.translate(-1.8, -ry * 0.45)
        ctx.scale(2.9, 3.6)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        ctx.set_source_rgba(1, 1, 1, 0.97)
        ctx.fill()
        ctx.arc(2.2, ry * 0.45, 1.5, 0, TAU)
        ctx.set_source_rgba(1, 1, 1, 0.85)
        ctx.fill()
        ctx.restore()
        if op < 0.98:
            _sc(ctx, eye_col)
            ctx.set_line_width(1.6)
            ctx.move_to(-rx * 1.05, top)
            ctx.line_to(rx * 1.05, top)
            ctx.stroke()
        if expr == "sad":
            _sc(ctx, eye_col, 0.9)
            _taper(ctx, [(-s * 2, -ry - 5), (s * 6, -ry - 2.5)], 1.6, 1.0)
        ctx.restore()

    # ---- mouth
    mx, my = lx * 1.5, 11.5 + ly * 1.5
    ctx.save()
    ctx.translate(mx, my)
    o = clamp(mouth_open)
    v = vowel if vowel in _VOWEL else "x"
    mcol = mix("#8A7A6A", "#7A3A22", smoothstep(0, 0.7, g))
    if o > 0.08 and v not in ("x", "n"):
        hw, hh, rnd, _t = _VOWEL[v]
        hw *= 0.5
        hh = max(1.4, hh * 0.55 * o)
        ctx.move_to(-hw, 0)
        ctx.curve_to(-hw * 0.5, -hh * 0.3, hw * 0.5, -hh * 0.3, hw, 0)
        ctx.curve_to(hw, hh * (0.6 + 0.4 * rnd), -hw, hh * (0.6 + 0.4 * rnd), -hw, 0)
        ctx.close_path()
        _sc(ctx, "#7A2A2A")
        ctx.fill_preserve()
        _sc(ctx, mcol)
        ctx.set_line_width(1.1)
        ctx.stroke()
    elif expr == "surprised":
        ctx.save()
        ctx.scale(2.4, 3.0)
        ctx.arc(0, 0.3, 1, 0, TAU)
        ctx.restore()
        _sc(ctx, "#7A2A2A")
        ctx.fill()
    elif expr in ("sad", "hurt"):
        _sc(ctx, mcol)
        _taper(ctx, [(-3.2, 1.6), (0, -0.6), (3.2, 1.6)], 1.1, 1.1, 1.7)
    elif expr == "happy":
        ctx.move_to(-4.2, -0.5)
        ctx.curve_to(-2, 0.3, 2, 0.3, 4.2, -0.5)
        ctx.curve_to(3, 4.4, -3, 4.4, -4.2, -0.5)
        ctx.close_path()
        _sc(ctx, "#7A2A2A")
        ctx.fill()
    else:
        _sc(ctx, mcol)
        # tiny 'w' cat-like mouth
        _taper(ctx, [(-3.4, -0.4), (-1.7, 1.3), (0, 0.1), (1.7, 1.3), (3.4, -0.4)], 1.0, 1.0, 1.5)
    ctx.restore()

    ctx.pop_group_to_source()
    ctx.paint_with_alpha(alpha_eff)
    # hot core bloom when blazing
    if g > 0.6:
        k = (g - 0.6) / 0.4
        gr3 = cairo.RadialGradient(0, 0, 0, 0, 0, R * 0.9)
        gr3.add_color_stop_rgba(0, 1, 1, 0.95, 0.35 * k)
        gr3.add_color_stop_rgba(1, 1, 1, 0.9, 0)
        ctx.set_source(gr3)
        ctx.arc(0, 0, R * 0.9, 0, TAU)
        ctx.fill()
    out = {"center": anchor(0, 0), "glow_radius": 60 * scale * (1.1 + 1.6 * g),
           "eyes": [anchor(*e) for e in eye_pos], "mouth": anchor(mx, my), "top": anchor(0, -R)}
    ctx.restore()
    return out


# =============================================================== Lantern
def _lantern_geom(scale=1.0):
    loop = 18.0
    cap_h = 8.0
    rx, ry = 42.0, 46.0
    body_top = loop + cap_h
    body_cy = body_top + ry * 0.88
    return dict(loop=loop, cap_h=cap_h, rx=rx, ry=ry, body_top=body_top, body_cy=body_cy,
                body_bot=body_cy + ry * 0.88)


def draw_lantern_glow(ctx, x, y, scale=1.0, lit=1.0, sway=0.0, t=0.0):
    """Warm halo for a lit lantern (call BEFORE draw_lantern).  (x, y) = hang point."""
    if lit <= 0.01:
        return
    g = _lantern_geom()
    cx = x - math.sin(sway) * g["body_cy"] * scale
    cy = y + math.cos(sway) * g["body_cy"] * scale
    fl = 1 + 0.03 * math.sin(t * 11) + 0.03 * noise1(t * 5, 9)
    R = 210 * scale * lit * fl
    gr = cairo.RadialGradient(cx, cy, 0, cx, cy, R)
    a = _c("#FFD9A0")
    b = _c("#FF9A3D")
    gr.add_color_stop_rgba(0, *a, 0.5 * lit)
    gr.add_color_stop_rgba(0.3, *b, 0.22 * lit)
    gr.add_color_stop_rgba(1, *b, 0)
    ctx.save()
    ctx.set_source(gr)
    ctx.arc(cx, cy, R, 0, TAU)
    ctx.fill()
    ctx.restore()


def draw_lantern(ctx, x, y, scale=1.0, lit=0.0, sway=0.0, t=0.0, kira=0.0, _mirror=1, alpha=1.0):
    """Round white paper lantern (chōchin).  (x, y) = top of the hanging loop.
    lit 0..1 warm inner light; sway = swing angle (rad) around the hang point;
    kira 0..1 = Kira glowing inside (a golden star silhouette shows through the paper).
    Returns {'hang', 'body_center', 'radius', 'top', 'bottom'} in caller coordinates."""
    M0 = ctx.get_matrix()
    M0inv = cairo.Matrix(*M0)
    M0inv.invert()

    def anchor(px, py):
        dx, dy = ctx.user_to_device(px, py)
        return M0inv.transform_point(dx, dy)

    G = _lantern_geom()
    rx, ry, cy = G["rx"], G["ry"], G["body_cy"]
    lit = clamp(lit)
    lw = line_weight(scale, 2.2)
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale, scale)
    ctx.rotate(sway)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    if alpha < 1:
        ctx.push_group()
    # hanging loop (wire)
    _sc(ctx, "#2A2020")
    ctx.set_line_width(1.8)
    ctx.move_to(-9, G["loop"] + 1)
    ctx.curve_to(-8, 4, -3, 0, 0, 0)
    ctx.curve_to(3, 0, 8, 4, 9, G["loop"] + 1)
    ctx.stroke()
    top_cap_y = G["loop"]
    cap_w = rx * 0.52

    def body(c):
        c.save()
        c.translate(0, cy)
        c.scale(rx, ry)
        c.arc(0, 0, 1, 0, TAU)
        c.restore()

    # paper
    if lit > 0:
        gr = cairo.RadialGradient(-4, cy - 6, 2, 0, cy, rx * 1.05)
        gr.add_color_stop_rgb(0, *mix(PAPER, "#FFF6D8", lit))
        gr.add_color_stop_rgb(0.5, *mix(PAPER, "#FFC870", lit))
        gr.add_color_stop_rgb(1, *mix(PAPER_SH, "#F0862C", lit))
        base_src = gr
    else:
        base_src = None

    def extra(c):
        if kira > 0:
            kc = cairo.RadialGradient(0, cy, 0, 0, cy, rx * 0.8)
            kc.add_color_stop_rgba(0, 1, 1, 0.9, 0.9 * kira)
            kc.add_color_stop_rgba(0.5, 1, 0.9, 0.55, 0.5 * kira)
            kc.add_color_stop_rgba(1, 1, 0.8, 0.4, 0)
            c.set_source(kc)
            c.paint()
            c.save()
            c.translate(0, cy + 2)
            c.rotate(math.sin(t * 1.3) * 0.1)
            _rounded_star(c, 0, 0, rx * 0.48, rx * 0.26, -PI / 2, 0.5)
            c.restore()
            _sc(c, "#FFE9A8", 0.55 * kira)
            c.fill()
        # ribs: horizontal bands seen slightly from above
        n = 10
        for i in range(1, n):
            u = -1 + 2 * i / n
            yy = cy + u * ry * 0.93
            hw = rx * math.sqrt(max(0, 1 - (u * 0.93) ** 2))
            c.new_path()
            c.move_to(-hw, yy)
            c.curve_to(-hw * 0.55, yy + hw * 0.16, hw * 0.55, yy + hw * 0.16, hw, yy)
            col = mix(PAPER_SH, "#C8651C", lit)
            _sc(c, col, 0.55 + 0.25 * lit)
            c.set_line_width(1.3 + 0.9 * lit)
            c.stroke()
        # vertical paper panels (subtle)
        for s in (-0.55, 0.0, 0.55):
            c.new_path()
            c.move_to(s * cap_w * 0.95, cy - ry)
            c.curve_to(s * rx * 1.25, cy - ry * 0.4, s * rx * 1.25, cy + ry * 0.4, s * cap_w * 0.95, cy + ry)
            _sc(c, mix(PAPER_SH, "#D9782A", lit), 0.25)
            c.set_line_width(0.9)
            c.stroke()
        # small star mark
        c.save()
        c.translate(rx * 0.18, cy + 4)
        _rounded_star(c, 0, 0, 9, 4.3, -PI / 2, 0.4)
        c.restore()
        _sc(c, mix("#D8404A", "#B8281E", lit), 0.9)
        c.fill()
        # cool shading (unlit) / hot spot (lit)
        if lit < 1:
            c.save()
            c.translate(-9, -6)
            body(c)
            c.restore()
            c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            c.rectangle(-500, -500, 1000, 1000)
            _sc(c, PAPER_SH, 0.75 * (1 - lit))
            c.fill()
            c.set_fill_rule(cairo.FILL_RULE_WINDING)
        if lit > 0:
            hs = cairo.RadialGradient(0, cy + 6, 0, 0, cy + 6, rx * 0.7)
            hs.add_color_stop_rgba(0, 1, 1, 0.92, 0.65 * lit)
            hs.add_color_stop_rgba(1, 1, 0.95, 0.8, 0)
            c.set_source(hs)
            c.paint()

    _cel(ctx, body, PAPER, None, None, mix(PAPER_LINE, "#B0501A", lit), lw, extra=extra, base_src=base_src)

    # caps
    for (yy, h) in ((top_cap_y, G["cap_h"]), (cy + ry * 0.88 - 1, G["cap_h"])):
        def cap(c, yy=yy, h=h):
            c.new_path()
            r = 3
            c.move_to(-cap_w + r, yy)
            c.line_to(cap_w - r, yy)
            c.curve_to(cap_w, yy, cap_w, yy, cap_w, yy + r)
            c.line_to(cap_w, yy + h - r)
            c.curve_to(cap_w, yy + h, cap_w, yy + h, cap_w - r, yy + h)
            c.line_to(-cap_w + r, yy + h)
            c.curve_to(-cap_w, yy + h, -cap_w, yy + h, -cap_w, yy + h - r)
            c.line_to(-cap_w, yy + r)
            c.curve_to(-cap_w, yy, -cap_w, yy, -cap_w + r, yy)
            c.close_path()

        _cel(ctx, cap, mix(LANT_RED, "#E0503A", lit * 0.4), LANT_RED_SH, "#FF9A8A", LANT_RED_LINE, lw * 0.9,
             soff=(-5, -2.5), roff=(-1.5, 0), rim_a=0.5)
    # bottom tassel stub
    _sc(ctx, LANT_RED_LINE)
    ctx.set_line_width(1.6)
    by = cy + ry * 0.88 - 1 + G["cap_h"]
    ctx.move_to(0, by)
    ctx.line_to(0, by + 5)
    ctx.stroke()
    if alpha < 1:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(alpha)
    out = {"hang": anchor(0, 0), "body_center": anchor(0, cy), "radius": rx * scale,
           "top": anchor(0, G["body_top"]), "bottom": anchor(0, by)}
    ctx.restore()
    return out
