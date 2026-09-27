"""s07 — Dawn: goodbye, end card (global 55.0–60.0 s).

Shot A (0.00–0.95): low-angle medium-wide, Haru's back under the camphor tree,
          looking up at the last star (Kira); breeze, light motes, birds.
Shot B (0.95–5.00): closer 3/4 shot, Haru looks up, says 「……またね、キラ。」,
          waves (2.4–3.8), smile brightens; 「おわり」 fades into the sky while the
          camera pulls back / drifts up toward the star; calm hold for the fade.
"""
import math

import numpy as np

from common import core
from common.core import clamp, lerp, ease_in_out, ease_out, smoothstep, noise1, TAU
from common import world
from common.characters import draw_haru
from common.world import draw_hilltop, draw_hilltop_fg, hilltop_ground_y, draw_glint, draw_dust_motes, draw_light_pool

T0, T1 = core.scene_bounds("s07")
SUN_X = 1500.0
HARU_X = 1000.0
HARU_S = 0.95
CUT = 0.95
RIM = "#FFC27A"
GOLD = (1.0, 0.80, 0.45)

_YY = np.linspace(0, 1, core.H, dtype=np.float32)[:, None]


# ------------------------------------------------------------------ timing
def phase_at(t):
    return lerp(0.85, 1.0, smoothstep(0.0, 4.2, t))


def star_pulse(t):
    """Twinkle 'answers' at 0.4 s and 2.0 s: fast rise, slow decay."""
    p = 0.0
    for tp in (0.4, 2.0):
        d = t - tp
        if d > -0.12:
            p += (smoothstep(-0.12, 0.0, d) if d < 0 else math.exp(-d * 3.2))
    return p


def camera(t):
    """Returns (cx, cy, zoom) hilltop camera + handheld jitter."""
    hx = 3.0 * noise1(t * 0.7, 11) + 1.5 * noise1(t * 1.9, 12)
    hy = 2.5 * noise1(t * 0.6, 13) + 1.2 * noise1(t * 2.1, 14)
    if t < CUT:
        k = ease_in_out(t / CUT)
        cx, cy, z = lerp(905, 925, k), lerp(470, 462, k), lerp(1.0, 1.05, k)
        return cx + hx, cy + hy, z
    # shot B: slight push during the line, then pull back & drift up toward the star
    u = t - CUT
    push = ease_out(u / 1.6, 2)
    cx, cy, z = 1212 + 10 * push, 408 - 4 * push, 1.66 + 0.05 * push
    pb = ease_in_out(smoothstep(3.0, 4.35, t) * 0.5 + invl(2.9, 4.3, t) * 0.5)
    cx = lerp(cx, 1150, pb)
    cy = lerp(cy, 300, pb)
    z = lerp(z, 1.36, pb)
    calm = 1.0 - 0.7 * smoothstep(3.6, 4.4, t)   # handheld settles for the hold
    return cx + hx * calm, cy + hy * calm, z


def invl(a, b, x):
    return clamp((x - a) / (b - a))


def star_pos(t, cam):
    cx, cy, z = cam
    if t < CUT:
        bx, by, rcx, rcy = 1215.0, 150.0, 905.0, 470.0
    else:
        bx, by, rcx, rcy = 1150.0, 150.0, 1212.0, 408.0
    # sky is far away: tiny parallax
    return bx - (cx - rcx) * 0.12 * z, by - (cy - rcy) * 0.35 * z


# ------------------------------------------------------------------ helpers
def _cam_apply(ctx, cam):
    cx, cy, z = cam
    ctx.translate(960, 540)
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)


def _birds(ctx, t, cam):
    """A small flock crossing the dawn sky (tiny flapping silhouettes)."""
    rng = np.random.default_rng(21)
    col = (0.30, 0.22, 0.28)
    for i in range(6):
        ox, oy = rng.uniform(-60, 60), rng.uniform(-30, 30)
        sp = rng.uniform(150, 175)
        x = 1080 + i * 38 + ox + sp * t
        y = 330 + oy - 22 * t + 6 * math.sin(t * 1.3 + i)
        if t >= CUT:          # further away in shot B
            x = 700 + i * 26 + ox * 0.6 + 95 * (t - CUT)
            y = 330 + oy * 0.6 - 12 * (t - CUT)
            s = 0.6
        else:
            s = 1.0
        f = math.sin(t * 11 + i * 1.7)
        w = 9 * s * (1 + 0.1 * i % 3)
        ctx.move_to(x - w, y - f * 5 * s)
        ctx.curve_to(x - w * 0.5, y - f * 4 * s - 2 * s, x - 2 * s, y - 1 * s, x, y + 1.5 * s)
        ctx.curve_to(x + 2 * s, y - 1 * s, x + w * 0.5, y - f * 4 * s - 2 * s, x + w, y - f * 5 * s)
        ctx.set_line_width(2.2 * s)
        ctx.set_source_rgba(*col, 0.75)
        ctx.stroke()


def _rim_from_mask(alpha, dx, dy, blur_r=2.0):
    """Edge facing the light: alpha minus alpha shifted away from the light."""
    sh = np.zeros_like(alpha)
    h, w = alpha.shape
    sy0, sy1 = max(0, dy), h + min(0, dy)
    sx0, sx1 = max(0, dx), w + min(0, dx)
    sh[sy0:sy1, sx0:sx1] = alpha[sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx]
    r = np.clip(alpha - sh, 0, 1)
    return core.blur(r[..., None], blur_r, downscale=2)[..., 0]


def _warm_grade(img, t):
    rgbp = img[..., :3]
    # warm peach wash at the horizon band
    band = np.exp(-((_YY - 0.62) / 0.2) ** 2) * 0.16
    wash = np.array([1.0, 0.70, 0.45], np.float32)
    rgbp[...] = 1 - (1 - rgbp) * (1 - band[..., None] * wash)
    # pull the lilac haze toward peach/gold, gentle S-curve for depth
    rgbp *= np.array([1.07, 0.99, 0.86], np.float32)
    np.clip(rgbp, 0, 1, out=rgbp)
    rgbp[...] = rgbp + 0.35 * (rgbp * rgbp * (3 - 2 * rgbp) - rgbp)
    lum = rgbp.mean(axis=2, keepdims=True)
    rgbp[...] = lum + (rgbp - lum) * 1.18        # a touch more saturation
    rgbp += (1 - lum) ** 3 * np.array([0.02, 0.0, 0.03], np.float32)   # plum in deep shadows
    return img


def _sun(ctx, cam):
    hs = world.hilltop_horizon_screen(cam)
    y = hs - 75
    draw_light_pool(ctx, SUN_X, y, 900, 380, color="#FFB070", strength=0.40)
    draw_light_pool(ctx, SUN_X, y, 380, 200, color="#FFD9A0", strength=0.55)
    draw_light_pool(ctx, SUN_X, y + 8, 90, 70, color="#FFF4DC", strength=0.9)


def _haru_pose(t, tg):
    """Pose parameters for Haru at local time t."""
    bl = core.blink(tg, seed=12, period=3.1)
    mo, vw, _ = core.mouth("haru", tg)
    if t < CUT:
        # back view, looking up; small settling sway
        return dict(view="back", head_pitch=0.55 + 0.05 * math.sin(t * 1.3), head_tilt=0.04 * math.sin(t * 0.9),
                    arms="rest", wind=1.2, wind_dir=1, blink=0.0, rim=RIM, light="right",
                    expr="gentle")
    look_up = 0.95 + 0.04 * math.sin(t * 0.8)
    expr = "gentle"
    smile = None
    if t < 2.4:
        smile = 0.25 + 0.15 * smoothstep(2.0, 2.4, t)   # small sad-happy smile
    hand_l = None
    arms = "rest"
    if t >= 2.3:
        expr = "happy" if t >= 2.45 else "gentle"
        # anticipation dip (2.3-2.4) -> raise (2.4-2.72) -> wave (-3.8) -> lower (3.9-4.5)
        rest = (40.0, 14.0, -22.0)
        dip = (44.0, 18.0, -8.0)
        up = (110.0, 40.0, -300.0)
        if t < 2.4:
            k = ease_in_out((t - 2.3) / 0.1)
            hand_l = tuple(lerp(a, b, k) for a, b in zip(rest, dip))
        elif t < 2.72:
            k = core.ease_out_back((t - 2.4) / 0.32, 1.3)
            hand_l = tuple(lerp(a, b, k) for a, b in zip(dip, up))
        elif t < 4.5:
            amp = smoothstep(2.72, 2.9, t) * (1 - smoothstep(3.55, 3.95, t))
            wv = math.sin((t - 2.72) * 1.7 * TAU) * amp
            hold = (110 + 20 * wv, 40 - 6 * wv, -300 + 8 * abs(wv))
            if t < 3.95:
                hand_l = hold
            else:
                k = ease_in_out((t - 3.95) / 0.55)
                low = (40.0, 14.0, -22.0)   # hand comes back down to rest
                hand_l = tuple(lerp(a, b, k) for a, b in zip(hold, low))
        else:
            hand_l = (40.0, 14.0, -22.0)
    # after the line: small bounce of joy on the wave, then settles
    bounce = 0.0
    if 2.4 <= t < 3.2:
        bounce = -0.05 * math.sin((t - 2.4) / 0.8 * math.pi)
    tilt = 0.05 * math.sin(t * 0.9) + (0.07 * smoothstep(2.5, 3.0, t) * (1 - 0.5 * smoothstep(3.8, 4.6, t)))
    return dict(yaw=0.5, expr=expr, smile=smile, mouth_open=mo, vowel=vw, blink=bl,
                head_pitch=look_up, look=(0.4, 0.8), head_tilt=-0.12 - tilt, arms=arms, hand_l=hand_l,
                bounce=bounce, wind=1.1, wind_dir=-1, rim=RIM, light="right")


def _screen_glow(img, src_rgb, radius, strength, tint=None, box=None, ds=None):
    """add_glow restricted to a box (x0, y0, x1, y1) and/or with a forced downscale."""
    if box is None:
        box = (0, 0, core.W, core.H)
    pad = int(radius * 3)
    x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x1, y1 = min(core.W, int(box[2]) + pad), min(core.H, int(box[3]) + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return img
    g = core.blur(np.ascontiguousarray(src_rgb[y0:y1, x0:x1]), radius, downscale=ds)
    if tint is not None:
        g = g * np.array(tint, np.float32)
    sub = img[y0:y1, x0:x1, :3]
    sub[...] = 1 - (1 - sub) * (1 - np.clip(g * strength, 0, 1))
    return img


def _over_box(img, src, box, opacity=1.0):
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(core.W, int(box[2])), min(core.H, int(box[3]))
    if x1 > x0 and y1 > y0:
        core.over(img[y0:y1, x0:x1], src[y0:y1, x0:x1], opacity)


def _bloom(img, threshold=0.86, radius=38.0, strength=0.38):
    """core.bloom with both blur passes done at reduced resolution (same look, faster)."""
    rgbp = img[..., :3]
    lum = rgbp.max(axis=2, keepdims=True)
    bright = rgbp * np.clip((lum - threshold) / (1 - threshold + 1e-6), 0, 1)
    g = core.blur(bright, radius, downscale=6) * strength + core.blur(bright, radius * 0.3, downscale=3) * strength * 0.6
    img[..., :3] = 1 - (1 - rgbp) * (1 - np.clip(g, 0, 1))
    return img


_VIG = None


def _vignette(img, strength=0.32, color=(0.16, 0.08, 0.10)):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:core.H, 0:core.W].astype(np.float32)
        d = np.sqrt(((xx - core.W / 2) / (core.W * 0.62)) ** 2 + ((yy - core.H / 2) / (core.H * 0.62)) ** 2)
        _VIG = (np.clip(d - 0.55, 0, 1)[..., None] ** 1.5 * strength).astype(np.float32)
    img[..., :3] = img[..., :3] * (1 - _VIG) + np.array(color, np.float32) * _VIG
    return img


# ------------------------------------------------------------------ render
_TEXT = None
TEXT_BOX = (1250, 200, 1840, 430)


def _text():
    """Cached end-card layer + its amber halo (static)."""
    global _TEXT
    if _TEXT is None:
        tx = core.text_layer("おわり", (1540, 310), size=96, font="klee", color="#FFF6E0", spacing=12,
                             stroke=2, stroke_color="#FFF6E0")
        halo = core.blur(tx[..., 3:4], 16)[..., 0]
        halo = np.clip(halo * 2.0, 0, 1) * 0.62
        _TEXT = (tx, halo)
    return _TEXT


def render(t: float) -> np.ndarray:
    tg = T0 + t
    frame = int(round(t * core.FPS))
    ph = phase_at(t)
    cam = camera(t)

    # ---------------- background world
    s, ctx = core.new_surface()
    draw_hilltop(ctx, ph, tg, cam=cam, stars=False, sun_x=SUN_X, wind=1.0, props=t < CUT)
    _birds(ctx, t, cam)
    _sun(ctx, cam)
    img = core.surface_rgba(s)

    # gold rim on the camphor canopy (edges facing the low sun on the right)
    tl = core.draw_layer(lambda c: draw_hilltop(c, ph, tg, cam=cam, sky=False, far=False, ground=False,
                                                props=False, tree=True, wind=1.0))
    rim = _rim_from_mask(tl[..., 3], -7, 5, 1.6) * tl[..., 3]
    rl = np.zeros_like(img)
    rl[..., :3] = rim[..., None] * np.array(GOLD, np.float32) * 0.85
    img[..., :3] = 1 - (1 - img[..., :3]) * (1 - np.clip(rl[..., :3], 0, 1))
    _screen_glow(img, rl[..., :3], 10, 0.8, ds=3)

    # ---------------- Haru
    s2, c2 = core.new_surface()
    c2.save()
    _cam_apply(c2, cam)
    gy = hilltop_ground_y(HARU_X) + 4
    # warm light pool at her feet from the sunrise
    draw_light_pool(c2, HARU_X + 40, gy, 300, 60, color="#FFC890", strength=0.25)
    pose = _haru_pose(t, tg)
    draw_haru(c2, HARU_X, gy, HARU_S, t=tg, **pose)
    c2.restore()
    # foreground grass on the same layer
    draw_hilltop_fg(c2, ph, tg, cam=cam, wind=1.0)
    core.over(img, core.surface_rgba(s2))

    # ---------------- light motes / petals of light
    ml = core.draw_layer(draw_dust_motes, tg, (0, 0, core.W, core.H), 46, 17, "#FFE2A8", 1.3, 0.55,
                         None, (14.0, -10.0))
    core.over(img, ml)
    _screen_glow(img, ml[..., :3], 8, 0.7, ds=4)

    # ---------------- the last star: Kira
    sx, sy = star_pos(t, cam)
    p = star_pulse(t)
    breathe = 0.5 + 0.5 * math.sin(tg * 2.3)
    size = 40 + 7 * breathe + 80 * p
    sl = core.draw_layer(lambda c: (
        draw_light_pool(c, sx, sy, 90 + 40 * p, None, "#FFE9A8", 0.35 + 0.25 * p),
        draw_glint(c, sx, sy, size, clamp(0.9 + 0.1 * p), "#FFF1C8", rot=0.08 * math.sin(tg * 0.7) + 0.35 * p),
        draw_glint(c, sx, sy, size * 0.45, 0.6 * clamp(p), "#FFFFFF", rot=math.pi / 4 + 0.2 * p, glow=False),
    ))
    r = max(size, 130 + 40 * p) + 10
    sbox = (sx - r, sy - r, sx + r, sy + r)
    _over_box(img, sl, sbox)
    _screen_glow(img, sl[..., :3], 22, 1.0 + 0.8 * p, tint=(1.0, 0.92, 0.72), box=sbox)

    # ---------------- end card
    a = ease_in_out(invl(3.0, 4.6, t))
    if a > 0.002:
        tx, halo = _text()
        x0, y0, x1, y1 = TEXT_BOX
        hsub = halo[y0:y1, x0:x1, None] * a
        amber = np.array([0.86, 0.52, 0.28], np.float32)
        sub = img[y0:y1, x0:x1, :3]
        sub[...] = sub * (1 - hsub) + amber * hsub
        _screen_glow(img, tx[..., :3] * a, 5, 0.5 * a, tint=(1.0, 0.85, 0.55), box=TEXT_BOX)
        _over_box(img, tx, TEXT_BOX, a)

    # ---------------- post
    _warm_grade(img, t)
    _bloom(img)
    _vignette(img)
    core.grain(img, frame, 0.016)
    return core.finish(img)
