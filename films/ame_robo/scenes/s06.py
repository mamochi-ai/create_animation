"""s06 — 49.0–60.0 s: the rain stops, sunbeams, rainbow, thank you, end card (finale).

Shots (local t):
  A  0.00–3.60  shopfront two-shot, slow pull-back.  Rain thins and stops, last drips off the canopy
               rim; Mina notices, looks up and lowers the randoseru (back on her back ~3.3).
               2.0 the sun breaks: god rays sweep, warm flood, glints; Robo folds the canopy 2.6–3.6
               (reverse transformation, neat bounce at the end).
  B  3.60–4.95  street wide crane-up from the rainbow reflected in the wet street to the sky; the
               rainbow sweeps in.  Mina VO 「はれた！」.
  C  4.95–9.30  two-shot: Mina cheers (arms up, hop) 「ありがとう、ロボさん！」 then bows.  6.8 kitten mews,
               hops onto Robo's tread.  7.2 Robo's screen pops a pixel heart, 「どういたしまして。ピピッ。」
               antenna blinks twice on ピピッ, happy wiggle, drying off.
  D  9.30–11.0  final wide at the side-street corner under the rainbow: Mina waves, Robo waves (happy
               beep 9.3), kitten; 「おわり」 bounces in gently ~9.4, calm hold.
"""
import math
from functools import lru_cache

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from common import core
from common.core import clamp, lerp, smoothstep, ease_in_out, ease_out, ease_out_back, noise1
from lib import world
from lib.characters import draw_mina, draw_robo, draw_kitten, draw_droplet, draw_sparkle, \
    draw_drops_on, draw_rim_drips

T0 = 49.0
W, H = 1920, 1080

# --------------------------------------------------------------- staging (shopfront world coords)
FY = 965.0                         # feet on the wet pavement in front of the awning
SD = world.shop_scale(FY)          # depth scale (~1.25)
SM = 0.92 * SD                     # Mina
SR = 0.915 * SD                    # Robo
SK = 0.95 * SD                     # kitten
MX = 760.0
RX = MX + 190 * SM
KX = MX + 112 * SM
CAN_OFF = (420.0, 0.0)

# final wide (shot D)
FY_D = 968.0
SD_D = world.shop_scale(FY_D)
MX_D, RX_D, KX_D = 1215.0, 1470.0, 1345.0
RB_CENTER, RB_RADIUS = (2050.0, 780.0), 1020.0

CUT_B, CUT_C, CUT_D = 3.6, 4.95, 9.3


def seg(t, a, b):
    return clamp((t - a) / (b - a))


def damp_osc(t, freq, decay):
    if t < 0:
        return 0.0
    return math.sin(t * freq * math.tau) * math.exp(-t * decay)


def weather(t):
    return clamp(0.8 * smoothstep(0.15, 1.5, t) + 0.2 * smoothstep(1.5, 2.6, t))


def _drift(t, amp=3.0, seed=0):
    return (amp * noise1(t * 0.35, seed), amp * 0.7 * noise1(t * 0.31, seed + 9))


# --------------------------------------------------------------- camera
def camera(t):
    if t < CUT_B:
        u = ease_in_out(seg(t, 0.0, CUT_B))
        cx, cy, z = lerp(915, 960, u), lerp(655, 560, u), lerp(1.24, 1.0, u)
        dx, dy = _drift(t, 2.5, 1)
        return (cx + dx, cy + dy, z)
    if t < CUT_C:
        u = ease_in_out(seg(t, CUT_B, CUT_C + 0.15))
        return (960.0, lerp(760, 330, u), lerp(1.13, 1.0, u))
    if t < CUT_D:
        u = ease_in_out(seg(t, CUT_C, CUT_D))
        p = ease_in_out(seg(t, 6.5, 7.4))        # attention shifts to Robo
        cx = lerp(880, 975, p) + 10 * u
        z = lerp(1.42, 1.48, u)
        dx, dy = _drift(t, 2.0, 4)
        return (cx + dx, 712 + dy, z)
    u = ease_out(seg(t, CUT_D, 10.3), 2)
    return (lerp(1395, 1380, u), lerp(330, 345, u), lerp(0.74, 0.765, u))


# --------------------------------------------------------------- character poses
def mina_pose(t, tg):
    o, v, talk = core.mouth("mina", tg)
    blink = core.blink(tg, 3)
    p = dict(view="three_quarter_right", t=tg, wet=lerp(0.3, 0.12, seg(t, 2.0, 9.0)), blink=blink,
             mouth_open=o, vowel=v, wind=0.35, light="left")
    if t < CUT_B:
        # notice -> look up -> lower randoseru -> swing on back
        up = smoothstep(0.55, 1.1, t) * (1 - smoothstep(2.9, 3.5, t))
        p.update(head_pitch=0.55 * up, look=(0.35 * up, -0.9 * up))
        if t < 0.6:
            p["expr"] = "gentle"
        elif t < 2.0:
            p["expr"] = "surprised"
        elif t < 2.9:
            p["expr"] = "amazed"
        else:
            p["expr"] = "happy"
        if t < 3.25:
            k = ease_in_out(seg(t, 1.3, 2.7))
            p["arms"] = "hold_randoseru_overhead"
            p["hold_target"] = None  # filled in by caller (needs Robo anchors)
            p["_lower"] = k
            p["hold_tilt"] = -0.12 * math.sin(k * math.pi)
        else:
            p["arms"] = "rest"
            p["bounce"] = 0.16 * damp_osc(t - 3.25, 2.4, 5.0)
        return p
    if t < CUT_D:
        if t < 5.85:
            k = t - CUT_C
            p.update(expr="laugh", arms="arms_up", arm_phase=tg * 3.0,
                     bounce=-0.10 * abs(math.sin(k * 5.2)) * (1 - seg(t, 5.5, 5.85)),
                     tiptoe=0.5 * (1 - seg(t, 5.5, 5.85)), head_pitch=0.15)
            if not talk:
                p["mouth_open"] = 0.7
        elif t < 6.75:
            b = math.sin(seg(t, 5.85, 6.75) * math.pi)
            b = smoothstep(0, 0.6, b)
            p.update(expr="happy", arms="hands_clasped", body_lean=0.30 * b, head_pitch=-0.45 * b,
                     eye_open=lerp(1.0, 0.15, b))
        else:
            p.update(expr="happy", arms="hands_clasped", head_tilt=0.08 * smoothstep(7.0, 7.6, t),
                     look=(0.4, -0.1), blush=0.8)
            if t > 7.25:
                p["expr"] = "amazed" if t < 7.9 else "happy"
            if t > 8.7:  # giggles at the ピピッ
                p.update(expr="laugh", arms="hands_to_mouth",
                         bounce=0.08 * damp_osc(t - 8.75, 2.5, 3.0))
        return p
    # final wide: wave to camera
    p.update(view="three_quarter_right", yaw=0.35, expr="happy", arms="wave", hand="screen_left",
             arm_phase=tg * 2.6 * (1 - 0.5 * seg(t, 10.2, 11.0)), look=(-0.2, 0.0),
             bounce=0.05 * damp_osc(t - CUT_D, 1.8, 3.0))
    p["view"] = None
    return p


def robo_pose(t, tg):
    o = core.mouth("robo", tg)[0]
    p = dict(view="three_quarter_left", t=tg, speak=o, screen_glow=1.0, canopy_offset=CAN_OFF,
             wet=lerp(1.0, 0.3, smoothstep(2.4, 9.0, t)), antenna_blink=True)
    if t < CUT_B:
        if t < 0.9:
            p["face"] = "warm"
        elif t < 2.0:
            p.update(face="scan", look=(0, -1.5))
        elif t < 2.55:
            p.update(face="surprised", look=(0, -1.2))
        else:
            p["face"] = "happy"
        cv = 1.0 - ease_in_out(seg(t, 2.6, 3.45))
        # lib draw_robo raises KeyError('apex') while the pole is ~retracted (canopy < ~0.035)
        p["canopy"] = cv if cv > 0.04 else 0.0
        # anticipation dip before folding, bounce when it clicks shut
        p["squash"] = 0.06 * smoothstep(2.35, 2.6, t) * (1 - seg(t, 2.6, 2.75)) \
            + 0.12 * damp_osc(t - 3.45, 2.6, 5.5)
        p["bob"] = 3 * math.sin(tg * 2.1)
        p["body_tilt"] = 0.04 * damp_osc(t - 3.45, 1.8, 4.0)
        return p
    if t < CUT_D:
        p.update(face="happy", look=(-0.6, -0.3), bob=2.5 * math.sin(tg * 2.3))
        if t >= 7.2:
            k = t - 7.2
            p["face"] = "heart"
            p["squash"] = -0.12 * damp_osc(k, 2.2, 4.5)
            p["screen_glow"] = 1.0 + 0.0 * k
            env = smoothstep(7.4, 7.8, t) * (1 - smoothstep(9.0, 9.3, t))
            p["body_tilt"] = 0.07 * math.sin(tg * 8.0) * env
            p["head_tilt"] = -0.05 * math.sin(tg * 8.0 + 1.0) * env
        # two antenna blinks on ピピッ
        if 8.6 < t < 9.2:
            on = (8.74 <= t < 8.84) or (8.88 <= t < 8.98)
            p["antenna_blink"] = 1.0 if on else 0.2
            p["bob"] += -6 * (1.0 if on else 0.0)
        return p
    p.update(view=None, yaw=-0.45, face="happy", arms="wave", hand="screen_right", arm_phase=tg * 1.4,
             look=(0.0, 0.0), bob=2.5 * math.sin(tg * 2.3), wet=0.3)
    k = t - CUT_D
    p["squash"] = -0.10 * damp_osc(k, 2.4, 4.0)
    p["antenna_blink"] = 1.0 if (0.0 <= k < 0.12 or 0.2 <= k < 0.32) else 0.35
    return p


def kitten_pose(t, tg, robo_feet):
    """Returns (x, y, kwargs)."""
    kw = dict(t=tg, facing=1, blink=core.blink(tg, 21, 2.6), wet=lerp(0.5, 0.1, seg(t, 2.0, 9.0)))
    if t < CUT_B:
        kw.update(pose="nuzzle" if t < 0.9 else "sit", expr="happy" if t > 2.3 else "neutral",
                  look=(0.3, -1) if 0.9 < t < 2.4 else (0, 0))
        return KX, FY, kw
    if t < CUT_D:
        tx, ty = robo_feet[0] - 52 * SR, robo_feet[1] - 40 * SR
        if t < 6.75:
            kw.update(pose="sit", expr="happy", look=(-0.4, -0.6))
            return KX, FY, kw
        if t < 6.98:
            kw.update(pose="stand", expr="mew", look=(0.3, -0.6))
            return KX, FY, kw
        u = seg(t, 6.98, 7.32)
        if u < 1:
            x = lerp(MX + 135 * SM, tx, ease_in_out(u))
            y = lerp(FY, ty, u) - 70 * SK * math.sin(u * math.pi)
            kw.update(pose="stand", expr="happy")
            return x, y, kw
        kw.update(pose="nuzzle" if t < 8.6 else "sit", expr="happy")
        return tx, ty, kw
    kw.update(pose="sit", expr="happy" if not (9.5 < t < 9.8) else "mew", look=(0, -0.2))
    return KX_D, FY_D, kw


# --------------------------------------------------------------- helpers
def _tint_group(ctx, draw, mul, strength=0.65):
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


def _warm_flood(ctx, k, origin=(-200, -300)):
    """Screen-blended warm sunlight from the upper left."""
    if k <= 0.003:
        return
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_SCREEN)
    g = cairo.RadialGradient(origin[0], origin[1], 50, origin[0], origin[1], 2300)
    g.add_color_stop_rgba(0.0, 1.0, 0.92, 0.70, 0.55 * k)
    g.add_color_stop_rgba(0.45, 1.0, 0.88, 0.62, 0.22 * k)
    g.add_color_stop_rgba(1.0, 1.0, 0.85, 0.6, 0.0)
    ctx.set_source(g)
    ctx.paint()
    ctx.restore()


def _hair_glints(ctx, pts, tg, scale, amount, seed=0):
    """Sun twinkles on the water drops on hair/hat/Robo."""
    if amount <= 0:
        return
    for i, p in enumerate(pts):
        ph = (tg * (0.7 + 0.25 * ((i * 37 + seed) % 5)) + i * 0.618) % 1.0
        a = max(0.0, math.sin(ph * math.pi)) ** 3 * amount
        if a > 0.05:
            draw_sparkle(ctx, p[0], p[1], 15 * scale * a, a, ph * 2.0 + i, "#FFFFFF")


def _under_canopy_mask(ctx, ra):
    """Path covering the canopy dome and the dry space beneath it (screen coords at current ctx)."""
    if "canopy_span" not in ra:
        return False
    x0, x1 = ra["canopy_span"]
    ax, ay = ra["canopy_apex"]
    cy = ra["canopy_center"][1]
    ctx.move_to(x0 - 8, cy)
    ctx.curve_to(x0 + (ax - x0) * 0.2, ay - 10, x1 - (x1 - ax) * 0.2, ay - 10, x1 + 8, cy)
    ctx.line_to(x1 + 8, FY + 400)
    ctx.line_to(x0 - 8, FY + 400)
    ctx.close_path()
    return True


@lru_cache(maxsize=24)
def _end_card(size_q, dy_q):
    size = size_q
    cx, cy = 1460, 205 + dy_q
    shadow = core.text_layer("おわり", (cx + 6, cy + 9), size=size, font="maru", color="#1E7F82",
                             stroke=6, stroke_color="#1E7F82")
    shadow = core.blur(shadow, 9)
    glow = core.text_layer("おわり", (cx, cy), size=size, font="maru", color="#FFF6D8")
    txt = core.text_layer("おわり", (cx, cy), size=size, font="maru", color="#FFFFFF",
                          stroke=3, stroke_color="#FFFFFF")
    return shadow, glow, txt


def _comp_end_card(img, t):
    if t < 9.35:
        return
    u = seg(t, 9.4, 10.15)
    a = smoothstep(0.0, 0.45, u)
    if a <= 0.001:
        return
    s = 0.72 + 0.28 * ease_out_back(u, 2.2)
    size = int(round(110 * s / 2) * 2)
    dy = int(round(-14 * (1 - ease_out(u))))
    shadow, glow, txt = _end_card(size, dy)
    core.over(img, shadow, 0.85 * a)
    core.add_glow(img, glow, radius=16, strength=0.45 * a)
    core.over(img, txt, a)


# --------------------------------------------------------------- post
_VIG = None


def _post(img, frame, threshold=0.72, radius=36.0, strength=0.55, vig=0.26, warm=0.0):
    global _VIG
    rgbp = img[..., :3]
    if warm > 0:
        rgbp = rgbp * np.array([1 + 0.035 * warm, 1 + 0.01 * warm, 1 - 0.03 * warm], np.float32)
    small = rgbp[::2, ::2]
    lum = small.max(axis=2, keepdims=True)
    bright = small * np.clip((lum - threshold) / (1 - threshold), 0, 1)
    q = bright[::2, ::2]
    g1 = np.stack([gaussian_filter(q[..., c], radius / 4, mode="nearest") for c in range(3)], -1)
    g2 = np.stack([gaussian_filter(bright[..., c], radius * 0.15, mode="nearest") for c in range(3)], -1)
    g = np.repeat(np.repeat(g1, 2, 0), 2, 1)[:540, :960] * strength + g2 * strength * 0.6
    if warm > 0:
        g = g * np.array([1.0, 0.95, 0.8], np.float32)
    gi = Image.fromarray(np.clip(g * 255, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    g = np.asarray(gi, np.float32) / 255.0
    out = 1 - (1 - rgbp) * (1 - g)
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _VIG = (1 - np.clip(d - 0.55, 0, 1) ** 1.5 * vig)[..., None].astype(np.float32)
    out = out * _VIG + 0.03 * (1 - _VIG)
    core.grain(out, frame, 0.015)
    return out


# --------------------------------------------------------------- shots
def _render_shop(t, tg, frame):
    cam = camera(t)
    w = weather(t) if t < CUT_B else 1.0
    C = world.palette(w)
    final = t >= CUT_D
    sun = smoothstep(1.9, 2.8, t) if t < CUT_B else 1.0
    surf, ctx = core.new_surface()
    if final:
        world.draw_shopfront(ctx, tg, w, cam, rainbow=smoothstep(CUT_D - 0.5, CUT_D + 0.6, t) * 0.85 + 0.15,
                             rainbow_center=RB_CENTER, rainbow_radius=RB_RADIUS)
    else:
        world.draw_shopfront(ctx, tg, w, cam, rainbow=smoothstep(2.6, 3.6, t))

    mp = mina_pose(t, tg)
    rp = robo_pose(t, tg)
    if final:
        mx, rx, fy, sm, sr, sk = MX_D, RX_D, FY_D, 0.92 * SD_D, 0.915 * SD_D, 0.95 * SD_D
    else:
        mx, rx, fy, sm, sr, sk = MX, RX, FY, SM, SR, SK
        if t >= CUT_C:
            rx = MX + 250 * SM
    rp_rim = C["char_rim"]

    def chars(c, reflect=False):
        c.save()
        world._apply_cam(c, cam)
        ra = draw_robo(c, rx, fy, sr, rim=rp_rim, layer="body", **rp)
        mpp = dict(mp)
        low = mpp.pop("_lower", None)
        if mpp.get("arms") == "hold_randoseru_overhead":
            top = (ra["head_top"][0] - 8 * sr, ra["head_top"][1] - 62 * sm)
            down = (mx + 25 * sm, fy - 575 * sm)
            k = low or 0.0
            # pull it back over her own hat, a little dip-and-lift, then it swings onto her back
            mpp["hold_target"] = (lerp(top[0], down[0], k),
                                  lerp(top[1], down[1], k) + 25 * sm * math.sin(k * math.pi))
        kx, ky, kkw = kitten_pose(t, tg, (rx, fy))
        if CUT_C <= t < CUT_D and kx == KX:
            kx = MX + 135 * SM
        if not (CUT_B <= t < CUT_D and t > 7.2):
            ka = draw_kitten(c, kx, ky, sk, **kkw)
        ma = draw_mina(c, mx, fy, sm, rim=rp_rim, **mpp)
        if CUT_B <= t < CUT_D and t > 7.2:
            ka = draw_kitten(c, kx, ky, sk, **kkw)
        c.restore()
        return ma, ra, ka

    def canopy(c):
        if rp.get("canopy", 0) <= 0:
            return {}
        c.save()
        world._apply_cam(c, cam)
        a = draw_robo(c, rx, fy, sr, rim=rp_rim, layer="canopy", **rp)
        c.restore()
        return a

    if not final or True:
        world.draw_ground_reflection(ctx, tg, w, lambda c2: (canopy(c2), chars(c2, True)), cam)

    # canopy behind characters, bg rain behind both
    if C["rain_amt"] > 0:
        world.draw_rain(ctx, tg, w, "bg", cam)
    ca = canopy(ctx)
    ma, ra, ka = _tint_group(ctx, chars, C["char_mul"])

    # mid / fg rain, kept out from under the canopy
    if C["rain_amt"] > 0:
        ctx.push_group()
        world.draw_rain(ctx, tg, w, "mid", cam)
        world.draw_rain(ctx, tg, w, "fg", cam)
        if ca:
            ctx.save()
            world._apply_cam(ctx, cam)
            if _under_canopy_mask(ctx, ca):
                ctx.set_operator(cairo.OPERATOR_DEST_OUT)
                ctx.set_source_rgba(0, 0, 0, 1)
                ctx.fill()
            ctx.restore()
        ctx.pop_group_to_source()
        ctx.paint()

    ctx.save()
    world._apply_cam(ctx, cam)
    sc = sr
    # last drips from the canopy rim
    if ca and "canopy_rim_points" in ca:
        amt = 1.0 - smoothstep(0.3, 1.9, t)
        draw_rim_drips(ctx, ca["canopy_rim_points"], tg, sc * 0.9, seed=4, amount=amt,
                       fall=(fy - ca["canopy_center"][1]) / sc)
        pts = [(p[0], p[1], fy + 6) for i, p in enumerate(ca["canopy_rim_points"]) if i % 2 == 0]
        if t < 3.0:
            world.draw_drips_from(ctx, tg, pts, weather=max(w, 0.7), size=4.5 * sc, gravity=2600,
                                  seed=17, period=(0.6, 1.3), alpha=0.9 * (1 - seg(t, 2.6, 3.0)),
                                  sparkle=sun, rate=1.0)
    if t < CUT_B and 3.18 < t < 3.45:
        k = seg(t, 3.18, 3.45)
        cx_, cy_ = mx - 10 * sm, fy - 400 * sm
        for i, (wdt, al) in enumerate(((22, 0.18), (12, 0.35), (4, 0.7))):
            ctx.set_line_width(wdt * sm)
            ctx.set_source_rgba(1, 0.97, 0.9, al * (1 - k))
            ctx.arc(cx_, cy_, 150 * sm, -1.7 + 1.4 * k, -0.4 + 1.6 * k)
            ctx.stroke()
    # water beading on characters, glistening in the sun
    damp = 1.0 - 0.6 * seg(t, 2.0, 9.0)
    draw_drops_on(ctx, ma["drop_points"], tg, sm, seed=2, amount=0.75 * damp)
    draw_drops_on(ctx, ra["drop_points"], tg, sr, seed=5, amount=0.85 * damp)
    _hair_glints(ctx, ma["drop_points"], tg, sm, 0.9 * sun * damp, 1)
    _hair_glints(ctx, ra["drop_points"], tg, sr, 0.8 * sun * damp, 3)

    # pixel-heart pop sparkle burst at 7.2
    if CUT_C <= t < CUT_D and 7.15 < t < 8.2:
        k = seg(t, 7.2, 8.0)
        sx, sy = ra["screen_center"]
        for i in range(8):
            ang = i * math.tau / 8 + 0.3
            r = (40 + 110 * ease_out(k)) * sr
            a = (1 - k) ** 1.5
            col = "#FF9FC4" if i % 2 else "#FFFFFF"
            draw_sparkle(ctx, sx + math.cos(ang) * r, sy + math.sin(ang) * r * 0.8, 14 * sr * a, a, ang, col)
    # musical-ish happy sparkles around the pair while Robo speaks / in the final
    if t >= 7.4:
        for i in range(10):
            ph = (tg * 0.45 + i * 0.137) % 1.0
            a = math.sin(ph * math.pi) ** 2 * (0.6 if t < CUT_D else 0.8)
            px = lerp(mx - 120 * sm, rx + 140 * sr, (i * 0.618) % 1.0)
            py = fy - (260 + 330 * ((i * 0.381) % 1.0)) * sm - ph * 90 * sm
            draw_sparkle(ctx, px, py, 11 * sm * a, a, ph * 3 + i, "#FFF4C8")
    ctx.restore()

    world.draw_shopfront_fg(ctx, tg, w, cam)

    # god rays sweeping as the clouds part
    if t < CUT_B:
        k = smoothstep(1.9, 2.4, t) * (1 - 0.45 * smoothstep(3.0, 3.6, t))
        if k > 0.01:
            sw = ease_in_out(seg(t, 1.9, 3.6))
            world.draw_sunbeams(ctx, tg, 1.6 * k, origin=(lerp(-700, -150, sw), -650), n=12,
                                color="#FFF0C4", length=3200, spread=(0.55, 1.1), seed=31)
        _warm_flood(ctx, sun * (0.45 + 0.6 * math.exp(-max(0, t - 2.0) * 2.5) * smoothstep(1.95, 2.05, t)))
    else:
        world.draw_sunbeams(ctx, tg, 0.7, origin=(-300, -650), n=9, color="#FFF0C4", length=3200,
                            spread=(0.55, 1.1), seed=31)
        _warm_flood(ctx, 0.45)

    img = core.surface_rgba(surf)
    if final:
        _comp_end_card(img, t)
    flash = 0.25 * math.exp(-max(0, t - 2.0) * 3.0) * smoothstep(1.95, 2.05, t) if t < CUT_B else 0.0
    thr = lerp(0.74, 0.8, sun)
    return _post(img, frame, threshold=thr - flash, radius=lerp(34, 40, sun),
                 strength=lerp(0.5, 0.48, sun) + flash, vig=0.24, warm=sun)


def _render_street(t, tg, frame):
    cam = camera(t)
    w = 1.0
    appear = ease_in_out(seg(t, CUT_B + 0.05, CUT_C - 0.1))
    surf, ctx = core.new_surface()
    world.draw_street(ctx, tg, w, cam, rainbow=appear)
    hz = world.to_screen(0, world.STREET_HORIZON_Y, cam)[1]
    poly = world.street_ground_poly(cam)
    world.draw_glints(ctx, tg, w, poly, n=70, horizon_y=hz, size=18)
    world.draw_sunbeams(ctx, tg, 1.0, origin=(-400, -700), n=11, color="#FFF0C4", length=3600,
                        spread=(0.5, 1.1), seed=33)
    _warm_flood(ctx, 0.5)
    img = core.surface_rgba(surf)
    return _post(img, frame, threshold=0.8, radius=40, strength=0.5, vig=0.24, warm=1.0)


def render(t: float) -> np.ndarray:
    tg = T0 + t
    frame = int(round(tg * core.FPS))
    if CUT_B <= t < CUT_C:
        out = _render_street(t, tg, frame)
    else:
        out = _render_shop(t, tg, frame)
    return (np.clip(out[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
