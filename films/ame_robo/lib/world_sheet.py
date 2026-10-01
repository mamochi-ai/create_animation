"""Reference stills for lib/world.py -> films/ame_robo/build/preview/world_*.png

    FILM_DIR=films/ame_robo python3 films/ame_robo/lib/world_sheet.py [name ...]

Each still shows the recommended layer order and post chain (bloom / vignette / grain).
A yellow translucent bar marks where an ~8-year-old (Mina, ~430 px incl. hat) stands at GROUND_Y.
"""
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[3] / "src")]

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import core  # noqa: E402
from lib import world  # noqa: E402

OUT = core.ROOT / "build" / "preview"


_POST_T = [0.0]


def post(s, frame=0, bloom=(0.74, 34, 0.5), vig=0.32):
    t0 = time.time()
    img = core.surface_rgba(s)
    core.bloom(img, *bloom)
    core.vignette(img, vig)
    core.grain(img, frame)
    out = core.finish(img)
    _POST_T[0] = time.time() - t0
    return out


def marker(ctx, cam, x=760, h=430, w=96, a=0.45):
    """Stand-in silhouette for a character standing at GROUND_Y (world x)."""
    ctx.save()
    world._apply_cam(ctx, cam)
    gy = world.GROUND_Y
    ctx.set_source_rgba(0.97, 0.79, 0.2, a)
    ctx.rectangle(x - w / 2, gy - h, w, h)
    ctx.fill()
    ctx.set_source_rgba(0.85, 0.2, 0.2, a)
    ctx.arc(x, gy - h + 40, 42, 0, 6.3)
    ctx.fill()
    ctx.restore()


def street(weather, t, cam, rainbow=0.0):
    s, ctx = core.new_surface()
    world.draw_street(ctx, t, weather, cam, rainbow=rainbow)
    hz = world.to_screen(0, world.STREET_HORIZON_Y, cam)[1]
    poly = world.street_ground_poly(cam)
    world.draw_rain(ctx, t, weather, "bg", cam)
    world.draw_rain(ctx, t, weather, "mid", cam)
    world.draw_splashes(ctx, t, weather, poly, horizon_y=hz)
    world.draw_glints(ctx, t, weather, poly, n=60, horizon_y=hz, size=16)
    world.draw_rain(ctx, t, weather, "fg", cam)
    return s


def shop(weather, t, cam=world.CAM_ID, dof=0.0, rainbow=0.0, mark=True):
    s, ctx = core.new_surface()
    world.draw_shopfront(ctx, t, weather, cam, dof=dof, rainbow=rainbow)
    if mark:
        world.draw_ground_reflection(ctx, t, weather, lambda c: marker(c, cam, a=0.9), cam)
        marker(ctx, cam)
    world.draw_rain(ctx, t, weather, "bg", cam, dof=dof / 20)
    world.draw_rain(ctx, t, weather, "mid", cam, dof=dof / 20)
    world.draw_shopfront_fg(ctx, t, weather, cam)
    world.draw_rain(ctx, t, weather, "fg", cam, dof=dof / 20)
    return s


def canopy_fx():
    """s04 helper demo: water sheeting off a big umbrella (stand-in canopy drawn here)."""
    import math
    t = 2.0
    cam = (900, 560, 1.15)
    s, ctx = core.new_surface()
    world.draw_shopfront(ctx, t, 0.0, cam)
    cx, cy, r = 900, 330, 520
    # stand-in teal/cream striped canopy (dome above the rim ellipse)
    ctx.save()
    n = 8
    for k in range(n):
        a0 = math.pi + k * math.pi / n
        a1 = math.pi + (k + 1) * math.pi / n
        ctx.move_to(cx, cy - 300)
        ctx.line_to(cx + math.cos(a0) * r, cy + 40)
        ctx.curve_to(cx + (math.cos(a0) + math.cos(a1)) * r * 0.5, cy + 70,
                     cx + (math.cos(a0) + math.cos(a1)) * r * 0.5, cy + 70, cx + math.cos(a1) * r, cy + 40)
        ctx.close_path()
        ctx.set_source_rgb(*((0.27, 0.62, 0.6) if k % 2 == 0 else (0.95, 0.92, 0.84)))
        ctx.fill()
    ctx.restore()
    marker(ctx, (960, 540, 1), x=cx, h=430)
    world.draw_rain(ctx, t, 0.0, "bg", cam)
    world.draw_rain(ctx, t, 0.0, "mid", cam)
    world.draw_canopy_rain(ctx, t, (cx, cy + 40), r, 0.0, ry=40, ground_y=world.to_screen(0, world.GROUND_Y + 40, cam)[1])
    world.draw_shopfront_fg(ctx, t, 0.0, cam, drips=False)
    world.draw_rain(ctx, t, 0.0, "fg", cam)
    return s


def shots():
    # (a) s01: high-angle establishing street in heavy rain (+ the crane start framing)
    yield "s01_street_rain", lambda: post(street(0.0, 2.0, world.STREET_CAM_HIGH))
    yield "s01_crane_start", lambda: post(street(0.0, 0.5, world.STREET_CAM_START))
    # (b) ground-level shopfront, heavy rain
    yield "shopfront_rain", lambda: post(shop(0.0, 3.0))
    # (c) weather 0.5: rain thinning, clouds breaking
    yield "shopfront_w05", lambda: post(shop(0.5, 3.0))
    yield "street_w05", lambda: post(street(0.5, 3.0, world.STREET_CAM_HIGH))
    # (d) weather 1: rainbow, glistening street
    yield "street_sunny_rainbow", lambda: post(street(1.0, 4.0, (960, 300, 1.0), rainbow=1.0), bloom=(0.82, 40, 0.45))
    yield "shopfront_sunny", lambda: post(shop(1.0, 4.0, rainbow=1.0), bloom=(0.82, 40, 0.45))
    # (e) close framing for character close-ups (shallow depth of field)
    yield "closeup_rain", lambda: post(shop(0.0, 3.0, (760, 600, 1.9), dof=14.0))
    yield "fx_canopy_rain", lambda: post(canopy_fx())
    yield "closeup_sunny", lambda: post(shop(1.0, 3.0, (1150, 600, 1.9), dof=14.0), bloom=(0.82, 40, 0.45))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    want = set(sys.argv[1:])
    for name, fn in shots():
        if want and name not in want:
            continue
        t0 = time.time()
        fn()  # warm caches
        t1 = time.time()
        img = fn()
        dt = time.time() - t1
        p = OUT / f"world_{name}.png"
        Image.fromarray(img).save(p)
        print(f"{p.name:34s} first {t1 - t0:5.1f}s | cached frame: draw {(dt - _POST_T[0]) * 1000:4.0f} ms"
              f" + post {_POST_T[0] * 1000:4.0f} ms")


if __name__ == "__main__":
    main()
