"""Reference stills for world.py -> build/preview/world_*.png

    python3 src/common/world_sheet.py [name ...]
"""
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from PIL import Image  # noqa: E402

from common import core, world  # noqa: E402

OUT = core.ROOT / "build" / "preview"


def finish(s, glow=None, bloom=(0.72, 36, 0.55)):
    img = core.surface_rgba(s)
    if glow is not None:
        core.over(img, glow)
        core.add_glow(img, glow, 16, 1.2)
    core.bloom(img, *bloom)
    core.vignette(img, 0.3)
    core.grain(img, 0)
    return core.finish(img)


def town(phase, cam_y=0.0, t=2.0, shooting=False):
    s, ctx = core.new_surface()
    world.draw_sky(ctx, phase, t, cam_y=cam_y)
    world.draw_stars(ctx, phase, t, cam_y=cam_y)
    world.draw_moon(ctx, 380, 170 - cam_y * world.SKY_PAR, 30, phase, t)
    world.draw_clouds(ctx, phase, t, cam_y=cam_y)
    if shooting:
        world.draw_shooting_star(ctx, (1080, 330 - cam_y * world.SKY_PAR), 2.62, 620, t, 1.0)
    world.draw_town(ctx, "all", phase, t, cam_y=cam_y)
    return s


def shots():
    yield "s01_night", lambda: finish(town(0.0, shooting=True))
    yield "s01_tilt", lambda: finish(town(0.0, cam_y=-900, shooting=True))
    yield "town_predawn", lambda: finish(town(0.5))
    yield "town_dawn", lambda: finish(town(1.0), bloom=(0.8, 40, 0.4))

    def garden():
        s, ctx = core.new_surface()
        t = 3.0
        light = (1000, 820, 260, "#FFE9A8", 0.9)
        world.draw_garden(ctx, t, 0.05, light=light)
        # stand-in for Kira's crater glow
        world.draw_light_pool(ctx, 1000, 830, 200, 60, "#FFE9A8", 0.7)
        world.draw_grass(ctx, t, 1080, "fg", phase=0.05, light=light)
        glow = core.draw_layer(world.draw_fireflies, t, 26, (380, 380, 1920, 1000), 3)
        return finish(s, glow)
    yield "garden", garden

    def steps(cam_x):
        def f():
            s, ctx = core.new_surface()
            t = 1.0
            world.draw_steps_path(ctx, cam_x, t, 0.12)
            # foot marker where Haru would stand
            wx = cam_x + 960
            sx, sy = world.steps_to_screen(wx, world.steps_height_at(wx), cam_x)
            ctx.set_source_rgba(1, 0.3, 0.3, 0.9)
            ctx.arc(sx, sy, 6, 0, math.tau)
            ctx.fill()
            world.draw_parallax_forest(ctx, cam_x, "fg", 0.12)
            glow = core.draw_layer(world.draw_fireflies, t, 30, (0, 200, 1920, 1000), 5, flow=(-160, 40))
            return finish(s, glow)
        return f
    yield "steps_a", steps(0.0)
    yield "steps_b", steps(2300.0)

    def hill(phase, cam, name_sparks=False):
        def f():
            s, ctx = core.new_surface()
            t = 2.0
            world.draw_hilltop(ctx, phase, t, cam=cam)
            if name_sparks:
                world.draw_sparkles(ctx, 0.5, (1100, 300), seed=2)
            world.draw_hilltop_fg(ctx, phase, t, cam=cam)
            return finish(s)
        return f
    yield "hilltop_medium", hill(0.3, world.HILLTOP_CAM_MEDIUM)
    yield "hilltop_wide", hill(0.55, world.HILLTOP_CAM_WIDE, True)
    yield "hilltop_dawn", hill(1.0, world.HILLTOP_CAM_MEDIUM)

    def attic():
        s, ctx = core.new_surface()
        world.draw_attic_window(ctx, 1.0, 0.0)
        world.draw_attic_sill(ctx, 1.0)
        return finish(s)
    yield "attic", attic


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    want = set(sys.argv[1:])
    for name, fn in shots():
        if want and name not in want:
            continue
        fn()  # warm caches
        t0 = time.time()
        img = fn()
        dt = time.time() - t0
        p = OUT / f"world_{name}.png"
        Image.fromarray(img).save(p)
        print(f"{p}  {dt * 1000:.0f} ms (incl. post)")


if __name__ == "__main__":
    main()
