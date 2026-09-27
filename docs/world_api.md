# World API (src/common/world.py)

Deterministic; 1920x1080 screen px (y down); draws onto a cairo ctx. `phase`: 0 night, 0.5 pre-dawn, 1 dawn.
Previews: build/preview/world_*.png (`python3 src/common/world_sheet.py [name…]`; its `finish()` shows the post chain).
Cost: town+sky 35 ms, steps 32, hilltop 48, garden+fg grass 84, attic 85 (first call per process builds caches ~0.5–1 s). Post (bloom/vignette/grain) ≈400–500 ms.

## Palette
`sky_colors(phase)` → dict of RGB tuples: top, upper, mid, low, horizon, hglow, far, hill, mids, near, fog, rim, roof, roof_near,
cloud, cloud_lit, cloud_under, grass, grass_tip, leaf, leaf_mid, leaf_lit, ground, stone, stone_lit, wood, bark, torii; scalars stars, milky, windows, moon, sun, mist.
Use e.g. `C['rim']` to tint characters' rim light.

## Sky
- `draw_sky(ctx, phase, t, cam_y=0, cam_x=0, horizon_y=None, sun_x=1500)` — gradient, horizon glow, dawn sun glow, Milky Way. Horizon at HORIZON_Y=660 when cam_y=0; parallax SKY_PAR=0.55; negative cam_y = looking up (s01 tilt: animate cam_y ≈ −1400 → 0).
- `draw_stars(ctx, phase, t, cam_y, cam_x, seed=7, horizon_y=None)` ~2600 twinkling stars, 22 with cross flares.
- `draw_moon(ctx, x, y, r, phase, t, illum=0.3, tilt=-2.4)`; `draw_clouds(ctx, phase, t, cam_y, cam_x, alpha, speed)`.
- Always pass the SAME cam_y to draw_sky / draw_stars / draw_clouds / draw_town.

## Effects
- `draw_shooting_star(ctx, head_xy, dir_angle, length, t, intensity=1, path=None, width=1)` — dir_angle = travel direction; `path` = recent head positions (oldest first) for curved/sputtering tails.
- `draw_glint(ctx, x, y, size, alpha, color, rot)` 4-point flare.
- `draw_sparkles(ctx, t_since_burst, center, n=70, speed, life, gravity, seed, spread, direction)` gold burst.
- `draw_fireflies(ctx, t, n, region=(x0,y0,x1,y1), seed, size, flow=(vx,vy))`; `draw_dust_motes(ctx, t, region, n, seed, alpha, light=(cx,cy,r))`.
- `draw_light_pool(ctx, x, y, rx, ry, color, strength)` screen-blended pool of light.
- Glow recipe: `g = core.draw_layer(world.draw_fireflies, ...)`; `core.over(img, g)`; `core.add_glow(img, g, 16, 1.2)`; final `core.bloom(img, 0.72, 36, 0.55)`, `core.vignette`, `core.grain` (hides banding).

## Town (s01, s07, s02 window)
`draw_town(ctx, layer, phase, t, cam_x=0, cam_y=0)`, layer 'far'|'hill'|'mid'|'near'|'all' (draw in that order; parallax TOWN_PAR 0.82/0.9/1.0/1.12).
At cam 0: far roofs y 640–830; hill rises on right (x>1050) to summit y≈352; camphor tree trunk TOWN_TREE=(1745,356), canopy top y≈60; shrine TOWN_SHRINE=(1585,352); steps from TOWN_STEPS=(1150,800) up to (1497,362); mid town y 700–900; big near roofs + wires. Suggested landing TOWN_IMPACT=(560,860). Windows flicker, switch off toward dawn.
Props: `draw_torii(ctx, x, base_y, w, h, phase, base_r=None)`, `draw_toro(ctx, x, base_y, h, phase, lit, t)`, `draw_furin(ctx, x, y, t)`.

## Garden (s03)
`draw_garden(ctx, t, phase, cam_x=0, cam_y=0, sky=True, window=1, light=None)` draws everything behind characters (house wall x<370 with shoji window GARDEN_WINDOW=(70,330,230,220), lawn from GARDEN_GROUND_Y=650, hydrangeas, bg grass). GARDEN_IMPACT=(1000,820) kept clear for Kira. `light=(x,y,r,colour,strength)` warms grass + light pool.
Then characters, then `draw_grass(ctx, t, 1080, 'fg', phase=..., light=...)`. Signature: `draw_grass(ctx, t, y_base (num or fn x→y), layer='bg'|'fg', x_range, phase, seed, density, height, wind, light)`.

## Steps (s04)
`draw_steps_path(ctx, cam_x, t, phase, sky=True, forest=True)` → characters → `draw_parallax_forest(ctx, cam_x, 'fg', phase)`.
Tread k spans world x k·84…(k+1)·84 at y = 900 − 36k. Feet: `steps_height_at(wx)` → `steps_to_screen(wx, wy, cam_x)`. Camera auto-follows diagonal via `steps_cam_y(cam_x)`; stair line always passes STEPS_ANCHOR=(960,820). Keep Haru near screen x 960, move cam_x ≈ 300–400 px/s. `steps_line_y(x)` smooth line for bobbing. Torii every 900 px from x=700; lit lanterns between. Forest layers far/mid/near/fg parallax 0.12/0.45/0.75/1.35.

## Hilltop (s05–s07)
`draw_hilltop(ctx, phase, t, cam=(cx,cy,zoom), sky, stars, sun_x, tree, far, ground, props, wind)` then `draw_hilltop_fg(...)` (fg grass; skipped when zoom<0.6). Camera maps world (cx,cy) to screen centre ×zoom; distant layers at depth 0.35.
HILLTOP_CAM_MEDIUM=(960,540,1) = identity: tree trunk x≈330 (shimenawa rope), canopy fills top-left to y≈300–380; hokora x 700; lit tōrō x 1620; ground y≈800–860; horizon 600; valley town lights on right.
HILLTOP_CAM_WIDE=(1150,−700,0.27): s06 framing, small mound + tree at bottom under huge sky.
Helpers: `hilltop_ground_y(x)`, `hilltop_to_screen(x, y, cam)`, `hilltop_horizon_screen(cam)`. No moon included (call draw_moon). s07: phase=1, sun_x≈1500 → warm horizon; add one bright draw_glint star yourself.

## Attic (s02)
`draw_attic_window(ctx, t, phase=0, view_fn=None, flash=0, lamp=1, chime=True)` → Haru → `draw_attic_sill(ctx, t, phase)`.
Round window ATTIC_WIN=(960,430,r=300); sill top face y≈732–770, front face from ATTIC_SILL_Y=770. `view_fn(ctx)` runs clipped inside the window in town/sky screen coords (scale ATTIC_VIEW_SCALE=0.4) — use for the streak at 7.6 s and garden glow at 8.0 s. `flash` washes the room with cool light.

Caching: static geometry baked into masks recoloured per phase; heavy stacks baked at phase steps of 0.1 and crossfaded. Parallel-safe.
Known weak spots: valley town from hilltop reads as rows of little roofs; dawn town a bit uniformly lilac; hilltop foreground plain; attic window view doesn't twinkle.
