# World API: `films/ame_robo/lib/world.py` (import as `lib.world`)

Rainy 商店街, the dagashi shopfront 「ねこや」, rain/drips/splashes, overcast→clearing sky, sunbeams, rainbow and after-rain glints.
It's deterministic: there's no RNG state, caches are pure functions of their keys, and renders match across processes, so it's parallel-safe.
Units are 1920×1080 screen px (y down), and everything draws onto a cairo ctx.
Previews: `FILM_DIR=films/ame_robo python3 films/ame_robo/lib/world_sheet.py [name…]` writes `build/preview/world_*.png`. Its `shop()` / `street()` / `post()` functions show the layer order and the post chain.

## Conventions
- **weather** 0..1 (continuous): 0 = heavy rain and overcast, 0.5 = rain thinning and clouds breaking, 1 = after rain, sunny and sparkling. Rain density is `palette(w)['rain_amt']`, which is 0 from w≈0.8. For s06, ramp w from 0 to 1 over about 49.5–52 s.
- **cam = (cx, cy, zoom)**: world point (cx, cy) maps to the screen centre, scaled by zoom. `CAM_ID = (960, 540, 1)` is the identity (world = screen).
  - `to_screen(x, y, cam, depth=1)` and `from_screen(sx, sy, cam)` convert between the two. Use `depth<1` for parallax layers.
  - Use the **same cam** for the background, the characters (`to_screen` their feet) and the fg calls.
- **`palette(weather)`** returns a dict.
  - RGB colours: sky_top/mid/hor, sun_glow, cloud_dark/mid/lit/rim, haze, amb, street, street_hi, pave, pave_dry, rain, warm (#FFC777 shop light), vend (#CFF3FF), sun, sky_refl, char_rim, shadow, glint.
  - Scalars: `weather`, `sunk` (sun strength), `lamp` (artificial lights), `rain_amt`, `fog`, `refl`, `ripple`, `glints`.
  - `char_mul`: an RGB multiply tint suggested for characters (cool grey-blue in rain, warm in sun). Use `char_rim` for rim light.

## Shopfront, ground level (s02–s06)
- `draw_shopfront(ctx, t, weather, cam=CAM_ID, sky=True, dof=0, rainbow=0, rainbow_center=None, rainbow_radius=None, ripples=True)` draws everything **behind** the characters:
  - Sky (visible over the side street on the right, and when tilting up).
  - The facade: a lit lattice window, glass sliding doors with candy jars, a curtain and an 「おやすみ」 notice.
  - The red/cream awning with a scalloped valance reading 「だがし・ねこや」.
  - The signboard 「だがし ねこや商店」, the 2F and the tiled eave (when tilting up).
  - Props: a blue bench, an empty umbrella stand, a hydrangea and other pots, 2 gachapon machines and a glowing vending machine.
  - The dry patch under the awning, and the wet pavement in front with rippling mirrored reflections and puddles.
  - `dof` = background blur radius in px, for close-ups (e.g. cam zoom 1.9 with dof 14).
- **Recommended order:**
  1. `draw_shopfront`
  2. `draw_ground_reflection(ctx, t, w, draw_fn, cam)` (optional; character reflections in the wet zone)
  3. characters
  4. `draw_rain(..., 'bg')`, `draw_rain(..., 'mid')`
  5. `draw_shopfront_fg(ctx, t, w, cam)`: the drip curtain from the valance, splashes on the wet pavement and side street, glints and sun shafts when sunny
  6. `draw_rain(..., 'fg')`
  7. post: `core.bloom(img, .74, 34, .5)` (or `(.82, 40, .45)` when sunny), `vignette(.32)`, `grain`
- `draw_ground_reflection(ctx, t, weather, draw_fn, cam, axis_y=GROUND_Y, strength=.75, blur=4)`:
  - `draw_fn(ctx2)` draws the characters in screen coords. The result is mirrored about `axis_y` (world), blurred, rippled and masked to the wet areas.
  - Costs about 50 ms.
- **Constants** (world px = screen px at CAM_ID):

  | Constant | Value | Meaning |
  |---|---|---|
  | `GROUND_Y` | 880 | Characters' feet (`ground_y(x)` returns 880; flat) |
  | `SHOP_PX_PER_M` | ≈344 | Scale at GROUND_Y: Mina (1.25 m + hat) ≈ 430–450 px tall, Robo ≈ 0.85 × that |
  | `shop_scale(y_feet)` | | Relative scale when walking in depth (1.0 at 880) |
  | `SHOP_HORIZON_Y` | 536 | Eye level (camera 1 m high), VP x = 1660 |
  | `WALL_BASE_Y` | ≈857 | Base of the wall |
  | `DRIP_GROUND_Y` | ≈906 | Drip line; the dry patch is 857–906, wet in front of it |
  | `AWNING_WALL_Y` | ≈54 | Awning underside meets the wall |
  | `AWNING_TOP_Y` | ≈91 | Valance top |
  | `AWNING_EDGE_Y` | ≈191 | Valance scallop bottoms = drip edge |
  | `AWNING_X0`, `AWNING_X1` | ≈-174, 1616 | The left end is off-screen |
  | `awning_drip_points(cam)` | | The 14 scallop tips in screen coords (use one for the 9.8 s drip on Mina's hat: `draw_drips_from` with a custom ground_y = hat top) |
  | `DOOR_RECT` | (616, 199, 1066, 857) | |
  | `WINDOW_RECT` | (215, 263, 504, 552) | |
  | `BENCH_SEAT` | (190, 728, 653) | x0, seat y, x1 (kitten can sit here) |
  | `GACHAPON_X` | (1069, 1311) | |
  | `VENDING_RECT` | (1331, 257, 1566, 872) | |
  | `SIDE_STREET_X` | ≈1580 | Right of this, a wet side street recedes; Robo can roll in from the right along y = 880 |
  | `SHOP_RAINBOW` | ((2350, 1050), 1250) | Default rainbow centre/radius (world). Visible over the side street, or tilt up e.g. cam (1300, -250, 0.8) |
  | `SHOPFRONT_CAM` | CAM_ID | Medium wide |

  Suggested close-ups: Mina (760, 600, 1.9) with dof ≈ 14; Robo (1150, 600, 1.9).
- `shop_to_screen(X, Y, Z, cam)` maps 3D metres to screen. The wall is at Z = 9, the awning front at Z = 7.8 and characters at Z = 8.4.

## Street, high angle (s01; wide after-rain views in s06)
- `draw_street(ctx, t, weather, cam=STREET_CAM_HIGH, sky=True, passersby=True, near=True, rainbow=0, rainbow_center=(1350,700), rainbow_radius=820, drips=True)` draws:
  - The sky, plus an optional rainbow that is also reflected in the street.
  - The wet street with rippling mirrored reflections of shops, signs, lights and sky; puddles are sharper.
  - Two rows of 2–3-storey shops in one-point perspective, with:
    - striped awnings
    - vertical hanging signs: だがし パン くすり とうふ ほん きっさ はな さかな ゆ やおや
    - facade boards and closed shutters
    - lit windows with curtains
    - a vending machine and a bus stop 「バス」
    - lamp posts with 「にじいろ」 banners and bunting
  - The far arch 「にじいろ商店街」, utility poles and sagging wires (beads that drip and sparkle after the rain), umbrella passers-by and awning drips.
- **Rain is separate:**
  - `draw_rain` bg/mid/fg.
  - `draw_splashes(ctx, t, w, street_ground_poly(cam), horizon_y=to_screen(0, STREET_HORIZON_Y, cam)[1])`.
  - `draw_glints(...)` with the same region.
- **Camera:**
  - `STREET_CAM_HIGH = (960, 540, 1)` is the establishing shot.
  - `STREET_CAM_START = (960, -170, 1.1)` looks up into the rain, the wires and the pole tops.
  - Crane down by easing START→HIGH. The plates cover zoom ≥ 0.95 and are crisp up to about 1.2.
  - For the s06 rainbow wide shot use cam (960, 300, 1) and `rainbow=appear`.
- **Constants:** `STREET_HORIZON_Y = 290` (VP x 960), eye 5.6 m, f = 1450, facades at ±3.6 m, gutters at ±3.0 m.
- **Helpers:**
  - `street_to_screen(X, Y, Z, cam)` maps 3D metres to screen.
  - `street_ground_poly(cam)` returns the walkable street polygon.
  - `draw_passersby(ctx, t, w, cam, mirror, alpha)` draws the pedestrians on their own. Umbrellas fold away when w > 0.6.

## Sky & light
- `draw_sky(ctx, t, weather, cam, horizon_y, depth=0.35, sunbeams=True)`:
  - Gradient plus two drifting perspective cloud decks: low heavy overcast → breaking → white cumulus on blue.
  - Pass `horizon_y=SHOP_HORIZON_Y` or `STREET_HORIZON_Y`, and depth = 1 to stay aligned with the ground plates.
  - `sky_array(...)` returns the memoised low-res RGB array, which is reused for reflections.
- `draw_sunbeams(ctx, t, strength, origin, n, color, length, spread, seed)` draws crepuscular rays at ¼ res. The sun is up-left, behind the camera.
- `draw_rainbow(ctx, t, appear, center, radius, width=None, alpha=.55, secondary=True, fade_y=None)`:
  - `appear` 0..1 sweeps the arc in from its left foot and fades it in.
  - Drawn at ½ res, with a faint secondary bow and a brighter sky inside the bow.

## Water FX (all screen px)
- `draw_rain(ctx, t, weather, layer='bg'|'mid'|'fg', cam, wind=0.2, seed, density=1, color=None, region=None, dof=0)`:
  - Slanted streaks with a head/tail motion-blur feel, drifting gust sheets and cam parallax.
  - `region=(x0,y0,x1,y1)` restricts the streaks, e.g. to keep them out from under an umbrella.
  - Costs: bg ≈12 ms, mid ≈10 ms, fg ≈7 ms.
- `draw_splashes(ctx, t, weather, region(rect or polygon), seed, density, horizon_y, ground_y_fn, color, scale, rings, crowns)`:
  - Crown splashes plus elliptical ripple rings, with perspective-scaled sizes.
  - Costs 15–35 ms.
- `draw_awning_drips(ctx, t, weather, x0, x1, y, ground_y=None, spacing=38, seed, size=4, streams=True, edge_fn=None)`: beads form and fall, with thin continuous streams in heavy rain.
- `draw_drips_from(ctx, t, points[(x,y[,ground_y])], ground_y, period=(.7,1.6), size, gravity=2600, color, weather, seed, alpha, splash, sparkle, rate)`:
  - Generic drips: a bead swells, detaches, falls and splashes.
  - After rain they slow down and sparkle.
- `draw_canopy_rain(ctx, t, center, radius, weather, ry=.28r, ribs=8, ground_y, rim_angles, seed, intensity, color, hits, tilt)`:
  - For s04: sheets of water from each scallop between rib tips, heavy rib-tip drips, ground splashes, bouncing drops and spray mist on the canopy top.
  - `center` = the rim-ellipse centre. See `world_fx_canopy_rain.png`.
- `draw_glints(ctx, t, weather, region, n=40, seed, size=14, horizon_y, color, strength)`: twinkling 4-point sun glints that fade in with `palette['glints']`. `draw_glint(ctx, x, y, size, alpha, color, rot)` draws a single glint.
- `draw_puddle(ctx, x, y, w, h, t, reflect_fn=None, weather, seed, ripples, sky, alpha)`:
  - An irregular puddle with sky tint, rain ripples and a rim.
  - `reflect_fn(ctx)` runs clipped, with the mirror about y already applied. Draw normally.

## Costs (cached frame, excluding post)
- **Street:** about 125 ms. A full frame with all rain, splashes and glints is 135–190 ms.
  - The first use per process and weather key (0 / 0.5 / 1; intermediate weathers crossfade two keys) builds the plates in about 10–14 s. Each key holds about 90 MB of plates.
- **Shopfront:** about 45 ms with a static cam. A full frame with character reflection, rain and fg is 105–165 ms.
  - The shop layers are rendered at the exact cam and weather (0.01 steps), with an LRU of 4.
  - When the cam or weather changes per frame, rebuilds cost about 90–150 ms, which includes the sky.
- **Close-ups with dof:** about 30–110 ms.
- **Post:** `core.bloom` + `vignette` + `grain` costs about 0.5–1.1 s and dominates the frame time.
- The first call in a process also builds the cloud texture (about 0.4 s).

## Known limits
- The street "crane" is a 2D pan/zoom over cached plates, without true camera-height parallax; only the sky, the near poles and wires, and the rain move differently.
- Passers-by are simple silhouettes.
- The shopfront's left neighbour is only a sliver, and the 2F above the awning is plain (only seen on tilt-up).
