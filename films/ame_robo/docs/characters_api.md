# Character API — あめふりロボ (films/ame_robo/lib/characters.py)

```python
import lib.characters as ch      # films/ame_robo and src/ on sys.path
from lib.characters import draw_mina, draw_robo, robo_glow_layer, draw_kitten, \
    draw_droplet, draw_drops_on, draw_rim_drips, draw_sparkle, ROBO_FACES
```
Every draw function respects the current cairo transform, so cameras are fine. Each one returns anchors in the caller's user space.
To rebuild the preview sheets, run `FILM_DIR=films/ame_robo python3 films/ame_robo/lib/char_sheet.py [mina|robo|kitten]`. This writes
`build/preview/model_sheet_{mina,robo,kitten}.png`.
On-screen cost: Mina at scale 1.8 takes about 25–31 ms. Robo with the full canopy, glitch and sparks takes about 24 ms. The kitten takes about 4 ms.

Common conventions:
- `view` is one of `front | three_quarter_left | three_quarter_right | side_left | side_right` (Mina also has `back`). You can pass `yaw` in radians instead: 0 is front, +π/2 faces screen-right.
- Left-facing views are mirrored right-facing views.
- `t` is the time in seconds and drives all secondary motion, so pass the scene's global or local time.
- `light='left'|'right'` sets the key-light side. `rim` is the rim colour (default cool `#D6EAF8`; `None` turns it off). `alpha` sets opacity.

## draw_mina(ctx, x, y, scale=1.0, **pose) -> anchors
(x, y) is the ground point between her feet. At scale 1 she is about 480 px tall including the hat (about 4.5 heads).

| kwarg | values |
|---|---|
| expr | neutral, happy, laugh (^^ eyes, wide mouth), sad, surprised, worried, pout (puffed cheeks + puff ticks + 怒りマーク, for 「もう！」), amazed (star catch-lights + sparkles), gentle |
| mouth_open, vowel | lip sync: `o, v, _ = core.mouth('mina', tg)` |
| blink, look=(dx,dy) | eyes; use `core.blink(t, seed)` for `blink` |
| eye_open, blush, brow_tilt, smile | override the expression preset |
| head_tilt, head_turn, head_pitch | radians / extra yaw / −1..1 (+ looks up) |
| body_lean, bounce (+squash/−stretch), crouch 0..1, tiptoe 0..1 | body |
| wet 0..1 | darker hair, hat and fabric; clumped pointed strands; droplets on the hat brim, fringe, pigtails and hem |
| wind 0..2, wind_dir, t | pigtails swing with overlap (the tip lags the root); skirt sway; breathing |
| arms | rest, hands_clasped, hands_to_mouth (giggle), point, wave, arms_up (cheer), hold_randoseru_overhead, shrug. `arm_phase` drives wave/cheer (default t*2.2). `hand='screen_left'|'screen_right'` picks the arm for point/wave; the default is the near hand in 3/4/side views and the screen-right hand in front view. |
| hand_l / hand_r | wrist override (l, f, v) in body space: l is +her left, f is forward, v is up-negative from the ground. Shoulders are at (±31, 0, −303) and the head centre at v=−380. |
| randoseru | 'back' (default) \| 'held' \| None. `arms='hold_randoseru_overhead'` implies 'held'. |
| hold_target=(x,y) | caller coords for the box centre. She auto-leans toward it (up to 0.22 rad), and her chibi arms may stretch up to 25 %. Beyond that the target is clamped (`anchors['hold_clamped']`). |
| hold_offset=(dx,dy) | px*scale offset from the default spot (above her hat). Use instead of `hold_target`. |
| hold_tilt (rad), hold_flap 0..1 | the lid swings open and hangs in front, showing the clasp |
| crop='bust' | skips legs and shoes; (x, y) is still the virtual feet point, so put it off-screen and use scale 2–3 |

Anchors: `head, face, eyes[], mouth, chin, hat_top, top, neck, chest, hip, feet, foot_l, foot_r, hand_l, hand_r, hands,`
`randoseru_center, randoseru_top` (roof top surface when held), `randoseru_span` (x0, x1 when held), `drop_points` (hat, brim, shoulders, top of the held box; pass to `draw_drops_on`), `facing, scale`.

s05 recipe: draw Robo first, then Mina about 150–180 px*scale beside him in `three_quarter_right` (or `side_right`) with
`arms='hold_randoseru_overhead', hold_target=(robo['head_top'][0], robo['head_top'][1] - 60*s)`. Her raised arms go behind her
head and only the gripping hands are drawn over the box, so the face stays clean.

## draw_robo(ctx, x, y, scale=1.0, **pose) -> anchors
(x, y) is the ground contact centre. At scale 1 he is about 410 px tall to the top of the antenna bulb, which is about 0.85× Mina.

| kwarg | values |
|---|---|
| face | `ROBO_FACES`: dots, happy (^ ^), blink, question (bobbing ? ?), loading (spinner, uses t), glitch (X X + noise, row tear, scanlines, uses t), warm (orange ^ ^ + pink blush pixels), heart (pulsing), surprised (O O), scan (sweeping line, uses t) |
| screen_glow 0..1 | LED brightness plus warm/cyan spill onto the head frame |
| speak 0..1 | pixel waveform mouth: `speak=core.mouth('robo', tg)[0]` |
| look=(dx,dy) | moves the LED eyes up to 2 cells |
| body_tilt, head_tilt (rad), bob (px, + down), squash | body motion; the upper body pivots above the treads |
| tread_phase | scrolls the lugs and turns the wheel spokes (≈ distance_px / 9) |
| antenna_blink | True (auto blink from t) \| False \| float brightness |
| arms | rest, wave, point (toward screen-right / facing), hold_handle (grip in front of the chest; `anchors['handle']`), salute, flail (uses t). `hand=` picks the arm as for Mina. hand_l/hand_r overrides; shoulders at (±88, 0, −186). |
| wet 0..1 | darker paint, streaks and drips running down, drops falling off, drops on the head top |
| sparks 0..1 | electric bolts from the neck seam, shoulders, ears and lower seams. Seeded by `int(t*16)` and `seed`, so they flicker each frame. |
| mini_umbrella 0..1 | 0–.2 hatch lid flips open; .12–.62 spring shoots up with overshoot; .55–1 a TINY red umbrella opens (radius = head width/8) over the antenna. Wobbles with t. Anchors `umbrella_top, umbrella_radius`. |
| canopy 0..1 | 0–.1 back panel doors open; .03–.3 three-section telescoping pole extends from his back over his head; .3–.7 the 8 ribs swing out and lengthen with a per-rib stagger (glints at the tips around .6–.8, "clamps lock"); .7–1 teal/cream fabric blooms with an overshoot bounce |
| canopy_offset=(dx,dy) | px*scale offset of the canopy apex. The default (470, 0) puts the canopy toward screen-right (his facing side). Its rim then ends right at Robo, so a girl about 300–450 px*scale to that side is covered and Robo stands outside in the rain (the gag). |
| canopy_tilt | rad |
| layer | 'all' \| 'body' \| 'canopy': draw the canopy and pole separately so rain can be layered between |
| glow_only | emissive parts only (LEDs, bulb, sparks) |

Canopy geometry at scale 1: diameter 984 px (2.4 × Robo's height), apex 765 px above the ground, rim drop 230. The rim ellipse is seen
slightly from below, so the underside stripes show beneath the front rim.
Anchors: `screen_center, screen_size, head, head_top, antenna, hatch, umbrella_top, umbrella_radius, hand_l, hand_r, hands,`
`handle, chest, belly, feet, tread_l, tread_r, canopy_apex, canopy_center, canopy_radius, canopy_span (x0, x1),`
`canopy_rim_points` (rib tips + scallop mids; pass to `draw_rim_drips`), `drop_points, facing, scale`.
Use `canopy_span` to mask rain behind and under the canopy.

## draw_kitten(ctx, x, y, scale=1.0, pose='sit', expr='neutral', blink=0, t=0, wet=0, facing=1, look=(0,0))
The kitten is a tiny calico: white with orange and black patches, and an orange tail with a black tip. At scale 1 it is about 70 px tall sitting. Use the same scale as the humans so it stays tiny next to them.
- pose: `sit`, `curl` (sleeping loaf), `nuzzle` (head rubbing, uses t; use it at Robo's tread in s05), `stand`.
- expr: `neutral`, `happy` (^ ^), `mew` (open mouth).
- facing: +1 faces right, −1 faces left.
- wet: spiky fur and droplets.
Anchors: `head, nose, mouth, tail_tip, top, feet`.

## Rain helpers
- `draw_droplet(ctx, x, y, r, alpha=1, stretch=1)` draws one shiny drop.
- `draw_drops_on(ctx, points, t, scale, seed=0, amount=1)` makes drops bead, slide and fall off. Use it with `anchors['drop_points']`.
- `draw_rim_drips(ctx, rim_points, t, scale, seed=0, amount=1, fall=420)` draws streams falling from the canopy rim. Use it with `canopy_rim_points`.

## Glow tips
- Robo's LEDs already have soft halos drawn in cairo. For real bloom, use
  `core.add_glow(frame, ch.robo_glow_layer(x, y, s, **same_pose), radius=14*s, strength=1.2)`. Pass the same face, t, view, bob, tilt and so on as the main draw. Then finish with `core.bloom(frame, 0.8)`.
- `robo_glow_layer` allocates a full-frame layer, which takes about 0.1–0.4 s in numpy conversion. Use it on close-ups, or reuse one layer per frame for several glow sources.
- Use `face='warm'` with `screen_glow=1` for 「……あたたかい。」 and add a warm `#FFB45E` tint. Use `face='glitch'` + `sparks` + `wet` for s05.
- The antenna bulb halo is built in. For a spark flash, add a short-lived global bloom or `add_glow` the glow layer, since it includes the sparks.

## Caveats
- Robo has no back view; `yaw` is clamped to side at most. The canopy is placed in screen space (offset from Robo), not in 3D, so it does not rotate with `view`.
- The held randoseru is drawn with its long axis along screen x, seen from below; it is not intended for back view.
- When targets are far, Mina's raised arms stretch (chibi cheat), so keep her within about 180 px*scale of the hold target.
- Left-facing views are mirrors: Mina's side-swept fringe and Robo's antenna/hatch flip sides. Robo's "7" and "No.7" are kept readable.
