# Character API (src/common/characters.py)

`from common.characters import draw_haru, draw_kira, draw_kira_glow, kira_glow_layer, draw_lantern, draw_lantern_glow, draw_sparkle, line_weight`

All draw functions honour the current cairo transform (cameras OK) and return anchors in that space.
Preview: build/preview/model_sheet_haru.png, model_sheet_kira.png (`python3 src/common/char_sheet.py`).

## draw_haru(ctx, x, y, scale=1.0, **pose) -> anchors
(x,y) = ground point between feet; scale 1.0 ≈ 520 px tall (~4.4 heads).
- view: front | three_quarter_left | three_quarter_right | side_left | side_right | back — or `yaw` (rad; 0 front, +π/2 faces screen-right, −π/2 screen-left, π back).
- expr: neutral happy surprised determined sad gentle sleepy worried laugh. Overrides: eye_open, blush, brow_tilt, smile.
- mouth_open 0..1 + vowel 'aiueonx' (feed core.mouth('haru', t_global)), blink 0..1, look=(dx,dy) −1..1.
- head_tilt (rad), head_turn (extra yaw for head), head_pitch (−1..1, + = looks up), body_lean (rad), crouch 0..1, bounce (+squash/−stretch).
- arms: rest chin_on_hands cup_hands hold_lantern raise_lantern wave reach_up run hands_to_mouth hips; arm_phase animates wave/run; hand='screen_left'|'screen_right'|'near'|'far'; hand_l/hand_r=(lateral, forward, v) wrist override in body space (v from hip ≈205px above feet; head centre v=−230).
- run_phase 0..1 full side run cycle (legs, bob, lean, trailing hair, sleeves); combine with arms='run' or 'cup_hands'. None = standing.
- wind (0/1/2), wind_dir (+1 → screen-right), t (seconds, breathing/hair sway), rim colour (default cool moonlight; '#FFC27A' near lantern; None off), light='left'|'right', alpha.
- lantern: False | True (in hand for hold_lantern/raise_lantern) | 'back' (on her back, for running); lantern_lit 0..1, lantern_kira 0..1, lantern_scale (0.95).
- held=callback(ctx, x, y, s) drawn at the cupped hands (between palms and front fingers for cup_hands). Kira at ≈0.4*s.
- crop='bust' skips legs (x,y still virtual feet point → use scale 1.5–2.5 with feet off-screen; draw sills over her).
Anchors: head, face, eyes(list), mouth, chin, top, neck, chest, hip, feet, hand_l, hand_r, hands, cup (cup_hands), lantern, lantern_body, lantern_radius (if lantern), facing (+1 right/−1 left/0), scale.

## draw_kira(ctx, x, y, scale=1.0, **pose) -> anchors
(x,y)=centre; scale 1.0 ≈ 120 px across.
expr: neutral happy sad hurt surprised sleepy; mouth_open, vowel, blink, look, rot, squash (−1..1), glow 0..1 (0.1 pale/grey/translucent … 1 blazing), pose: none rub_head wave cheer droop (+pose_phase, defaults to t), t, blush, alpha.
Anchors: center, glow_radius, eyes, mouth, top.

## draw_lantern(ctx, x, y, scale=1.0, lit=0.0, sway=0.0, t=0.0, kira=0.0, alpha=1.0) -> anchors
(x,y)=top of hanging loop; ~84 px wide at scale 1. lit 0..1 warm glow with rib shadows; kira 0..1 = golden star glowing inside. Anchors: hang, body_center, radius, top, bottom.

draw_sparkle(ctx, x, y, size, alpha, rot, color) — 4-point twinkle.

## Glow
- Cheap: draw_kira_glow(ctx, x, y, scale, glow, t) / draw_lantern_glow(ctx, hang_x, hang_y, scale, lit, sway, t) BEFORE the draw call.
- Real bloom: lay = kira_glow_layer(x, y, s, g, t, **pose); core.add_glow(frame, lay, radius=28*s, strength=1.2*g). Lantern: core.draw_layer(lambda c: draw_lantern(c, ...)) → add_glow. Finish with core.bloom().

Caveats: left views are mirrored right views; hands above the big head overlap hair in front view (use 3/4/side for raise_lantern); no 3/4-back view.
