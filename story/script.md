# ほしのおとしもの — "The Star That Fell"

A 60-second 2D animated short. 1920×1080, 24 fps. Everything (art, animation,
voices, music, sound) is generated procedurally from code in this repository.

Logline: On a summer night a small star falls into a sleepy hillside town.
A girl named Haru has until dawn to carry it back to the sky — and the only
way up is the warm air of her own paper lantern.

## Characters

**ハル (Haru)** — 10-year-old girl. Short black bob with a blunt fringe and one
small cowlick; a red hair clip (star-shaped, foreshadowing). Big dark eyes
with two white catch-lights, rosy cheeks. Wears an indigo summer yukata-style
top with a pale-blue wave pattern, yellow obi, and straw sandals. Carries a
round white paper lantern (chōchin) with a red band top and bottom and a
bamboo handle. Energetic, brave, a little stubborn.

**キラ (Kira)** — a fallen star about the size of Haru's two hands. Soft,
rounded five-pointed star shape (points are plump, like a cookie), warm
pale-gold body (#FFE9A8 core → #FFC94D rim) that glows. Huge oval eyes,
tiny mouth. Floats and bobs; trail of sparkles when it moves. Its glow is
its life: bright in s03, fading/flickering in s04–s05, blazing in s06.

## Palette (shared)

- Night sky: #0B1030 (top) → #1B2350 → #2E3A78 (horizon)
- Pre-dawn: horizon #3D4C8F → #7C6FA8 → #E7A6A0
- Dawn: #F6C58F horizon, #F7E3C4 glow, sky #7FA7D9 top
- Town silhouettes: #141A3A, roofs #1E2650 with rim light #3B4A8C
- Warm windows / lantern: #FFB85C, #FFD9A0, glow #FF9A3D
- Haru: hair #1A1426, yukata #2B3A7A, pattern #9EC5FF, obi #F2C14E, skin #FCE3D2, cheeks #F59CA0
- Kira: #FFF6D6 / #FFE9A8 / #FFC94D, eyes #3A2A1A

## Shot list (global times, seconds)

| Scene | Time | Content |
|---|---|---|
| s01 | 0–7 | Deep starry sky, slow camera tilt down to a hillside town at night (layered roofs, lit windows, a shrine and a giant camphor tree on the hilltop). At 2.2s a shooting star arcs across the sky, sputters, and dives toward the town. Title 「ほしのおとしもの」 fades in ~2.5–6s. |
| s02 | 7–13 | Haru, chin on hands, at her round attic window; unlit paper lantern on the sill. A streak of light flashes past (7.6), a soft glow blooms in the garden below (8.0). Haru's eyes pop open. **Haru (8.6):「……いまの、なに？」** She leaps up and runs off-frame (~10–12). |
| s03 | 13–25 | Night garden: tall grass, hydrangeas, fireflies. A small crater of light. Kira sits up rubbing its head. Haru crouches in. **Kira (13.8):「いたたた……おっこちちゃった。」** **Haru (16.5):「わっ！ほしが、しゃべった！」** Kira floats up to eye level, sad. **Kira (19.1):「ぼく、キラ。よあけまでに、そらへかえらないと、きえちゃうんだ。」** Kira's glow flickers. |
| s04 | 25–37 | Haru, determined. **Haru (25.3):「じゃあ、いちばんたかいところまで、いっしょにいこう！」** Dynamic side-scrolling run up stone shrine steps through red torii gates, parallax, fireflies streaming; Kira rides cupped in her hands, lantern swinging on her back. Kira dims. **Kira (31.1):「ハル……ぼく、ひかりが、よわくなってきた。」** **Haru (35.6):「もうすこし！」** |
| s05 | 37–48 | Hilltop under the giant camphor tree; horizon starts to turn faintly violet. Kira tries to fly, falls into Haru's hands, dim. Haru looks at her lantern. **Haru (38.6):「そうだ……！」** She lights the lantern (40.2, warm bloom), places Kira inside; the lantern glows gold and begins to lift. **Kira (42.6):「あったかい……ハルのちょうちん、あったかいよ。」** Haru lets go. **Haru (46.3):「いって、キラ！」** |
| s06 | 48–55 | Wide shot: lantern rising into the vast pre-dawn sky, Haru tiny on the hilltop. **Kira (49.4):「ありがとう、ハル！ずっと、みてるからね！」** At 52.8 the lantern bursts into a shower of golden sparks and a streak of light shoots up and becomes a bright star. |
| s07 | 55–60 | Dawn over the town. Haru under the tree, looking up. One star still visible, twinkles twice. **Haru (56.0):「……またね、キラ。」** She waves; end card 「おわり」 fades in ~58. |

## Technical contract

- Every scene module `src/scenes/sNN.py` exposes `render(t: float) -> numpy uint8 array (1080,1920,3)`
  where `t` is **local** time in seconds from scene start. Rendering must be deterministic
  (seed RNGs) so frames can be rendered in parallel.
- Scenes must not draw subtitles — subtitles are composited globally.
- Lip sync: use `common.core.mouth(line_id, t_global)` → openness 0..1 and vowel.
