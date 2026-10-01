# あめふりロボ — "Rainy-Day Robo"

A 60-second 2D animated short (1920×1080, 24 fps), second film of this repo. Bright, funny, warm —
a deliberate contrast to the night-time 「ほしのおとしもの」. Everything procedurally generated.

Logline: On a rainy afternoon a girl who forgot her umbrella is rescued by an old delivery robot —
whose umbrella is hilariously tiny, then heroically huge, and who turns out to have zero waterproofing.
She ends up protecting him back. The rain stops; a rainbow.

## Characters

**ミナ (Mina)** — 8-year-old schoolgirl. Brown hair (#5A3A2A) in two short pigtails tied with yellow ribbons,
side-swept fringe, big warm-brown eyes with catch-lights, rosy cheeks. Classic yellow school hat (黄色い帽子,
#F7C934, round brim), white short-sleeve blouse with a round collar, navy pinafore skirt (#2E3F73), white socks,
red rain-less shoes (#D9483B), and a big red randoseru backpack (#C8323A, metal clasp, side reflector). Slightly
damp — hair strands stuck, a few water droplets. Expressive, laughs easily, gets indignant ("もう！") when someone
neglects themselves.

**ロボ (Robo)** — an old neighbourhood delivery robot "No.7", about 0.85× Mina's height. Rounded capsule body,
mint paint (#8FD3C1, shadow #5FA79A) with a cream belly panel (#F3EBD3), rust spots (#B8693A), rivets, a stencilled
「7」. Head = rounded-rectangle box with a dark glass screen (#1C2A33) on which cyan LED pixel eyes (#6FF2FF) form
expressions (dots, ^ ^, > <, ? ?, X X glitch, loading spinner, hearts, warm orange when moved). One antenna with a
red bulb that blinks. Accordion arms with 3-finger clamp hands. Moves on two small rubber treads (slight wobble,
squeak). A hatch on top of the head pops out a comically TINY umbrella; its back panel unfolds into a GIANT
umbrella canopy (teal-and-cream striped, 8 telescoping ribs). Not waterproof at all: when soaked, sparks, screen
static, voice glitches. Voice: male, slow, monotone, beeps.

**こねこ (kitten)** — tiny calico kitten sheltering beside Mina; mews at key beats.

## Palette
- Rain day: sky #8C9BAA→#B7C2CC, street #4E5A66 (wet, mirror reflections), shop fronts muted teal/brown/cream,
  rain streaks #D8E3EC at 30–60% alpha, warm shop lights #FFC777, vending machine glow #CFF3FF.
- After rain (s06): sky #7EC3F0→#D8F0FF, sun-warm #FFE6A8, wet highlights sparkle, rainbow, puddles reflect blue.
- Accents: Mina's yellow hat & red randoseru must pop against the grey rain.

## Shot list (global seconds) — see timeline.json for dialogue & SFX cue times
| Scene | Time | Content |
|---|---|---|
| s01 | 0–6 | Rainy 商店街 (shopping street) from a high angle; camera cranes down through falling rain; puddles ripple; umbrellas of passers-by (simple shapes) cross. Title 「あめふりロボ」 bounces in playfully. |
| s02 | 6–15.5 | Mina under a dripping shop awning (a closed dagashi shop) with the kitten. **Mina (7.6)「あーあ……かさ、わすれちゃった。」** A drip lands on her hat (9.8). Squeaky robot rolls in from the right (10.6), stops (11.3) and turns its screen to her. **Robo (11.5)「ピピッ。ぬれている、こどもを、はっけん。」** (scan-line beam over her). |
| s03 | 15.5–27 | **Mina (16.0)「え……ロボット？」** Robo: **(18.6)「かさ、てんかい。」** — head hatch pops a TINY umbrella (20.2) barely covering one antenna. Beat. **Mina (21.0)「ちっちゃ！」** — she bursts out laughing. Robo's screen shows ? ?, then a loading spinner. **Robo (23.4)「……さいけいさん。もんだい、なし。」** |
| s04 | 27–38 | Transformation montage: back panel splits, ribs telescope out, clamps lock (anime mecha-transformation staging, dramatic low angles, speed lines, light glints) — canopy blooms open (30.8) over Mina and the kitten. Rain drums on it. **Mina (32.0)「すごーい！おっきい！」** Then pull back: Robo itself stands outside the canopy's cover, getting soaked; sparks crackle from its seams (34.6). **Mina (35.4)「ロボさん、ぬれちゃってるよ！」** |
| s05 | 38–49 | Close on Robo: screen static, eyes X X, voice stutters. **Robo (38.6)「ぼうすいきのう……ゼロ、です。」** Mina puffs her cheeks: **(42.6)「もう！」** — swings off her red randoseru and holds it over Robo's head like a roof (43.5), standing close. Robo's screen clears; eyes soften, glow warm orange. **Robo (45.6)「……あたたかい。」** Kitten nuzzles Robo's tread. |
| s06 | 49–60 | The rain thins and stops (49.5); clouds part, sunbeams (51), the wet street sparkles, a rainbow arcs over the shopping street. Robo folds the canopy. **Mina (53.2)「はれた！ありがとう、ロボさん！」** Robo's screen shows a pixel heart. **Robo (56.2)「どういたしまして。ピピッ。」** Kitten mews. End card 「おわり」 (58). |

## Technical contract (same as film 1)
- Run everything with `FILM_DIR=films/ame_robo` set (core/render/compose/audio read story/ and build/ from there).
- Scene modules `films/ame_robo/scenes/sNN.py` expose `render(t)` (local seconds) → uint8 (1080,1920,3). No subtitles.
- Film-specific shared art lives in `films/ame_robo/lib/` (import as `lib.characters`, `lib.world`).
- Shared repo helpers in `src/common/core.py` (easing, colour, glow, bloom, text_layer, mouth/blink lip sync).
- Render: `FILM_DIR=films/ame_robo python3 src/render.py s03 --sheet 12` etc.
