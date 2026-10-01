# Procedural 2D anime shorts

Two 60-second 2D anime shorts with Japanese voices, made entirely from code in this repo:

| Film | Video | Mood |
|---|---|---|
| ほしのおとしもの — The Star That Fell | [`output/hoshi_no_otoshimono_web.mp4`](output/hoshi_no_otoshimono_web.mp4) | a night of wonder, a lantern, and a farewell |
| あめふりロボ — Rainy-Day Robo | [`output/amefuri_robo_web.mp4`](output/amefuri_robo_web.mp4) | rainy-day comedy with a heart, ending in a rainbow |

---

# Film 1: ほしのおとしもの — The Star That Fell

A 60-second 2D animated short with a Japanese voice cast, generated end to end from code in this repo.
The art, character rigs, animation, voices, lip sync, score, sound effects and mix are all made programmatically.
There is no stock footage, no audio samples and no image-generation model.

▶ **Film:** [`output/hoshi_no_otoshimono_web.mp4`](output/hoshi_no_otoshimono_web.mp4) — 1920×1080, 24 fps, 60.0 s, AAC stereo, with Japanese subtitles burned in.

## Story
On a summer night a small star named Kira falls into a sleepy hillside town. A girl named Haru has until dawn to carry it back to the sky.
She climbs the shrine steps to the hilltop, and the only way up is the warm air of her own paper lantern.

| Scene | Time | Content |
|---|---|---|
| s01 | 0–7 s | Tilt down from the Milky Way onto the town; a shooting star and the title |
| s02 | 7–13 s | Haru at the attic window: 「……いまの、なに？」 |
| s03 | 13–25 s | Meeting Kira in the garden |
| s04 | 25–37 s | Running up the shrine steps through the torii as Kira fades |
| s05 | 37–48 s | Hilltop: Kira goes into the lantern and it lights up |
| s06 | 48–55 s | The lantern rises, bursts, and becomes a new star |
| s07 | 55–60 s | Dawn: 「……またね、キラ。」 |

## How it's made
- **Voices:** Open JTalk (pyopenjtalk) with the tohoku-f01 HTS voices in several emotional styles, pitch-shifted per character.
  Phoneme timings are exact: HTS durations depend only on each label, so the length of `synth(labels[:k])` gives each phoneme's onset.
  These timings drive vowel-shaped lip sync (`src/audio/make_voices.py`, `core.mouth`).
- **Graphics:** pycairo vector drawing plus numpy/scipy post effects (bloom, glow, vignette, grain).
  - `src/common/characters.py`: rigged characters (Haru, Kira, the lantern) with views, expressions, a run cycle and arm poses.
  - `src/common/world.py`: environments (sky, town, garden, shrine steps, hilltop, attic).
  - `src/scenes/sNN.py`: one module per scene, each exposing `render(t)`.
- **Audio:** everything is synthesized in `src/audio/`:
  - the score: a yo-nuki pentatonic motif, with per-scene tempi aligned to the cuts;
  - the sound effects;
  - the voice processing, including Kira's "star" shimmer;
  - the mix, with sidechain ducking, normalized to −16 LUFS.
- **Assembly:** `src/compose.py` adds the dissolves, fades and Japanese subtitles (Zen Maru Gothic), then encodes the video with ffmpeg.

## Rebuild
```bash
pip install numpy scipy pillow pycairo pyopenjtalk imageio-ffmpeg pyloudnorm
git clone https://github.com/icn-lab/htsvoice-tohoku-f01 /opt/voices/htsvoice-tohoku-f01
# film 2 also needs the nitech m001 male voice in /opt/voices/hts_voice_nitech_jp_atr503_m001-1.05/
python3 src/audio/make_voices.py          # dialogue + phoneme timings
python3 src/audio/build_all.py            # music, sfx, voice fx, mix -> build/audio/mix.wav
for s in s01 s02 s03 s04 s05 s06 s07; do python3 src/render.py $s; done
python3 src/compose.py                    # -> output/hoshi_no_otoshimono.mp4
```

Credits: voice models are the tohoku-f01 HTS voices (icn-lab); fonts are Zen Maru Gothic and Klee One (SIL OFL).
Design notes are in `story/` and `docs/`.

---

# Film 2: あめふりロボ — Rainy-Day Robo  (`films/ame_robo/`)

Mina, a schoolgirl, forgets her umbrella. Robo No.7 is an old delivery robot who comes to her rescue.
He first pops out a ridiculously tiny umbrella, then transforms heroically into a giant one, and then turns out to have zero waterproofing.
Mina shelters him under her randoseru, the rain stops, and a rainbow appears.

- Cast: Mina uses the tohoku-f01 voice in its happy, sad, neutral and angry styles. Robo uses the nitech m001 male HTS voice, processed into a retro robot voice with glitches.
- Film-specific code:
  - `films/ame_robo/lib/`: characters (Mina, Robo with the tiny umbrella and canopy transformation, the kitten) and the world (the rainy 商店街, rain, reflections, the sky clearing, the rainbow).
  - `films/ame_robo/scenes/`: s01–s06.
  - `films/ame_robo/audio/`: score, SFX, voice FX and mix.
- The shared pipeline selects a film with `FILM_DIR`:

```bash
export FILM_DIR=films/ame_robo
python3 src/audio/make_voices.py
python3 films/ame_robo/audio/build_all.py
for s in s01 s02 s03 s04 s05 s06; do python3 src/render.py $s; done
python3 src/compose.py                    # -> output/amefuri_robo.mp4
```
