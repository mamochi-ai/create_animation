# Scene animator brief (common to all scenes)

You are animating ONE scene of the 60-second 2D anime short 「あめふりロボ」 (film 2) in /home/user/create_animation (film directory films/ame_robo). ALWAYS run commands with the env var FILM_DIR=films/ame_robo from the repo root.
Read first (paths relative to films/ame_robo/): story/script.md (story, characters, shot list), story/timeline.json (scene times, dialogue start times, SFX cue times),
assets/voice/voices.json (line durations + phoneme timings), docs/characters_api.md, docs/world_api.md, and the repo's src/common/core.py.
Look at films/ame_robo/build/preview/model_sheet_*.png and world_*.png to see the assets. Import film libs as `from lib.characters import ...`, `from lib import world`; core as `from common.core import ...`.

## Contract
- Write `films/ame_robo/scenes/<id>.py` exposing `render(t: float) -> np.ndarray` (1080,1920,3) uint8, where t = LOCAL seconds from scene start.
- Global time = scene start + t. Use global time for `core.mouth(who, tg)` lip sync (returns openness, vowel, talking) and for `core.blink`.
- Deterministic (seeded; no global mutable state that depends on call order) so frames render in parallel processes.
- Do NOT draw subtitles (compositor adds them). Do not edit src/common/core.py, films/ame_robo/lib/*, or other scenes. If you need a helper, put it in your own scene file. If a shared library has a real bug, work around it locally and mention it in your report.
- Do NOT git commit.
- Target ≤ 1.5 s per frame including post (bloom/vignette/grain). Cache static layers at module level.

## Quality bar (this is a showcase piece — make it feel like real anime direction)
- Cinematic camera: slow push-ins, pans, parallax, subtle handheld drift where apt; cuts within the scene are allowed (e.g. a close-up), keep them motivated.
- Animation principles: anticipation, follow-through, overlapping action, squash & stretch (Robo bounces on its treads, Mina is springy), eased timing (core easing helpers), idle breathing, blinking; never static for more than a beat.
- Characters speaking must lip-sync via core.mouth; listeners react (blink, small head tilts).
- Lighting & weather: rain layers in front of AND behind characters, wet sheen, reflections in puddles; warm shop light accents; bloom via core.add_glow / a fast bloom, vignette + grain.
- Comedy timing matters in this film: holds, takes, snappy reactions, smear frames.
- Performance tip from film 1: core.bloom's small blur runs at full res (slow); do your own half-res bloom and cache the vignette mask.
- Keep characters readable against the background (rim light, value contrast, scale).

## Workflow
1. `FILM_DIR=films/ame_robo python3 src/render.py <id> --sheet 12` → build/preview/<id>_sheet.png (12 stills). `--at 3.5` for one still. Outputs land in films/ame_robo/build/preview/. LOOK at them with the Read tool; iterate several times.
2. When happy, render all frames + a preview video with audio: `FILM_DIR=films/ame_robo python3 src/render.py <id> --video --jobs 2 --force` (other animators share the 4-core machine, so use --jobs 2). Check a few frames around key moments (dialogue, action beats) again.
3. Report concisely: what the scene shows beat by beat, render time per frame, and any issues/limitations.

- Scratch files: use your own subfolder of the scratchpad (other animators share it).
