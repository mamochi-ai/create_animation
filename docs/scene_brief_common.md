# Scene animator brief (common to all scenes)

You are animating ONE scene of the 60-second 2D anime short 「ほしのおとしもの」 in /home/user/create_animation.
Read first: story/script.md (story, characters, shot list), story/timeline.json (scene times, dialogue start times, SFX cue times),
assets/voice/voices.json (line durations + phoneme timings), docs/characters_api.md, docs/world_api.md, src/common/core.py.
Look at build/preview/model_sheet_*.png and build/preview/world_*.png to see the assets.

## Contract
- Write `src/scenes/<id>.py` exposing `render(t: float) -> np.ndarray` (1080,1920,3) uint8, where t = LOCAL seconds from scene start.
- Global time = scene start + t. Use global time for `core.mouth(who, tg)` lip sync (returns openness, vowel, talking) and for `core.blink`.
- Deterministic (seeded; no global mutable state that depends on call order) so frames render in parallel processes.
- Do NOT draw subtitles (compositor adds them). Do not edit core.py / characters.py / world.py / other scenes. If you need a helper, put it in your own scene file. If a shared library has a real bug, work around it locally and mention it in your report.
- Do NOT git commit.
- Target ≤ 1.5 s per frame including post (bloom/vignette/grain). Cache static layers at module level.

## Quality bar (this is a showcase piece — make it feel like real anime direction)
- Cinematic camera: slow push-ins, pans, parallax, subtle handheld drift where apt; cuts within the scene are allowed (e.g. a close-up), keep them motivated.
- Animation principles: anticipation, follow-through, overlapping action, squash & stretch on Kira, eased timing (core easing helpers), idle breathing, blinking; never static for more than a beat.
- Characters speaking must lip-sync via core.mouth; listeners react (blink, small head tilts).
- Lighting: Kira and lanterns cast warm light (draw_light_pool, rim colour on Haru), bloom via core.add_glow / core.bloom, vignette + grain.
- Keep characters readable against the background (rim light, value contrast, scale).

## Workflow
1. `python3 src/render.py <id> --sheet 12` → build/preview/<id>_sheet.png (12 stills). `--at 3.5` for one still. LOOK at them with the Read tool; iterate several times.
2. When happy, render all frames + a preview video with audio: `python3 src/render.py <id> --video --jobs 2 --force` (other animators share the 4-core machine, so use --jobs 2). Check a few frames around key moments (dialogue, action beats) again.
3. Report concisely: what the scene shows beat by beat, render time per frame, and any issues/limitations.
