"""Rebuild the whole あめふりロボ soundtrack: music -> sfx -> voice_fx -> mix -> analyze.

    python3 films/ame_robo/audio/build_all.py       # (FILM_DIR defaults to films/ame_robo)
    FILM_DIR=films/ame_robo python3 films/ame_robo/audio/build_all.py

Outputs: films/ame_robo/build/audio/{music,sfx,dialogue,mix}.wav (+ spectrogram PNGs).
Uses the shared DSP toolkit src/audio/dsp.py and analyzer src/audio/analyze.py.
"""
import runpy
import sys
import time

import _common  # sets FILM_DIR and sys.path before dsp is imported

for step in ("music", "sfx", "voice_fx", "mix", "analyze"):
    t = time.time()
    print(f"=== {step}")
    path = (_common.REPO / "src" / "audio" / "analyze.py") if step == "analyze" else (_common.HERE / f"{step}.py")
    sys.argv = [str(path)]
    try:
        runpy.run_path(str(path), run_name="__main__")
    except ImportError as e:
        if step != "analyze":
            raise
        print(f"  (analysis skipped: {e})")
    print(f"    {time.time() - t:.1f}s")
