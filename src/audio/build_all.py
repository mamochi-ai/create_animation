"""Rebuild the whole soundtrack: music -> sfx -> voice_fx -> mix (-> analyze).

    python3 src/audio/build_all.py            # outputs in build/audio/
Requires numpy, scipy, pyloudnorm (numba optional but fast; matplotlib for PNGs).
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for step in ("music", "sfx", "voice_fx", "mix", "analyze"):
    t = time.time()
    print(f"=== {step}")
    try:
        runpy.run_path(str(HERE / f"{step}.py"), run_name="__main__")
    except ImportError as e:
        if step != "analyze":
            raise
        print(f"  (analysis skipped: {e})")
    print(f"    {time.time() - t:.1f}s")
