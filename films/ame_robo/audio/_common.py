"""Path bootstrap for the あめふりロボ audio scripts.

Makes `import dsp` (the shared toolkit in src/audio/) work and points it at this
film: FILM_DIR defaults to films/ame_robo when it is not already set.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILM = HERE.parent
REPO = FILM.parents[1]
os.environ.setdefault("FILM_DIR", str(FILM))
for p in (str(REPO / "src" / "audio"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
