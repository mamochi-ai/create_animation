"""Synthesize every dialogue line with Open JTalk (HTS voice: tohoku-f01)
and extract exact per-phoneme timings for lip sync.

HTS phoneme durations depend only on each full-context label, so the
length of synth(labels[:k]) is exactly the onset of phoneme k."""
import json, pathlib, numpy as np, scipy.io.wavfile as wav, pyopenjtalk
from pyopenjtalk.htsengine import HTSEngine

import os
ROOT = pathlib.Path(os.environ.get("FILM_DIR", pathlib.Path(__file__).resolve().parents[2])).resolve()
VOICE_DIR = pathlib.Path("/opt/voices/htsvoice-tohoku-f01")
OUT = ROOT / "assets" / "voice"
LINES = json.loads((ROOT / "story" / "dialogue.json").read_text())

_cast_file = ROOT / "story" / "cast.json"
# character -> (half_tone, speed[, voice]); voice "m001" = Open JTalk's bundled male voice
CAST = json.loads(_cast_file.read_text()) if _cast_file.exists() else {
    "haru": [1.5, 1.0],
    "kira": [8.0, 1.08],
}
_engines = {}
EXTRA_VOICES = {  # non-tohoku voices usable as a cast "style"
    "m001": "/opt/voices/hts_voice_nitech_jp_atr503_m001-1.05/nitech_jp_atr503_m001.htsvoice",  # male
    "mei": pyopenjtalk.DEFAULT_HTS_VOICE.decode() if isinstance(pyopenjtalk.DEFAULT_HTS_VOICE, bytes) else str(pyopenjtalk.DEFAULT_HTS_VOICE),
}

def engine(style):
    if style not in _engines:
        path = EXTRA_VOICES.get(style) or VOICE_DIR / f"tohoku-f01-{style}.htsvoice"
        _engines[style] = HTSEngine(str(path).encode())
    return _engines[style]

def synth(style, labels, half_tone, speed):
    e = engine(style)
    e.set_speed(speed); e.add_half_tone(half_tone)
    x = e.synthesize(labels)
    e.refresh()
    return x, e.get_sampling_frequency()

VOWEL = {"a": "a", "i": "i", "u": "u", "e": "e", "o": "o",
         "A": "i", "I": "i", "U": "u", "E": "e", "O": "o", "N": "n", "cl": "x", "pau": "x", "sil": "x"}

def main():
    meta = []
    OUT.mkdir(parents=True, exist_ok=True)
    for ln in LINES:
        c = CAST[ln["who"]]
        ht, sp = c[0], c[1]
        ht += ln.get("pitch", 0.0); sp *= ln.get("speed", 1.0)
        if len(c) > 2:
            ln = {**ln, "style": c[2]}
        labels = pyopenjtalk.extract_fullcontext(ln["text"])
        x, sr = synth(ln["style"], labels, ht, sp)
        # exact phoneme onsets
        phones = []
        for k, lab in enumerate(labels):
            p = lab.split("-")[1].split("+")[0]
            t0 = len(synth(ln["style"], labels[:k], ht, sp)[0]) / sr if k else 0.0
            phones.append([p, round(t0, 4)])
        segs = []
        for k, (p, t0) in enumerate(phones):
            t1 = phones[k + 1][1] if k + 1 < len(phones) else len(x) / sr
            segs.append({"p": p, "v": VOWEL.get(p, None), "t0": t0, "t1": round(t1, 4)})
        # trim leading/trailing silence
        lead = segs[0]["t1"] if segs[0]["p"] == "sil" else 0.0
        tail = segs[-1]["t0"] if segs[-1]["p"] == "sil" else len(x) / sr
        lead = max(0.0, lead - 0.03); tail = min(len(x) / sr, tail + 0.08)
        y = x[int(lead * sr):int(tail * sr)]
        y = y / 32768.0
        fade = int(0.01 * sr); y[:fade] *= np.linspace(0, 1, fade); y[-fade:] *= np.linspace(1, 0, fade)
        for s in segs:
            s["t0"] = round(max(0, s["t0"] - lead), 4); s["t1"] = round(max(0, min(tail, s["t1"]) - lead), 4)
        segs = [s for s in segs if s["t1"] > s["t0"]]
        name = ln["id"]
        wav.write(OUT / f"{name}.wav", sr, (np.clip(y, -1, 1) * 32000).astype(np.int16))
        meta.append({**ln, "file": f"assets/voice/{name}.wav", "dur": round(len(y) / sr, 3), "phones": segs})
        print(f"{name:10s} {ln['who']:5s} {len(y)/sr:5.2f}s  {ln['text']}")
    (OUT / "voices.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
