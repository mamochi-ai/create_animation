"""Dialogue processing for 「あめふりロボ」 -> build/audio/dialogue.wav (48 kHz stereo, 60.000 s).

Each synthesized line (assets/voice/*.wav, untouched) is processed and placed at its
start time from story/timeline.json.

Mina : HP 90 Hz, de-ess, de-buzz (LP 9 kHz + soft 7 kHz shelf cut), warmth (+2 dB 250 Hz),
       a little presence, 2.5:1 compression, location reverb: shop awning (s02-s03),
       under the umbrella canopy (s04-s05), open sunny street (s06).
Robo : charming retro robot that stays intelligible -
       small-speaker band EQ (300 Hz-5 kHz, 1.6 kHz honk), metallic feedback comb tuned to A4,
       30% 50 Hz ring/AM modulation, sample-rate (16 kHz hold) + 9-bit reduction, then a
       parallel blend with the cleaned dry voice; tuned beeps on the 「ピピッ」 syllables
       (C6/F6 in F major for l02, D6/G6 in G major for l13).
       l09 「ぼうすいきのう……ゼロ、です。」 'glitch' mode: heavier crush, random pitch wobble,
       short dropouts in consonants, syllable stutters placed in the pauses (の-う-う /
       ゼ-ゼ-ゼロ / ロ-ロ) and a tape-stop power-down on the final 「す」.
       l11 「……あたたかい。」 'warm' mode: wide band, no ring-mod/crush, light comb, warmth,
       gentle chorus and a longer, softer reverb (Robo's heart).
"""
import _common  # noqa: F401
import numpy as np
import scipy.signal as ss

from dsp import (SR, N, BUILD, ROOT, Bus, tvec, fade, lp, hp, bp, eq, compress, deess, make_ir, reverb,
                 read_wav, write_wav, report, db, undb, lufs, load_timeline, load_voices, hz, env_adsr)

RNG = np.random.default_rng(717)
LINE_LUFS = -20.0
LINE_TRIM = {"l10_mina": -1.5, "l05_mina": -0.5, "l11_robo": -0.5}
BEEPS = {"l02_robo": [(0.125, "C6"), (0.255, "F6")], "l13_robo": [(1.65, "D6"), (1.78, "G6")]}


def slow_noise(n, fc):
    x = lp(RNG.standard_normal(n), fc, 2)
    return x / (np.std(x) + 1e-12)


def mod_delay(x, d):
    """Read x through a variable delay d (seconds, per-sample array)."""
    n = np.arange(len(x), dtype=float)
    return np.interp(n - d * SR, n, x, left=0.0)


def comb(x, f, g):
    D = int(round(SR / f))
    a = np.zeros(D + 1)
    a[0], a[D] = 1.0, -g
    return ss.lfilter([1.0 - g], a, x)


def crush(x, bits=9, hold=3):
    pk = np.max(np.abs(x)) + 1e-9
    y = np.repeat(x[::hold], hold)[:len(x)] / pk
    q = 2 ** (bits - 1)
    return np.round(y * q) / q * pk


def cleanup(x):
    x = hp(x, 90, 2)
    x = deess(x, 6000, thresh_db=-34, ratio=3)
    x = lp(x, 9000, 4)
    x = eq(x, "highshelf", 7000, -3)
    return x


def beep(note, dur=0.06, amp=1.0):
    n = int((dur + 0.01) * SR)
    t = tvec(n)
    f = hz(note)
    y = np.sin(2 * np.pi * f * t) + 0.12 * np.sign(np.sin(2 * np.pi * f * t))
    return amp * lp(y, 6000) * env_adsr(n, 0.003, 0, 1, 0.012, hold=dur - 0.003)


# ------------------------------------------------------------------ Mina
def mina(x):
    x = cleanup(x)
    x = eq(x, "peak", 250, 2.0, 0.8)
    x = eq(x, "peak", 3000, 1.5, 1.0)
    x = compress(x, -24, 2.5, 0.004, 0.12)
    return np.stack([x, x])


# ------------------------------------------------------------------ Robo
def robot_core(x, mode):
    t = tvec(len(x))
    if mode == "warm":
        y = hp(x, 140, 2)
        y = lp(y, 7500, 2)
        y = eq(y, "peak", 280, 2.5, 0.8)
        y = eq(y, "peak", 1600, 2.0, 1.0)
        y = comb(y, hz("A3"), 0.25) * 0.6 + y * 0.4
        return y
    heavy = mode == "glitch"
    y = bp(x, 300, 5000, 2)
    y = eq(y, "peak", 1600, 4.5, 1.2)
    y = eq(y, "peak", 600, -3, 1.0)
    y = 0.55 * y + 0.45 * comb(y, hz("A4"), 0.55 if not heavy else 0.65)   # metallic resonance
    am = 1 - 0.3 + 0.3 * np.sin(2 * np.pi * 50 * t)                      # subtle ring/AM
    y = y * am
    y = crush(y, 7 if heavy else 9, 4 if heavy else 3)
    y = lp(y, 6500, 4)                                                  # keep the grit, lose the fizz
    return y


def robo(x, mode, phones):
    x = cleanup(x)
    x = eq(x, "peak", 200, -2, 0.8)
    n = len(x)
    if mode == "glitch":
        x = glitch_edit(x)
        t = tvec(n)
        wob = 0.0006 * np.sin(2 * np.pi * 6 * t) + 0.0012 * slow_noise(n, 3)
        x = mod_delay(x, 0.004 + wob)
    proc = robot_core(x, mode)
    dry = lp(hp(x, 150), 6000)
    y = (0.75 * proc + 0.45 * dry) if mode != "warm" else (0.7 * proc + 0.5 * dry)
    y = compress(y, -24, 3.0, 0.003, 0.1)
    st = np.stack([y, y])
    if mode == "warm":   # gentle chorus
        n_ = np.arange(n, dtype=float)
        for c, (base, rate) in enumerate(((0.012, 0.7), (0.016, 0.9))):
            d = base + 0.002 * np.sin(2 * np.pi * rate * n_ / SR + c)
            st[c] += 0.25 * mod_delay(y, d)
    return st


def glitch_edit(x):
    """Stutters in the pauses, short dropouts in consonants, tape-stop on the final 「す」."""
    y = x.copy()
    g = lambda a, b: x[int(a * SR):int(b * SR)]

    def put(t, seg, gain):
        i = int(t * SR)
        seg = fade(seg, 0.004, 0.012) * gain
        y[i:i + len(seg)] += seg

    put(1.03, g(0.90, 0.985), 0.8)          # の-う-う
    put(1.13, g(0.90, 0.985), 0.6)
    put(1.27, g(1.47, 1.585), 0.75)         # ゼ-ゼ-ゼロ
    put(1.37, g(1.47, 1.585), 0.85)
    put(1.88, g(1.69, 1.80), 0.7)           # ロ-ロ
    put(1.99, g(1.69, 1.80), 0.5)
    for a, b in ((0.30, 0.325), (0.56, 0.585), (2.50, 2.53)):    # dropouts (in s/k/s)
        i, j = int(a * SR), int(b * SR)
        w = np.ones(j - i)
        r = int(0.003 * SR)
        w[:r] = np.linspace(1, 0, r)
        w[-r:] = np.linspace(0, 1, r)
        w[r:-r] = 0.0
        y[i:j] *= w
    # tape-stop: from 2.62 s the playback rate falls 1 -> 0.45 (pitch drop, a bit longer)
    i0 = int(2.62 * SR)
    tail = y[i0:].copy()
    m = int(0.35 * SR)
    rate = np.concatenate([np.linspace(1.0, 0.45, m), np.full(len(tail) - m, 0.45)])
    pos = np.cumsum(rate) - rate[0]
    y[i0:] = np.interp(pos, np.arange(len(tail)), tail, right=0.0)
    return y


# ------------------------------------------------------------------ main
def main():
    tl = load_timeline()
    meta = {v["id"]: v for v in load_voices()}
    awning = make_ir(t60=0.4, predelay=0.006, seed=51, lf=1.0, hf=0.5, density_ms=3)    # shop front
    canopy = make_ir(t60=0.28, predelay=0.004, seed=52, lf=1.1, hf=0.4, density_ms=3)   # under the umbrella
    street = make_ir(t60=0.9, predelay=0.02, seed=53, lf=0.8, hf=0.6, density_ms=7)     # open, sunny
    heart = make_ir(t60=1.3, predelay=0.03, seed=54, lf=0.9, hf=0.6)                   # l11 warmth
    bus = Bus()
    for ln in tl["lines"]:
        m = meta[ln["id"]]
        x = read_wav(ROOT / m["file"])
        x = np.concatenate([x, np.zeros(int(2.0 * SR))])
        t0 = ln["start"]
        ir = awning if t0 < 27 else (canopy if t0 < 49 else street)
        if m["who"] == "robo":
            mode = {"l09_robo": "glitch", "l11_robo": "warm"}.get(ln["id"], "robot")
            y = robo(x, mode, m["phones"])
            y = y + (0.22 * reverb(hp(y, 250), heart) if mode == "warm" else 0.12 * reverb(hp(y, 250), ir))
        else:
            y = mina(x)
            y = y + 0.15 * reverb(hp(y, 200), ir, width=1.2)
        voiced = y[:, :int(m["dur"] * SR)]
        g = LINE_LUFS + LINE_TRIM.get(ln["id"], 0.0) - lufs(voiced)
        y = y * undb(g)
        # tuned beeps on 「ピピッ」 (blended under the voiced syllables, after loudness match)
        for tb, nt in BEEPS.get(ln["id"], []):
            b = beep(nt, 0.06, undb(-21))
            i = int(tb * SR)
            y[:, i:i + len(b)] += b
        y = fade(y, 0.002, 0.3)
        bus.add(t0, y)
        print(f"  {ln['id']:9s} @ {t0:5.2f}s  dur {m['dur']:.2f}s  gain {g:+.1f} dB")
    out = bus.out().copy()
    report("dialogue", out)
    print(write_wav(BUILD / "dialogue.wav", out))


if __name__ == "__main__":
    main()
