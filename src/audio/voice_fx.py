"""Dialogue processing -> build/audio/dialogue.wav (48 kHz stereo, 60.000 s).

Each synthesized line (assets/voice/*.wav, untouched) is processed and placed
at its start time from story/timeline.json.

Haru : HP 90 Hz, warmth (+2 dB @ 250 Hz), de-ess, LP 9.5 kHz (tames HTS buzz),
       2.5:1 compression, short room/outdoor reverb.
Kira : same clean-up, plus a quiet octave-up granular double, stereo chorus,
       tuned bell resonators (E6/B6/E7), a shimmering airy top band, tiny
       sparkle pings on phrase onsets, and a longer airy reverb.  The dry
       voice stays dominant so every word is intelligible.
"""
import numpy as np
import scipy.signal as ss

from dsp import (SR, N, BUILD, ROOT, Bus, tvec, fade, lp, hp, bp, eq, compress, deess, make_ir, reverb,
                 read_wav, write_wav, report, db, undb, lufs, load_timeline, load_voices)

RNG = np.random.default_rng(99)
LINE_LUFS = -20.0
LINE_TRIM = {"l07_haru": +0.5, "l08_haru": -1.0, "l12_haru": -2.0, "l06_kira": -0.5}


def pitch_shift(x, semitones, win=0.04):
    """Two-tap delay-line (granular) pitch shifter, duration preserving."""
    r = 2 ** (semitones / 12)
    W = win * SR
    n = np.arange(len(x), dtype=float)
    out = np.zeros(len(x))
    for off in (0.0, 0.5):
        p = np.mod(n * (r - 1) / W + off, 1.0)
        pos = n - (1 - p) * W
        out += np.interp(pos, n, x, left=0.0) * np.sin(np.pi * p) ** 2
    return out


def mod_delay(x, base, depth, rate, phase=0.0):
    n = np.arange(len(x), dtype=float)
    d = (base + depth * np.sin(2 * np.pi * rate * n / SR + phase)) * SR
    return np.interp(n - d, n, x, left=0.0)


def cleanup(x):
    x = hp(x, 90, 2)
    x = deess(x, 6000, thresh_db=-34, ratio=3)
    x = lp(x, 9500, 4)
    return x


def ping(f, tau, amp):
    n = int(tau * 6 * SR)
    t = tvec(n)
    return amp * fade(np.sin(2 * np.pi * f * t) * np.exp(-t / tau), 0.001, 0.01)


def haru(x):
    x = cleanup(x)
    x = eq(x, "peak", 250, 2.0, 0.8)
    x = eq(x, "peak", 3200, 1.0, 1.0)      # a touch of presence
    x = compress(x, -24, 2.5, 0.004, 0.12, makeup_db=0)
    return np.stack([x, x])


def kira(x, phones):
    x = cleanup(x)
    x = eq(x, "peak", 300, -1.5, 0.9)      # lighter, less body
    x = compress(x, -24, 2.5, 0.004, 0.12)
    dry = np.stack([x, x])
    # octave-up double (quiet) + stereo chorus
    octv = lp(pitch_shift(x, 12), 8000) * undb(-17)
    chL = mod_delay(x, 0.014, 0.003, 0.8) * undb(-11)
    chR = mod_delay(x, 0.019, 0.004, 1.1, 1.0) * undb(-11)
    # tuned bell resonators (ringing sympathetic "star" tone)
    res = np.zeros_like(x)
    for f, g in [(1318.5, 1.0), (1975.5, 0.7), (2637.0, 0.5)]:
        b, a = ss.iirpeak(f, 60, fs=SR)
        res += g * ss.lfilter(b, a, x)
    res *= undb(-15)
    # airy shimmer: HF band with a fast random tremolo
    air = bp(x, 4000, 9000)
    trem = 0.5 + 0.5 * np.sin(2 * np.pi * 17 * tvec(len(x)) + 2 * np.sin(2 * np.pi * 3.1 * tvec(len(x))))
    air = air * trem * undb(-12)
    wet = np.stack([chL + 0.8 * octv + res + air, chR + 0.8 * octv + res + air])
    y = dry + wet
    # tiny sparkle pings on phrase onsets (first voiced phone after silence/pause)
    prev = "sil"
    for ph in phones:
        if prev in ("sil", "pau") and ph["p"] not in ("sil", "pau"):
            i = int(ph["t0"] * SR)
            for k in range(3):
                pg = ping(RNG.uniform(5000, 9000), RNG.uniform(0.03, 0.07), undb(-36))
                j = i + int(RNG.uniform(0, 0.12) * SR)
                pan = RNG.uniform(-0.7, 0.7)
                L = min(len(pg), len(x) - j)
                if L > 0:
                    y[0, j:j + L] += pg[:L] * np.cos((pan + 1) * np.pi / 4) * 1.41
                    y[1, j:j + L] += pg[:L] * np.sin((pan + 1) * np.pi / 4) * 1.41
        prev = ph["p"]
    return y


def main():
    tl = load_timeline()
    meta = {v["id"]: v for v in load_voices()}
    room = make_ir(t60=0.45, predelay=0.008, seed=21, lf=1.0, hf=0.5, density_ms=4)     # attic
    outdoor = make_ir(t60=0.8, predelay=0.015, seed=22, lf=0.9, hf=0.5, density_ms=6)  # garden / hill
    airy = make_ir(t60=1.7, predelay=0.035, seed=23, lf=0.7, hf=0.8)                     # Kira
    bus = Bus()
    for ln in tl["lines"]:
        m = meta[ln["id"]]
        x = read_wav(ROOT / m["file"])
        pad = np.zeros(int(2.0 * SR))           # room for reverb tails
        x = np.concatenate([x, pad])
        if m["who"] == "kira":
            y = kira(x, m["phones"])
            y = y + 0.28 * reverb(hp(y, 300), airy, width=1.3)
        else:
            y = haru(x)
            ir = room if ln["start"] < 13.0 else outdoor
            y = y + 0.16 * reverb(hp(y, 200), ir)
        # per-line loudness normalisation (measured on the voiced part)
        voiced = y[:, :int(m["dur"] * SR)]
        g = LINE_LUFS + LINE_TRIM.get(ln["id"], 0.0) - lufs(voiced)
        y = fade(y * undb(g), 0.002, 0.3)
        bus.add(ln["start"], y)
        print(f"  {ln['id']:9s} @ {ln['start']:5.2f}s  dur {m['dur']:.2f}s  gain {g:+.1f} dB")
    out = bus.out().copy()
    report("dialogue", out)
    print(write_wav(BUILD / "dialogue.wav", out))


if __name__ == "__main__":
    main()
