"""Original score for 「あめふりロボ」 -> build/audio/music.wav (48 kHz stereo, 60.000 s).

"Ame-Robo" motif (F major, 2 bars; dotted, jaunty, a leap up to repeated C6 "pi-pi"):

    | C5.  F5 A5.  G5 | F5  C6  C6  - |  D6 C6 A5 F5 | G5 ---   (half cadence)
    answer bar 4:                         D6 C6 A5 G5 | F5 ---   (full cadence)

Key / tempo plan (downbeats placed on story beats):
  s01  0.0– 6.0  F major, 120 bpm (bar 2 s). Intro bar (pizz oom-pah, marimba off-beats,
                 brushes), then whistle/recorder states the motif; ends on the V (C) chord.
  s02  6.0–10.6  D minor, 72 bpm rubato: lone piano + celesta, minor version of the motif
                 (drip on her hat 9.8 = celesta "plink" A6).
      10.6–15.5  F major: squeaky bassoon + tuba "robot waddle" (16ths), "bwomp" stop at 11.3,
                 light pizz oom-pah + glock beep echoes in Robo's pauses.
  s03 15.5–27.0  F major 120 bpm: tiptoe pizzicato (curious, ends on C7 "?"), snare-roll
                 anticipation 19.4, tiny jaw-harp boing + toy ta-da at 20.2, comic silence,
                 xylophone laugh motif 21.6, woodblock tick-tock + "loading" marimba on C7
                 + thinking bassoon 23.2-26.2, "ding" resolution 26.35.
  s04 27.0–38.0  126 bpm (beat 0.475): mecha build bVI-bVII on a C pedal (Db-Eb-Db-Eb-Gm7/C-C7),
                 brass stabs + timpani + snare roll -> F major tutti + crash at 30.8 (canopy),
                 brass/whistle motif, march snare; 34.6 sad-trombone wah-wah deflate (Bb-A-Ab-G).
  s05 38.0–43.5  bit-crushed chiptune motif that stutters, sticks and powers down (D minor),
                 "hmph" stinger after 「もう！」.
      43.5–49.0  72 bpm tender theme: music box melody + warm strings/pad + cello;
                 Bbmaj7 -> F/A swell at 45.6 「あたたかい」 -> Dm7 -> Gm7 -> C7.
  s06 49.0–60.0  120 bpm. 49 F (rain stops), 50 D7 harp gliss + riser -> 51.0 G major (key lift)
                 full ensemble: strings, glockenspiel, marimba, whistle, light drums, rainbow
                 shimmer; motif restated triumphantly in G; button ending: G chord at 59.0 +
                 "pip" (G6) at 59.4, silent by 60.0.
"""
import _common  # noqa: F401  (sets FILM_DIR + sys.path)
import numpy as np

from dsp import (SR, N, BUILD, Bus, hz, midi, tvec, env_adsr, fade, saw, sine, lp, hp, bp, svf,
                 pluck, partials, make_ir, reverb, limiter, compress, lufs, true_peak, write_wav,
                 report, db, undb, eq)

RNG = np.random.default_rng(20261001)


def tr(note, semis):
    return midi(note) + semis


# =============================================================== instruments
def piano(note, vel=0.5, dur=None):
    """Additive piano: inharmonic partials, two-stage decay, detuned unison, hammer."""
    f = hz(note)
    T = float(np.clip(3.0 * (262.0 / f) ** 0.6, 0.6, 6.0))
    n = int(SR * min(T * 2.5, 6.0))
    t = tvec(n)
    y = np.zeros(n)
    B = 0.00035
    for det in (-0.9, 0.9):
        fd = f * 2 ** (det / 1200)
        for k in range(1, 14):
            fk = fd * k * np.sqrt(1 + B * k * k)
            if fk > 16000:
                break
            a = k ** -1.15 * np.exp(-(k - 1) * (0.42 - 0.3 * vel))
            tau = T / (1 + 0.35 * (k - 1))
            e = 0.55 * np.exp(-t / (tau * 0.18)) + 0.45 * np.exp(-t / tau)
            y += a * e * np.sin(2 * np.pi * fk * t + RNG.uniform(0, 6))
    m = int(0.006 * SR)
    y[:m] += fade(lp(RNG.standard_normal(m), 1800), 0.0003, 0.004) * 0.25
    if dur is not None and dur + 0.25 < n / SR:
        y *= env_adsr(n, 0.001, 0, 1, 0.25, hold=dur)
    return vel * 0.35 * fade(y, 0.001, 0.05)


def celesta(note, vel=0.5):
    f = hz(note)
    T = float(np.clip(1.1 * (523.0 / f) ** 0.5, 0.35, 2.5))
    n = int(SR * min(T * 5 + 0.05, 6))
    y = partials(f, n, [(1, 1, T), (2, 0.16, T * 0.45), (3, 0.05, T * 0.3), (4, 0.13, T * 0.2),
                        (6.1, 0.03, T * 0.08)], RNG, att=0.002)
    return vel * y


def glock(note, vel=0.5):
    f = hz(note)
    T = float(np.clip(1.3 * (1568.0 / f) ** 0.3, 0.5, 2.0))
    n = int(SR * T * 4)
    y = partials(f, n, [(1, 1, T), (2.76, 0.32, T * 0.22), (5.4, 0.14, T * 0.1), (8.93, 0.05, T * 0.05)],
                 RNG, att=0.0008)
    m = int(0.002 * SR)
    y[:m] += hp(RNG.standard_normal(m), 5000) * 0.1
    return vel * y


def musicbox(note, vel=0.5):
    f = hz(note)
    T = float(np.clip(1.3 * (880.0 / f) ** 0.4, 0.4, 2.5))
    n = int(SR * min(T * 5, 6))
    y = partials(f, n, [(1, 1, T), (2, 0.22, T * 0.5), (3, 0.06, T * 0.3), (5.4, 0.1, T * 0.1),
                        (8.9, 0.04, T * 0.05)], RNG, att=0.0015)
    click = hp(RNG.standard_normal(int(0.004 * SR)), 3000) * 0.08
    y[:len(click)] += fade(click, 0.0003, 0.002)
    return vel * y


def marimba(note, vel=0.5):
    f = hz(note)
    T = float(0.5 * (262.0 / f) ** 0.35)
    n = int(SR * T * 6)
    y = partials(f, n, [(1, 1, T), (3.93, 0.28, T / 4), (9.2, 0.06, T / 12)], RNG, att=0.001)
    m = int(0.005 * SR)
    y[:m] += fade(lp(RNG.standard_normal(m), 2500), 0.0003, 0.003) * 0.15
    return vel * y


def xylo(note, vel=0.5):
    f = hz(note)
    T = 0.16 * (1046.0 / f) ** 0.3
    n = int(SR * 0.9)
    y = partials(f, n, [(1, 1, T), (3.0, 0.4, T / 2.5), (6.1, 0.12, T / 6)], RNG, att=0.0006)
    m = int(0.004 * SR)
    y[:m] += fade(hp(RNG.standard_normal(m), 2500) * np.exp(-tvec(m) / 0.001), 0.0004, 0.001) * 0.2
    return vel * y


def pizz(note, vel=0.5, t60=0.45):
    y = pluck(hz(note), t60 + 0.3, t60=t60, damp=0.45, bright=3200, rng=RNG)
    y = eq(y, "peak", 250, 3, 1.0)
    return vel * lp(y, 3500)


def harp(note, vel=0.5):
    y = pluck(hz(note), 2.5, t60=1.8, damp=0.25, bright=6000, rng=RNG, pos=0.2)
    return vel * lp(y, 7000)


def whistle(note, dur, vel=0.5):
    """Recorder/whistle: near-sine with a few odd-ish harmonics, chiff, scoop, delayed vibrato."""
    f0 = hz(note)
    n = int((dur + 0.12) * SR)
    t = tvec(n)
    vib = 14 * np.clip((t - 0.18) / 0.25, 0, 1) * np.sin(2 * np.pi * 5.6 * t)
    cents = vib - 25 * np.exp(-t / 0.025)
    ph = np.cumsum(f0 * 2 ** (cents / 1200)) / SR
    tone = (np.sin(2 * np.pi * ph) + 0.12 * np.sin(4 * np.pi * ph) + 0.08 * np.sin(6 * np.pi * ph)
            + 0.02 * np.sin(10 * np.pi * ph))
    env = env_adsr(n, a=0.018, d=0.08, s=0.85, r=0.08, hold=dur)
    nz = RNG.standard_normal(n)
    chiff = bp(nz, 2000, 6000) * np.exp(-t / 0.018) * 0.25
    breath = bp(nz, f0 * 0.9, f0 * 1.1) * 0.05 * env
    return vel * fade(tone * env + chiff + breath, 0.001, 0.02)


def reed(note, dur, vel=0.5, squeak=0.0, formants=((480, 6), (1150, 3)), cutoff=2600, stac=True):
    """Bassoon-ish double reed: narrow pulse -> nasal formants. `squeak` adds a comic upward
    pitch overshoot on the attack (squeaky toy feel)."""
    f0 = hz(note)
    rel = 0.05 if stac else 0.15
    n = int((dur + rel + 0.02) * SR)
    t = tvec(n)
    cents = squeak * 900 * np.exp(-t / 0.03) + 10 * np.sin(2 * np.pi * 5 * t) * np.clip(t / 0.3, 0, 1)
    f = f0 * 2 ** (cents / 1200)
    s1 = saw(f, n, RNG.uniform())
    s2 = saw(f, n, RNG.uniform() + 0.3)
    p = s1 - np.roll(s2, 0)  # two offset saws -> pulse-ish
    y = lp(p, cutoff, 2)
    for fc, g in formants:
        y = eq(y, "peak", fc, g, 2.5)
    env = env_adsr(n, a=0.015, d=0.06, s=0.8, r=rel, hold=dur)
    return vel * 0.4 * fade(y * env, 0.001, 0.01)


def tuba(note, dur, vel=0.5):
    f0 = hz(note)
    n = int((dur + 0.12) * SR)
    t = tvec(n)
    f = f0 * 2 ** (-40 * np.exp(-t / 0.04) / 1200)
    y = lp(saw(f, n, RNG.uniform()), 700, 2) + 0.6 * sine(f, n)
    env = env_adsr(n, a=0.025, d=0.1, s=0.75, r=0.1, hold=dur)
    return vel * 0.5 * y * env


def brass(note, dur, vel=0.6, bright=1.0, att=0.03, rel=0.12):
    """Brassy synth: 3 detuned saws through a filter envelope that 'blats' open."""
    f0 = hz(note)
    n = int((dur + rel + 0.02) * SR)
    t = tvec(n)
    fenv = np.exp(-t / 0.18) * 0.6 + 0.4 * np.clip(t / 0.05, 0, 1)
    fc = 350 + bright * vel * 4200 * fenv * env_adsr(n, 0.02, 0, 1, rel, hold=dur)
    cents = -30 * np.exp(-t / 0.03) + 6 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.25) / 0.3, 0, 1)
    y = np.zeros(n)
    for dc in (-7, 0, 6):
        y += saw(f0 * 2 ** ((cents + dc) / 1200), n, RNG.uniform())
    y = svf(y, fc, 0.9, "lp")
    y = eq(y, "peak", 1200, 2.5, 1.0)
    env = env_adsr(n, a=att, d=0.12, s=0.8, r=rel, hold=dur)
    return vel * 0.22 * y * env


def trombone_wah(note, dur, vel=0.6, slide=0.0, wah_rate=0.0, wobble=0.0):
    """Plunger-muted trombone: saw through a 'wah' lowpass; optional slide down and wobble."""
    f0 = hz(note)
    n = int((dur + 0.15) * SR)
    t = tvec(n)
    x = np.clip(t / max(dur, 0.01), 0, 1)
    cents = slide * 100 * x ** 2 + wobble * np.sin(2 * np.pi * 6.5 * t) * x
    f = f0 * 2 ** (cents / 1200)
    y = saw(f, n, RNG.uniform()) + 0.5 * saw(f * 1.003, n, RNG.uniform())
    if wah_rate > 0:
        w = 0.5 - 0.5 * np.cos(2 * np.pi * wah_rate * t)
    else:  # one "wah": closed -> open -> closed
        w = np.sin(np.pi * x) ** 0.7
    y = svf(y, 280 + 1500 * w, 2.0, "lp")
    env = env_adsr(n, a=0.04, d=0.05, s=0.9, r=0.12, hold=dur)
    return vel * 0.35 * y * env


def crush(x, bits=4, hold=5):
    y = np.repeat(x[::hold], hold)[:len(x)]
    q = 2 ** (bits - 1)
    return np.round(y * q) / q


def chip(note, dur, vel=0.5, duty=0.25, bits=4, hold=6, bend=0.0):
    f0 = hz(note)
    n = int((dur + 0.01) * SR)
    t = tvec(n)
    f = f0 * 2 ** (bend * t / max(dur, 0.01) / 12)
    ph = np.mod(np.cumsum(f) / SR, 1.0)
    sq = np.where(ph < duty, 1.0, -1.0)
    sq -= sq.mean()
    env = env_adsr(n, 0.002, 0.04, 0.7, 0.01, hold=dur)
    y = crush(sq * env * 0.8, bits, hold)
    return vel * 0.3 * lp(y, 9000)


# ------------------------------------------------------------- percussion
def kick(vel=0.6):
    n = int(0.35 * SR)
    t = tvec(n)
    f = 52 * (1 + 1.6 * np.exp(-t / 0.03))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13)
    y += 0.2 * lp(RNG.standard_normal(n), 2000) * np.exp(-t / 0.004)
    return vel * fade(y, 0.0005, 0.03)


def snare(vel=0.5, tone=190, snappy=1.0, tau=0.09):
    n = int(0.3 * SR)
    t = tvec(n)
    nz = bp(RNG.standard_normal(n), 1200, 9000)
    y = snappy * nz * np.exp(-t / tau) * 0.8 + np.sin(2 * np.pi * tone * t) * np.exp(-t / 0.04) * 0.6
    return vel * fade(y, 0.0005, 0.03)


def hat(vel=0.3, tau=0.03):
    n = int(0.15 * SR)
    t = tvec(n)
    y = hp(RNG.standard_normal(n), 7000) * np.exp(-t / tau)
    return vel * fade(y, 0.0005, 0.02)


def brush_tap(vel=0.3):
    n = int(0.12 * SR)
    t = tvec(n)
    y = bp(RNG.standard_normal(n), 1500, 9000) * (1 - np.exp(-t / 0.003)) * np.exp(-t / 0.035)
    return vel * fade(y, 0.001, 0.02)


def brush_swish(dur=0.4, vel=0.3):
    n = int(dur * SR)
    t = tvec(n)
    fc = 2500 + 3000 * np.sin(np.pi * t / dur)
    y = svf(RNG.standard_normal(n), fc, 0.7, "bp")
    env = np.sin(np.pi * t / dur) ** 1.5
    return vel * 0.6 * fade(y * env, 0.005, 0.02)


def shaker(vel=0.3):
    n = int(0.12 * SR)
    t = tvec(n)
    y = bp(RNG.standard_normal(n), 5000, 12000)
    env = (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.03)
    return vel * fade(y * env, 0.001, 0.02)


def woodblock(f=1100.0, vel=0.4):
    n = int(0.15 * SR)
    t = tvec(n)
    y = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.025) + 0.4 * np.sin(2 * np.pi * f * 2.3 * t) * np.exp(-t / 0.01)
    y += 0.2 * bp(RNG.standard_normal(n), 2000, 6000) * np.exp(-t / 0.003)
    return vel * fade(y, 0.0004, 0.02)


def timpani(note, vel=0.6, tau=0.9):
    f = hz(note)
    n = int(2.5 * SR)
    t = tvec(n)
    ff = f * (1 + 0.08 * np.exp(-t / 0.05))
    y = (np.sin(2 * np.pi * np.cumsum(ff) / SR) + 0.35 * np.sin(2 * np.pi * np.cumsum(ff * 1.5) / SR)
         * np.exp(-t / 0.3)) * np.exp(-t / tau)
    y += 0.4 * lp(RNG.standard_normal(n), 900) * np.exp(-t / 0.03)
    return vel * fade(y, 0.0008, 0.1)


def cymbal(vel=0.4, tau=1.4, swell=0.0):
    """Crash; with swell>0 returns a reversed-envelope swell of that length (ending at its end)."""
    if swell > 0:
        n = int(swell * SR)
        t = tvec(n)
        y = hp(RNG.standard_normal(n), 4000) * (t / swell) ** 2.5
        return vel * fade(y, 0.01, 0.005)
    n = int(tau * 3 * SR)
    t = tvec(n)
    nz = RNG.standard_normal(n)
    y = hp(nz, 3000) * np.exp(-t / tau) + 0.4 * bp(nz, 400, 3000) * np.exp(-t / 0.15)
    return vel * fade(y, 0.0008, 0.2)


def roll(bus_put, t0, t1, v0, v1, stroke=0.032, pan=0.1):
    t = t0
    while t < t1:
        x = (t - t0) / (t1 - t0)
        bus_put(t + RNG.uniform(-0.003, 0.003), snare(v0 + (v1 - v0) * x ** 1.5, snappy=1.2, tau=0.05), pan)
        t += stroke * RNG.uniform(0.9, 1.1)


def boing(f0=330.0, dur=0.6, vel=0.5):
    """Jaw-harp/spring boing: pitch wobble with decaying depth, formant sweep."""
    n = int(dur * SR)
    t = tvec(n)
    f = f0 * 2 ** ((5 * np.exp(-t / 0.12) * np.sin(2 * np.pi * 13 * t) + 3 * np.exp(-t / 0.03)) / 12)
    y = saw(f, n, 0.0)
    fc = 600 + 2200 * (0.5 + 0.5 * np.sin(2 * np.pi * 7 * t)) * np.exp(-t / 0.4)
    y = svf(y, fc, 3.0, "bp")
    return vel * fade(y * np.exp(-t / (dur * 0.35)), 0.001, 0.05)


def beep(note, dur=0.09, vel=0.3):
    n = int((dur + 0.02) * SR)
    y = sine(hz(note), n) + 0.15 * sine(2 * hz(note), n)
    return vel * y * env_adsr(n, 0.003, 0, 1, 0.015, hold=dur)


# =============================================================== the score
class Score:
    def __init__(self):
        self.dry = Bus()
        self.send = Bus()
        self.pad = Bus()

    def put(self, t, sig, pan=0.0, gain=1.0, rev=0.2):
        self.dry.add(t, sig, pan, gain)
        if rev > 0:
            self.send.add(t, sig, pan, gain * rev)

    def strings(self, t0, t1, notes, vel=0.5, att=0.4, rel=0.8, swell=None, bright=2600):
        """Warm string section (detuned saw ensemble, lowpassed). swell=(t, dB, width)."""
        dur = t1 - t0
        n = int((dur + rel) * SR)
        env = env_adsr(n, a=att, d=0, s=1, r=rel, hold=dur)
        t = tvec(n)
        if swell is not None:
            ts, gdb, w = swell
            env = env * undb(gdb * np.exp(-0.5 * ((t - (ts - t0)) / w) ** 2))
        for note in notes:
            f0 = hz(note)
            drift = 3 * np.sin(2 * np.pi * RNG.uniform(0.15, 0.35) * t + RNG.uniform(0, 6))
            vib = 7 * np.sin(2 * np.pi * RNG.uniform(4.8, 5.6) * t + RNG.uniform(0, 6)) * np.clip(t / 0.6, 0, 1)
            for k, dc in enumerate((-9, -3, 4, 10)):
                y = saw(f0 * 2 ** ((dc + drift + vib) / 1200), n, RNG.uniform())
                self.pad.add(t0, lp(y, bright, 2) * env * vel * 0.07, (-0.7, -0.2, 0.25, 0.7)[k])

    def seq(self, fn, events, t0, beat, pan=0.0, gain=1.0, rev=0.2, sustained=False, **kw):
        """events: [(beat, note, dur_beats, vel), ...]"""
        for ev in events:
            b, note = ev[0], ev[1]
            dur = ev[2] if len(ev) > 2 else 1
            vel = ev[3] if len(ev) > 3 else 0.5
            if note is None:
                continue
            sig = fn(note, dur * beat, vel, **kw) if sustained else fn(note, vel, **kw)
            p = pan(b) if callable(pan) else pan
            self.put(t0 + b * beat, sig, p, gain, rev)


MOTIF_A = [(0, "C5", 0.75, 0.55), (0.75, "F5", 0.25, 0.5), (1, "A5", 0.75, 0.55), (1.75, "G5", 0.25, 0.5),
           (2, "F5", 0.5, 0.55), (2.5, "C6", 0.5, 0.6), (3, "C6", 0.45, 0.6)]
MOTIF_B = [(0, "D6", 0.5, 0.6), (0.5, "C6", 0.5, 0.55), (1, "A5", 0.5, 0.55), (1.5, "F5", 0.5, 0.5),
           (2, "G5", 1.5, 0.55)]
MOTIF_B2 = [(0, "D6", 0.5, 0.6), (0.5, "C6", 0.5, 0.55), (1, "A5", 0.5, 0.55), (1.5, "G5", 0.5, 0.5),
            (2, "F5", 1.8, 0.6)]


def shift(events, semis=0, boff=0.0, vel=1.0, octave=0):
    return [(b + boff, tr(nt, semis + 12 * octave), d, v * vel) for b, nt, d, v in events]


def oompah(S, t0, beat, chords, gain=1.0, pan=-0.25):
    """chords: [(beat, bass, [upper notes])] -> pizz bass on the beat, marimba off-beat chords."""
    for b, bass, up in chords:
        S.put(t0 + b * beat, pizz(bass, 0.7), pan, gain, 0.12)
        for nt in up:
            S.put(t0 + (b + 0.5) * beat, marimba(nt, 0.16), 0.3, gain, 0.2)


# ------------------------------------------------------------------ s01 0-6
def s01(S):
    bt = 0.5
    chords = []
    seq_ch = [("F2", ["A3", "C4", "F4"]), ("C3", ["A3", "C4", "F4"])] * 4 + \
             [("Bb2", ["Bb3", "D4", "F4"]), ("F2", ["Bb3", "D4", "F4"]), ("C3", ["Bb3", "C4", "E4"]), ("G2", ["Bb3", "C4", "E4"])]
    for i, (bs, up) in enumerate(seq_ch):
        chords.append((i, bs, up))
    oompah(S, 0.0, bt, chords)
    # intro: marimba "raindrop" pickup arpeggio in bar 0
    S.seq(marimba, [(0.5, "F5", 1, 0.25), (1.0, "A5", 1, 0.25), (1.5, "C6", 1, 0.3), (2.5, "A5", 1, 0.22),
                    (3.0, "C6", 1, 0.25), (3.25, "F6", 1, 0.28)], 0.0, bt, pan=0.45, rev=0.3)
    # brushes: taps on 8ths, swishes on 2 and 4
    for i in range(12):
        S.put(0.0 + i * bt, brush_tap(0.12 if i % 2 == 0 else 0.07), 0.2, rev=0.08)
        S.put(0.25 + i * bt, brush_tap(0.05), 0.25, rev=0.05)
        if i % 2 == 1:
            S.put(i * bt - 0.05, brush_swish(0.35, 0.10), -0.2, rev=0.1)
    # whistle motif (bars 1-2), glock doubling an octave up, very quiet
    S.seq(whistle, MOTIF_A + shift(MOTIF_B, boff=4), 2.0, bt, pan=0.05, rev=0.35, sustained=True, gain=0.9)
    S.seq(glock, shift(MOTIF_A, octave=1, vel=0.25), 2.0, bt, pan=-0.35, rev=0.3)
    S.strings(2.0, 4.0, ["F3", "C4", "A4"], 0.25, att=0.3)
    S.strings(4.0, 5.0, ["Bb3", "D4", "F4"], 0.25, att=0.15, rel=0.3)
    S.strings(5.0, 6.15, ["C4", "E4", "G4", "Bb4"], 0.27, att=0.15, rel=0.5)
    S.put(4.0, tuba("Bb1", 0.4, 0.2), 0)
    S.put(5.0, tuba("C2", 0.9, 0.22), 0)


# ------------------------------------------------------------------ s02 6-15.5
def s02(S):
    bt = 0.8333  # 72 bpm
    t0 = 6.0
    # lone piano: D minor -> Bbmaj7 -> Gm6 -> A7sus4..A7
    lh = [(0, "D3"), (0.5, "A3"), (1, "D4"), (1.5, "F4"), (2, "Bb2"), (2.5, "F3"), (3, "A3"), (3.5, "D4"),
          (4, "G2"), (4.5, "D3"), (5, "Bb3"), (5.5, "E4")]
    for b, nt in lh:
        S.put(t0 + b * bt, piano(nt, 0.32), -0.25, rev=0.4)
    S.put(t0 + 5.0 * bt, piano("A2", 0.3, dur=1.0), -0.3, rev=0.4)
    # celesta: the motif in minor, slowly (A4 D5 F5 E5 | D5 ... A5 G5 | F5 E5)
    mel = [(0.0, "A4", 0.75, 0.32), (0.75, "D5", 0.25, 0.28), (1, "F5", 0.75, 0.32), (1.75, "E5", 0.25, 0.26),
           (2, "D5", 1.0, 0.3), (3.0, "F5", 0.5, 0.26), (3.5, "E5", 0.5, 0.24), (4, "D5", 0.5, 0.25),
           (4.5, "Bb4", 0.5, 0.22), (5.0, "C#5", 0.6, 0.26)]
    S.seq(celesta, mel, t0, bt, pan=0.2, rev=0.5)
    S.strings(6.0, 10.4, ["D3", "A3"], 0.14, att=1.2, rel=0.6, bright=1400)
    # drip on her hat (9.8): a lone high plink
    S.put(9.8, celesta("A6", 0.22), 0.1, rev=0.6)
    # ---- 10.6 Robo rolls in: squeaky bassoon waddle + tuba, F major, 16ths of 120 bpm
    s16 = 0.125
    wad = [(0, "C3"), (1, "D3"), (2, "E3"), (3, "F3"), (4, "G3"), (5, "E3")]
    for i, nt in wad:
        S.put(10.6 + i * s16, reed(nt, 0.08, 0.55, squeak=0.6 if i % 2 == 0 else 0.25), 0.45, rev=0.15)
    for i in (0, 2, 4):
        S.put(10.6 + i * s16, tuba("F1" if i % 4 == 0 else "C2", 0.1, 0.5), 0.2, rev=0.05)
    # the stop: "bwomp" (11.3)
    S.put(11.3, tuba("F1", 0.25, 0.7), 0.1, rev=0.1)
    S.put(11.3, reed("F3", 0.2, 0.6, squeak=0.9), 0.3, rev=0.15)
    S.put(11.3, snare(0.12, snappy=0.4), 0.2, rev=0.1)
    # under Robo's line (11.5-15.25): tiptoe pizz oom-pah + glock beep-echoes in his pauses
    bt2 = 0.5
    ch = []
    for i in range(7):
        ch.append((2 * i, ["F2", "C3", "F2", "C3", "Bb2", "C3", "F2"][i],
                   [["A3", "C4"], ["A3", "C4"], ["A3", "C4"], ["G3", "Bb3"], ["Bb3", "D4"], ["Bb3", "E4"], ["A3", "C4"]][i]))
    oompah(S, 11.75, bt2, ch, gain=0.55)
    S.seq(glock, [(0, "A5", 1, 0.18), (0.25, "C6", 1, 0.2)], 12.05, 1.0, pan=0.4, rev=0.4)
    S.seq(glock, [(0, "C6", 1, 0.15), (0.2, "F6", 1, 0.18)], 13.15, 1.0, pan=0.4, rev=0.4)
    S.seq(glock, [(0, "F6", 1, 0.15), (0.2, "C6", 1, 0.13), (0.4, "A5", 1, 0.12)], 14.25, 1.0, pan=0.4, rev=0.4)
    S.put(15.25, reed("C3", 0.15, 0.3, squeak=0.4), 0.3, rev=0.1)


# ------------------------------------------------------------------ s03 15.5-27
def s03(S):
    bt = 0.5
    t0 = 15.5
    # tiptoe pizzicato, curious (15.5-18.5): staccato rising line ending on C7 "?"
    tip = [(0, "F3"), (0.5, "A3"), (1, "C4"), (1.5, "A3"), (2, "F3"), (2.5, "A3"), (3, "C4"), (3.5, "D4"),
           (4, "E3"), (4.5, "G3"), (5, "Bb3"), (5.5, "C4")]
    for b, nt in tip:
        S.put(t0 + b * bt, pizz(nt, 0.4, t60=0.25), -0.2 + 0.1 * (b % 2), rev=0.2)
    S.put(t0 + 6 * bt, pizz("E4", 0.35, t60=0.3), 0.3, rev=0.3)       # "?" (18.5)
    S.put(t0 + 6 * bt, pizz("Bb3", 0.3, t60=0.3), -0.3, rev=0.3)
    S.put(t0 + 6.0 * bt, xylo("G6", 0.12), 0.5, rev=0.4)
    S.put(t0 + 6.5 * bt, xylo("C7", 0.12), 0.5, rev=0.4)
    # anticipation (19.4-20.2): snare roll crescendo + chromatic pizz climb
    roll(lambda t, s, p: S.put(t, s, p, rev=0.1), 19.45, 20.15, 0.02, 0.16, stroke=0.03)
    for i, nt in enumerate(["C4", "C#4", "D4", "D#4", "E4"]):
        S.put(19.5 + i * 0.14, pizz(nt, 0.35, t60=0.2), -0.1, rev=0.15)
    # 20.2 tiny umbrella: tiny boing + toy ta-da
    S.put(20.2, boing(660, 0.45, 0.35), 0.15, rev=0.2)
    S.seq(glock, [(0, "C7", 1, 0.25), (0.12, "F7", 1, 0.3)], 20.25, 1.0, pan=0.2, rev=0.3)
    S.put(20.37, xylo("A6", 0.15), -0.2, rev=0.3)
    # (comic silence 20.5-21.6)
    # 21.6 laugh motif: xylophone "ha-ha-ha-ha" triplets, descending, + bouncy pizz + bassoon honk
    tri = 1 / 6
    laugh = ["C6", "A5", "C6", "A5", "Bb5", "G5", "A5", "F5", "G5", "E5", "F5"]
    for i, nt in enumerate(laugh):
        S.put(21.6 + i * tri, xylo(nt, 0.32 - 0.012 * i), 0.3 - 0.06 * (i % 2) * 6, rev=0.25)
    for i, nt in enumerate(["F3", "C3", "F3", "C3"]):
        S.put(21.6 + i * 0.5, pizz(nt, 0.4, t60=0.25), -0.3, rev=0.15)
    S.put(23.05, reed("F2", 0.18, 0.4, squeak=0.8), 0.2, rev=0.15)
    # 23.2-26.25 thinking: woodblock tick-tock + loading marimba on C7 + thinking bassoon
    k = 0
    t = 23.25
    while t < 26.2:
        S.put(t, woodblock(1250 if k % 2 == 0 else 880, 0.13), 0.35 if k % 2 == 0 else -0.35, rev=0.1)
        t += 0.5
        k += 1
    load = ["C5", "E5", "G5", "Bb5"]
    for i in range(23):
        S.put(23.4 + i * 0.125, marimba(load[i % 4], 0.06 + 0.002 * i), 0.4 * np.sin(i * 0.8), rev=0.2)
    S.put(23.5, reed("C3", 0.6, 0.18, squeak=0.0, stac=False), -0.1, rev=0.2)
    S.put(24.4, reed("D3", 0.6, 0.18, stac=False), -0.1, rev=0.2)
    S.put(25.35, reed("E3", 0.8, 0.2, stac=False), -0.1, rev=0.2)
    S.strings(23.3, 26.3, ["C3", "G3", "Bb3", "E4"], 0.1, att=1.0, rel=0.3, bright=1500)
    # 26.35 "もんだい、なし" -> resolution ding
    S.put(26.35, glock("F6", 0.35), 0.2, rev=0.5)
    S.put(26.35, glock("A6", 0.18), -0.2, rev=0.5)
    S.put(26.35, pizz("F2", 0.5), -0.2, rev=0.2)
    S.put(26.35, pizz("C4", 0.3), 0.2, rev=0.2)
    S.put(26.35, pizz("F4", 0.3), 0.3, rev=0.2)


# ------------------------------------------------------------------ s04 27-38
def s04(S):
    bt = 0.475
    t0 = 27.0
    # build: bVI-bVII on a C pedal, brass stabs on clacks, timpani, snare roll
    stabs = [(0, ["Db4", "F4", "Ab4"], 0.9), (1, ["Db4", "F4", "Ab4"], 0.4), (2, ["Eb4", "G4", "Bb4"], 0.9),
             (3, ["Eb4", "G4", "Bb4"], 0.4), (4, ["Db4", "F4", "Ab4", "Db5"], 0.9), (5, ["Eb4", "G4", "Bb4", "Eb5"], 0.95),
             (6, ["F4", "Bb4", "D5"], 0.8), (6.5, ["G4", "Bb4", "E5"], 0.85), (7, ["G4", "Bb4", "C5", "E5"], 1.0)]
    for b, notes, v in stabs:
        d = 0.18 if b < 7 else 0.4
        for j, nt in enumerate(notes):
            S.put(t0 + b * bt, brass(nt, d, 0.42 * v, bright=0.9), -0.35 + 0.7 * j / max(len(notes) - 1, 1), rev=0.25)
        S.put(t0 + b * bt, brass(tr(notes[0], -12), d, 0.3 * v), 0, rev=0.2)
    for b in range(8):
        S.put(t0 + b * bt, timpani("C2", 0.35 + 0.05 * b, tau=0.4), -0.1, rev=0.15)
        S.put(t0 + b * bt, tuba("C1", 0.15, 0.3), 0, rev=0.0)
    roll(lambda t, s, p: S.put(t, s, p, rev=0.12), t0 + 4 * bt, 30.78, 0.03, 0.3, stroke=0.028)
    S.put(30.8 - 1.6, cymbal(0.25, swell=1.6), 0.1, rev=0.2)
    S.strings(28.9, 30.8, ["C3", "G3", "C4", "E4", "G4", "Bb4"], 0.35, att=1.6, rel=0.05, bright=3200)
    # riser: rising string glissando
    n = int(1.9 * SR)
    t = tvec(n)
    f = hz("C4") * 2 ** (12 * (t / t[-1]) ** 1.5 / 12)
    gl = lp(saw(f, n) + saw(f * 1.004, n, 0.3), 3500) * (t / t[-1]) ** 1.5 * 0.05
    S.put(28.9, gl, 0.0, rev=0.4)
    # ---- 30.8 canopy opens: F major tutti + crash, joyful theme (motif) in brass + whistle
    t1 = 30.8
    S.put(t1, cymbal(0.45, tau=1.6), 0.2, rev=0.3)
    S.put(t1, timpani("F2", 0.8, tau=1.0), 0, rev=0.2)
    S.put(t1, kick(0.6), 0, rev=0.05)
    for j, nt in enumerate(["F3", "C4", "F4", "A4", "C5", "F5"]):
        S.put(t1, brass(nt, 0.9, 0.38, bright=1.0), -0.5 + 0.2 * j, rev=0.35)
    S.put(t1, tuba("F1", 0.9, 0.6), 0)
    S.seq(glock, [(0, "F6", 1, 0.3), (0.25, "A6", 1, 0.3), (0.5, "C7", 1, 0.3), (0.75, "F7", 1, 0.3)], t1, bt, pan=0.4, rev=0.4)
    tm = t1 + 2 * bt   # motif starts on beat 3 of the canopy bar (31.75) - fits the 126-bpm grid
    S.seq(brass, shift(MOTIF_A, octave=-1, vel=0.85), tm, bt, pan=-0.15, rev=0.3, sustained=True)
    S.seq(whistle, shift(MOTIF_A, vel=0.7), tm, bt, pan=0.15, rev=0.35, sustained=True)
    S.strings(t1, t1 + 2 * bt, ["F3", "C4", "A4", "F5"], 0.4, att=0.05, rel=0.3)
    S.strings(tm, tm + 2 * bt, ["F3", "C4", "A4"], 0.3, att=0.05)
    S.strings(tm + 2 * bt, tm + 4 * bt, ["Bb2", "F3", "D4"], 0.3, att=0.05)
    S.strings(tm + 4 * bt, tm + 5.6 * bt, ["C3", "G3", "E4", "Bb4"], 0.3, att=0.05, rel=0.25)
    # march: snare on 2&4, kick on 1&3, oom-pah tuba
    for i in range(8):
        tb = t1 + i * bt
        if tb >= 34.55:
            break
        if i % 2 == 0:
            S.put(tb, kick(0.45), 0, rev=0.03)
            S.put(tb, tuba(["F1", "Bb1", "F1", "C2"][i // 2], 0.2, 0.45), 0)
        else:
            S.put(tb, snare(0.22), 0.15, rev=0.12)
            S.put(tb + bt / 2, snare(0.07, snappy=0.7), 0.15, rev=0.1)
        S.put(tb + bt / 2, hat(0.08), 0.35, rev=0.05)
    # ---- 34.6 deflate: sad trombone wah-wah-wah-waaah
    for i, (nt, d) in enumerate([("Bb3", 0.28), ("A3", 0.28), ("Ab3", 0.28)]):
        S.put(34.6 + i * 0.36, trombone_wah(nt, d, 0.55), -0.1, rev=0.25)
    S.put(35.68, trombone_wah("G3", 1.25, 0.45, slide=-0.6, wah_rate=3.2, wobble=30), -0.1, rev=0.25)
    S.put(34.6, reed("C3", 0.5, 0.18, squeak=-0.5, stac=False), 0.3, rev=0.2)
    # worried soft strings under Mina (36.4-38.4)
    S.strings(36.6, 38.2, ["D3", "A3", "F4"], 0.12, att=0.8, rel=0.6, bright=1500)


# ------------------------------------------------------------------ s05 38-49
def s05(S):
    # glitchy chiptune motif: stutters, sticks, powers down (in Robo's pauses)
    def g(t, nt, d, v=0.4, **kw):
        S.put(t, chip(nt, d, v, **kw), RNG.uniform(-0.3, 0.3), rev=0.1)
    g(38.0, "F5", 0.1)
    g(38.12, "A5", 0.1)
    for i in range(4):
        g(38.24 + i * 0.045, "C6", 0.035, 0.38 - 0.05 * i)   # stuck stutter
    g(38.45, "C6", 0.12, 0.3, bend=-5)
    # 39.62-40.05 pause: stutter + glitch run
    for i in range(5):
        g(39.64 + i * 0.05, ["D6", "C6", "D6", "C6", "A5"][i], 0.04, 0.28, bits=3)
    g(39.9, "F5", 0.12, 0.25, bits=3, bend=-7)
    # 40.45-40.88 pause: falling "power down"
    g(40.46, "C6", 0.36, 0.28, bits=3, hold=10, bend=-12)
    # after line 41.35-42.45: motif tries again and fails
    for b, nt in [(0, "C5"), (0.2, "F5"), (0.35, "A5")]:
        g(41.4 + b, nt, 0.13, 0.3)
    for i in range(3):
        g(41.95 + i * 0.06, "G5", 0.04, 0.25, bits=3)
    g(42.15, "G5", 0.3, 0.25, bits=2, hold=14, bend=-19)
    # crushed bass drone underneath (D minor), soft
    n = int(4.4 * SR)
    t = tvec(n)
    dr = crush(sine(hz("D2"), n) * 0.7 + 0.3 * sine(hz("A2"), n), 4, 8) * env_adsr(n, 0.3, 0, 1, 0.6) * 0.07
    S.put(38.0, lp(dr, 1200), 0, rev=0.0)
    # 43.1 "hmph" stinger after 「もう！」
    S.put(43.1, reed("D3", 0.12, 0.45, squeak=0.7), 0.2, rev=0.15)
    S.put(43.1, pizz("D2", 0.5), -0.1, rev=0.1)
    S.put(43.12, snare(0.06, snappy=0.5, tone=230), 0.2, rev=0.1)
    # ---- 43.5 tender theme, 72 bpm: music box + warm strings + cello
    bt = 0.8333
    t0 = 43.5
    mel = [(0, "C5", 0.5, 0.5), (0.5, "F5", 0.5, 0.45), (1, "A5", 1.0, 0.5), (2, "G5", 0.5, 0.42),
           (2.5, "A5", 1.5, 0.5), (4, "C6", 0.5, 0.48), (4.5, "Bb5", 0.5, 0.42), (5, "A5", 0.5, 0.42),
           (5.5, "G5", 0.5, 0.4), (6, "F5", 0.6, 0.42)]
    S.seq(musicbox, mel, t0, bt, pan=0.15, rev=0.5, gain=0.7)
    S.seq(celesta, shift(mel, octave=-1, vel=0.25), t0, bt, pan=-0.2, rev=0.5)
    # harmony
    sw = (45.75, 5.0, 0.6)
    S.strings(43.5, 45.58, ["Bb2", "F3", "A3", "D4"], 0.26, att=0.9, rel=0.5, bright=2000)
    S.strings(45.58, 46.83, ["A2", "F3", "C4", "A4"], 0.27, att=0.25, rel=0.4, swell=sw, bright=2400)
    S.strings(46.83, 47.67, ["D3", "F3", "A3", "C4"], 0.25, att=0.2, rel=0.3, bright=2000)
    S.strings(47.67, 48.33, ["G2", "F3", "Bb3", "D4"], 0.25, att=0.2, rel=0.3, bright=2000)
    S.strings(48.33, 49.05, ["C3", "E3", "Bb3", "G4"], 0.25, att=0.2, rel=0.3, bright=2200)
    for t, nt in [(43.5, "Bb1"), (45.58, "A1"), (46.83, "D2"), (47.67, "G1"), (48.33, "C2")]:
        S.put(t, harp(nt, 0.35), -0.2, rev=0.4)
    # harp arpeggios (gentle)
    arps = [(43.5, ["Bb3", "D4", "F4", "A4"]), (45.58, ["A3", "C4", "F4", "A4", "C5"]),
            (46.83, ["D4", "F4", "A4"]), (47.67, ["G3", "Bb3", "D4"]), (48.33, ["C4", "E4", "G4", "Bb4"])]
    for t, notes in arps:
        for i, nt in enumerate(notes):
            S.put(t + 0.12 + i * 0.13, harp(nt, 0.2), 0.3 - 0.1 * i, rev=0.5)


# ------------------------------------------------------------------ s06 49-60
def s06(S):
    bt = 0.5
    # 49-51: rain stops -> light; F major, then D7 with harp gliss & riser into G at 51
    S.strings(49.0, 50.0, ["F3", "C4", "A4", "C5"], 0.3, att=0.15, rel=0.4, bright=2800)
    S.strings(50.0, 51.0, ["D3", "A3", "C4", "F#4", "A4"], 0.34, att=0.3, rel=0.1, bright=3200)
    S.put(49.0, harp("F2", 0.4), -0.2, rev=0.4)
    S.put(50.0, harp("D2", 0.45), -0.2, rev=0.4)
    for i, nt in enumerate(["F4", "A4", "C5", "F5", "A5", "C6"]):
        S.put(49.25 + i * 0.1, harp(nt, 0.18), -0.3 + 0.12 * i, rev=0.5)
    gl = ["D4", "F#4", "A4", "C5", "D5", "F#5", "A5", "C6", "D6", "F#6", "A6", "C7", "D7"]
    for i, nt in enumerate(gl):
        S.put(50.2 + i * 0.06, harp(nt, 0.16 + 0.01 * i), -0.6 + 1.2 * i / len(gl), rev=0.5)
    S.put(51.0 - 1.0, cymbal(0.25, swell=1.0), 0.0, rev=0.3)
    roll(lambda t, s, p: S.put(t, s, p, rev=0.15), 50.25, 50.98, 0.02, 0.14, stroke=0.03)
    for i in range(6):   # sun-ray shimmer (high sines with tremolo)
        nt = ["D6", "A6", "F#6", "D7", "A7", "F#7"][i]
        n = int(1.4 * SR)
        tt = tvec(n)
        y = sine(hz(nt), n) * (0.6 + 0.4 * np.sin(2 * np.pi * (5 + i) * tt)) * env_adsr(n, 0.5, 0, 1, 0.6) * 0.02
        S.put(49.7 + i * 0.12, y, -0.6 + 0.24 * i, rev=0.8)
    # ---- 51.0 G major tutti, motif restated (A+B on 51-55, A + B2 cadence on 55-59)
    t0 = 51.0
    G = 2
    mel = shift(MOTIF_A, G) + shift(MOTIF_B, G, boff=4) + shift(MOTIF_A, G, boff=8) + \
        [(12, tr("D6", G), 0.5, 0.6), (12.5, tr("C6", G), 0.5, 0.55), (13, tr("A5", G), 0.6, 0.55),
         (14.2, tr("G5", G), 0.5, 0.5)]  # ritard into the button
    mel = [m for m in mel if m[0] < 14.0]
    # the restatement under Robo's last line (55-58) is lighter so 「どういたしまして」 sits on top
    S.seq(whistle, [(b_, nt, d, v * (0.55 if b_ >= 9.5 else 1.0)) for b_, nt, d, v in mel], t0, bt, pan=0.05,
          rev=0.35, sustained=True)
    S.seq(glock, shift(mel, octave=1, vel=0.4), t0, bt, pan=-0.3, rev=0.35)
    S.seq(brass, shift(mel[:7], octave=-1, vel=0.6), t0, bt, pan=-0.15, rev=0.3, sustained=True)
    # chords per half bar: (beat, bass, strings, marimba)
    prog = [(0, "G2", ["G3", "D4", "B4"]), (2, "D3", ["G3", "D4", "B4"]),
            (4, "C3", ["C4", "E4", "G4"]), (6, "D3", ["D4", "F#4", "A4", "C5"]),
            (8, "G2", ["G3", "D4", "B4"]), (10, "E3", ["E3", "B3", "G4"]),
            (12, "C3", ["C4", "E4", "G4"]), (13, "D3", ["D4", "F#4", "A4", "C5"])]
    for k, (b, bass, up) in enumerate(prog):
        b1 = prog[k + 1][0] if k + 1 < len(prog) else 14.0
        S.strings(t0 + b * bt, t0 + b1 * bt, up + [tr(up[-1], 12)], 0.3, att=0.08, rel=0.3, bright=3400)
        S.put(t0 + b * bt, pizz(bass, 0.6), -0.25, rev=0.1)
        S.put(t0 + b * bt, tuba(tr(bass, -12), (b1 - b) * bt * 0.6, 0.3), 0)
        for j in range(int((b1 - b) * 2)):
            if j % 2 == 1:
                for nt in up:
                    S.put(t0 + (b + j * 0.5) * bt, marimba(tr(nt, 12), 0.1), 0.35, rev=0.2)
    # light drums: kick 1 & 3, snare/brush 2 & 4, shaker 8ths
    for i in range(14):
        tb = t0 + i * bt
        if i % 2 == 0:
            S.put(tb, kick(0.35), 0, rev=0.03)
        else:
            S.put(tb, snare(0.13, snappy=0.8), 0.15, rev=0.12)
        S.put(tb, shaker(0.07), 0.4, rev=0.05)
        S.put(tb + bt / 2, shaker(0.11), 0.4, rev=0.05)
    S.put(t0, cymbal(0.35, tau=1.8), 0.25, rev=0.3)
    S.put(t0, timpani("G2", 0.6), 0, rev=0.2)
    # rainbow shimmer: glock/celesta arpeggio cascades across the stereo field (52-55)
    rb = ["G6", "B6", "D7", "G7", "D7", "B6", "A6", "F#6"]
    for r in range(2):
        for i, nt in enumerate(rb):
            S.put(52.0 + r * 1.9 + i * 0.09, celesta(nt, 0.07), -0.8 + 1.6 * i / len(rb), rev=0.7)
    # ---- button ending: 58.1 V7 lead-in, G chord staccato at 59.0, pip (G6) at 59.4
    S.put(58.5, pizz("D3", 0.45), -0.2, rev=0.1)
    S.put(58.5, snare(0.08, snappy=0.6), 0.1, rev=0.1)
    tb = 59.0
    for j, nt in enumerate(["G2", "D3", "G3", "B3", "D4", "G4", "B4", "D5"]):
        S.put(tb, pizz(nt, 0.35, t60=0.25), -0.5 + 0.14 * j, rev=0.25)
    for j, nt in enumerate(["G4", "B4", "D5", "G5"]):
        S.put(tb, brass(nt, 0.12, 0.35), -0.3 + 0.2 * j, rev=0.25)
    S.put(tb, glock("G6", 0.35), 0.2, rev=0.3)
    S.put(tb, glock("B6", 0.25), -0.2, rev=0.3)
    S.put(tb, kick(0.4), 0, rev=0.0)
    S.put(tb, cymbal(0.12, tau=0.4), 0.2, rev=0.1)
    S.put(tb, tuba("G1", 0.12, 0.45), 0)
    S.put(59.4, beep("G6", 0.07, 0.11), 0.15, rev=0.15)


# =============================================================== render
def render():
    S = Score()
    for fn in (s01, s02, s03, s04, s05, s06):
        fn(S)
    hall = make_ir(t60=1.5, predelay=0.02, seed=31, lf=1.0, hf=0.5)
    dry = S.dry.out()
    send = S.send.out()
    pad = S.pad.out()
    pad = eq(pad, "peak", 300, -1.5, 0.8)
    wet = reverb(hp(send + 0.3 * pad, 200), hall, width=1.3)
    mix = dry + 0.9 * pad + 0.55 * wet
    mix = hp(mix, 30, 2)
    mix = compress(mix, -18, 1.8, 0.01, 0.2)
    mix *= undb(-19.0 - lufs(mix))
    for _ in range(2):
        mix = limiter(mix, -1.5)
    # button ending: everything faded out and silent by 60.0
    tt = tvec(N)
    mix *= np.where(tt < 59.45, 1.0, 0.5 + 0.5 * np.cos(np.pi * np.clip((tt - 59.45) / 0.5, 0, 1)))
    mix = fade(mix, 0.0, 0.02)
    return mix


def main():
    mix = render()
    report("music", mix)
    print(f"  LUFS={lufs(mix):.2f} TP={db(true_peak(mix)):.2f} dBTP")
    print(write_wav(BUILD / "music.wav", mix))


if __name__ == "__main__":
    main()
