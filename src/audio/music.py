"""Original score for 「ほしのおとしもの」 -> build/audio/music.wav (48 kHz stereo, 60.000 s).

Main motif (4 bars, yo-nuki major pentatonic, scale degrees 3-2-3-5 | 6-5-3-2 | 1-2-3-5-6 | 5):

    D major:  F#  E F#  A  |  B  A F#  E  |  D E F#  A  B  |  A ---
    (answer ends  ... F#  E  D)

Tempo map (bars placed so downbeats land on scene cuts / story beats):
  s01  0.0– 7.0   68.6 bpm (bar 3.5 s)  D major, Dmaj9 -> Gmaj7#11  celesta/music box, pad, gliss @2.2
  s02  7.0–13.0   80 bpm   (bar 3.0 s)  D major, curious pizzicato, playful marimba run when Haru jumps
  s03 13.0–25.0   80 bpm                theme: shakuhachi-like flute + marimba; turns sad (Em7, Bm7)
  s04 25.0–37.0  120 bpm   (bar 2.0 s)  driving: taiko, shaker, spiccato strings, bass eighths; G -> Gm tension
  s05 37.0–40.2  free                   hush (Bm add9 harmonics), "idea" sparkle
      40.2–48.0  61.5 bpm  (bar 3.9 s)  tender theme on solo strings + harp, pivot F#m7 -> ...
  s06 48.0–55.0  100 bpm   (bar 2.4 s)  modulates up to E major: IV-V-iii-vi -> I at the burst (52.8)
                                        full strings, choir "aah", taiko roll, cymbal swell, FM bells
  s07 55.0–60.0            E major      solo music box plays the motif + answer, final bell, fade to 0
"""
import numpy as np

from dsp import (SR, N, BUILD, Bus, hz, tvec, env_adsr, fade, curve, saw, sine, lp, hp, bp, svf,
                 pluck, partials, fm_bell, make_ir, reverb, limiter, lufs, true_peak, write_wav,
                 report, db, undb, eq)

RNG = np.random.default_rng(20240727)


# =============================================================== instruments
def celesta(note, vel=0.5):
    f = hz(note)
    T = float(np.clip(1.1 * (523.0 / f) ** 0.5, 0.35, 2.5))
    n = int(SR * min(T * 5 + 0.05, 6))
    y = partials(f, n, [(1, 1, T), (2, 0.16, T * 0.45), (3, 0.05, T * 0.3), (4, 0.13, T * 0.2),
                        (6.1, 0.03, T * 0.08)], RNG, att=0.002)
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


def pizz(note, vel=0.5, t60=0.7):
    y = pluck(hz(note), t60 + 0.3, t60=t60, damp=0.45, bright=3000, rng=RNG)
    return vel * lp(y, 3500)


def harp(note, vel=0.5):
    y = pluck(hz(note), 3.0, t60=2.2, damp=0.25, bright=6000, rng=RNG, pos=0.2)
    return vel * lp(y, 7000)


def flute(note, dur, vel=0.5, scoop=True):
    """Shakuhachi/flute-like lead: soft harmonics, breath noise, pitch scoop,
    delayed vibrato."""
    f0 = hz(note)
    n = int((dur + 0.25) * SR)
    t = tvec(n)
    vib_depth = 16 * np.clip((t - 0.3) / 0.5, 0, 1)
    cents = vib_depth * np.sin(2 * np.pi * 5.1 * t) + (-40 * np.exp(-t / 0.05) if scoop else 0)
    f = f0 * 2 ** (cents / 1200)
    ph = np.cumsum(f) / SR
    tone = (np.sin(2 * np.pi * ph) + 0.2 * np.sin(4 * np.pi * ph) + 0.06 * np.sin(6 * np.pi * ph)
            + 0.02 * np.sin(8 * np.pi * ph))
    env = env_adsr(n, a=0.07, d=0.2, s=0.85, r=0.2, hold=dur)
    env *= 1 + 0.12 * np.clip(t / max(dur, 0.1), 0, 1)
    nz = RNG.standard_normal(n)
    breath = 0.10 * bp(nz, f0 * 0.8, f0 * 1.3) + 0.025 * bp(nz, 2000, 7000)
    benv = env * (1 + 2.0 * np.exp(-t / 0.04))
    return vel * (tone * env + breath * benv)


def violin(note, dur, vel=0.5, att=0.18, rel=0.35):
    f0 = hz(note)
    n = int((dur + rel + 0.05) * SR)
    t = tvec(n)
    vib = 18 * np.clip((t - 0.2) / 0.4, 0, 1) * np.sin(2 * np.pi * 5.5 * t + RNG.uniform(0, 6))
    y = np.zeros(n)
    for dc in (-5, 0, 6):
        y += saw(f0 * 2 ** ((vib + dc) / 1200), n, RNG.uniform())
    y = lp(y, 3200, 2)
    y = eq(y, "peak", 2600, 3, 1.2)
    y = eq(y, "peak", 450, 2, 1.0)
    env = env_adsr(n, a=att, d=0.3, s=0.9, r=rel, hold=dur)
    return vel * 0.33 * y * env


def sub(note, dur, vel=0.5, att=0.3, rel=0.8):
    f = hz(note)
    n = int((dur + rel) * SR)
    y = sine(f, n) + 0.12 * sine(2 * f, n)
    return vel * y * env_adsr(n, a=att, d=0, s=1, r=rel, hold=dur)


def taiko(vel=0.8, f=62.0, decay=0.45):
    n = int(1.5 * SR)
    t = tvec(n)
    fp = f * (1 + 0.55 * np.exp(-t / 0.035))
    body = np.sin(2 * np.pi * np.cumsum(fp) / SR) * np.exp(-t / decay)
    nz = RNG.standard_normal(n)
    skin = lp(nz, 350) * np.exp(-t / 0.07) * 0.9
    click = bp(nz, 900, 3200) * np.exp(-t / 0.006) * 0.12
    return vel * fade(body + skin + click, 0.0008, 0.1)


def ka(vel=0.4):
    n = int(0.12 * SR)
    t = tvec(n)
    y = bp(RNG.standard_normal(n), 1800, 5000) * np.exp(-t / 0.012) + 0.4 * np.sin(2 * np.pi * 1650 * t) * np.exp(-t / 0.02)
    return vel * fade(y, 0.0005, 0.02)


def shaker(vel=0.3):
    n = int(0.12 * SR)
    t = tvec(n)
    y = bp(RNG.standard_normal(n), 5000, 12000)
    env = (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.03)
    return vel * fade(y * env, 0.001, 0.02)


def spiccato(note, vel=0.4):
    f0 = hz(note)
    n = int(0.28 * SR)
    y = saw(f0 * 2 ** (-6 / 1200), n, RNG.uniform()) + saw(f0 * 2 ** (6 / 1200), n, RNG.uniform())
    y = lp(y, 2600, 2)
    return vel * 0.5 * y * env_adsr(n, a=0.012, d=0.08, s=0.35, r=0.12, hold=0.1)


# =============================================================== the score
class Score:
    def __init__(self):
        self.dry = Bus()        # everything that is not pad/choir, pre-reverb
        self.send = Bus()       # reverb send
        self.pad = Bus()        # supersaw strings (filtered by an automation curve later)
        self.choir = {"a": Bus(), "u": Bus()}   # glottal sources, formant-filtered later
        self.mbox = Bus()       # sparkle layer with its own longer reverb

    def put(self, t, sig, pan=0.0, gain=1.0, rev=0.25):
        self.dry.add(t, sig, pan, gain)
        if rev > 0:
            self.send.add(t, sig, pan, gain * rev)

    def pad_chord(self, t0, t1, notes, vel=0.5, att=0.5, rel=0.9):
        dur = t1 - t0
        n = int((dur + rel) * SR)
        env = env_adsr(n, a=att, d=0, s=1, r=rel, hold=dur)
        t = tvec(n)
        for note in notes:
            f0 = hz(note)
            drift = 3 * np.sin(2 * np.pi * RNG.uniform(0.15, 0.35) * t + RNG.uniform(0, 6))
            for k, dc in enumerate((-13, -6, 0, 7, 12)):
                y = saw(f0 * 2 ** ((dc + drift) / 1200), n, RNG.uniform())
                pan = (-0.8, 0.5, 0.0, -0.4, 0.8)[k]
                self.pad.add(t0, y * env * vel * 0.09, pan)
            self.pad.add(t0, sine(f0, n) * env * vel * 0.12, 0.0)

    def choir_note(self, t0, t1, note, vel=0.4, vowel="a", att=0.6, rel=1.0, spread=0.7):
        dur = t1 - t0
        n = int((dur + rel) * SR)
        t = tvec(n)
        env = env_adsr(n, a=att, d=0, s=1, r=rel, hold=dur)
        f0 = hz(note)
        for k in range(4):
            rate = RNG.uniform(4.6, 5.8)
            depth = RNG.uniform(12, 22) * np.clip(t / 0.8, 0, 1)
            cents = RNG.uniform(-9, 9) + depth * np.sin(2 * np.pi * rate * t + RNG.uniform(0, 6))
            cents += 4 * np.sin(2 * np.pi * RNG.uniform(0.1, 0.3) * t)
            src = saw(f0 * 2 ** (cents / 1200), n, RNG.uniform())
            self.choir[vowel].add(t0, src * env * vel * 0.25, spread * (-1 + 2 * k / 3))

    def mb(self, t, sig, pan=0.0, gain=1.0):
        self.mbox.add(t, sig, pan, gain)


# formant tables (Hz, dB, bandwidth) — soprano "a", and a soft "u/oo"
FORMANTS = {
    "a": [(800, 0, 80), (1150, -6, 90), (2900, -32, 120), (3900, -20, 130), (4950, -50, 140)],
    "u": [(350, 0, 50), (600, -20, 60), (2700, -30, 170), (2900, -36, 180), (3300, -42, 200)],
}


def formant_filter(x, vowel):
    y = np.zeros_like(x)
    for f, g, bw in FORMANTS[vowel]:
        y += undb(g) * bp(x, f - bw, f + bw, 1) * 2.0
    return y


def seq(score, fn, events, t0, beat, pan=0.0, gain=1.0, rev=0.25, target="dry", **kw):
    """events: [(beat_offset, note, [dur_beats], [vel])...]"""
    for ev in events:
        b, note = ev[0], ev[1]
        dur = ev[2] if len(ev) > 2 else 1
        vel = ev[3] if len(ev) > 3 else 0.5
        t = t0 + b * beat
        sig = fn(note, dur * beat, vel, **kw) if fn in (flute, violin, sub) else fn(note, vel, **kw)
        p = pan(b) if callable(pan) else pan
        if target == "mb":
            score.mb(t, sig, p, gain)
        else:
            score.put(t, sig, p, gain, rev)


# ------------------------------------------------------------------ s01
def s01(S):
    bt = 0.875
    S.pad_chord(0.0, 3.5, ["D3", "A3", "E4", "F#4"], vel=0.55, att=2.2, rel=1.2)
    S.pad_chord(3.5, 7.0, ["G3", "D4", "F#4", "B4", "C#5"], vel=0.5, att=1.2, rel=1.0)
    S.put(0.3, sub("D2", 3.3, 0.35, att=1.5), 0, rev=0.1)
    S.put(3.5, sub("G1", 3.6, 0.35, att=0.8), 0, rev=0.1)
    # high string-harmonic shimmer
    n = int(7.5 * SR)
    t = tvec(n)
    sh = (sine(hz("A6"), n) + 0.6 * sine(hz("E7"), n)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t))
    S.put(0.0, sh * env_adsr(n, 2.5, 0, 1, 1.5) * 0.018, 0.2, rev=0.8)
    # music-box arpeggio, bar 1 (eighths)
    arp1 = ["A4", "D5", "E5", "F#5", "A5", "F#5", "E5", "D5"]
    for i, nt in enumerate(arp1[:4] + ["A5"]):
        S.mb(0.45 + i * bt / 2, musicbox(nt, 0.10 + 0.02 * i), -0.4 + 0.2 * i)
    # shooting star glissando: rising pentatonic run 2.2 -> 2.9, left to right
    gl = ["D5", "E5", "F#5", "A5", "B5", "D6", "E6", "F#6", "A6", "B6", "D7"]
    for i, nt in enumerate(gl):
        S.mb(2.2 + i * 0.066, celesta(nt, 0.10 + 0.022 * i), -0.8 + 1.6 * i / len(gl))
    # sputter & dive: stumbling descending twinkles
    for i, (dt, nt) in enumerate([(3.0, "B6"), (3.12, "A6"), (3.3, "F#6"), (3.37, "E6"), (3.62, "D6")]):
        S.mb(dt, celesta(nt, 0.2 - 0.03 * i), 0.7 - 0.25 * i)
    # bar 2: celesta states the motif head over Gmaj7#11 (title card)
    seq(S, celesta, [(0, "F#5", 1, 0.32), (1, "E5", 0.5, 0.26), (1.5, "F#5", 0.5, 0.28), (2, "A5", 2, 0.34)],
        3.5, bt, pan=-0.1, rev=0.5)
    seq(S, celesta, [(0, "F#4", 1, 0.12), (2, "D4", 1, 0.1)], 3.5, bt, pan=0.2, rev=0.5)
    arp2 = ["B4", "D5", "F#5", "C#6", "A5", "F#5", "D5", "B4"]
    for i, nt in enumerate(arp2):
        S.mb(3.5 + 0.44 + i * bt / 2, musicbox(nt, 0.08), 0.5 - 0.12 * i)
    S.choir_note(3.6, 7.0, "B4", 0.18, "u", att=1.5, rel=1.2)
    S.choir_note(3.6, 7.0, "F#5", 0.14, "u", att=1.8, rel=1.2)


# ------------------------------------------------------------------ s02
def s02(S):
    bt, b1, b2 = 0.75, 7.0, 10.0
    S.pad_chord(b1, b2, ["D3", "A3", "B3", "F#4"], vel=0.42, att=0.5)
    S.pad_chord(b2, 11.5, ["G3", "B3", "D4", "A4"], vel=0.42, att=0.3)
    S.pad_chord(11.5, 12.25, ["A3", "D4", "E4"], vel=0.42, att=0.2, rel=0.3)
    S.pad_chord(12.25, 13.0, ["A3", "C#4", "E4", "G4"], vel=0.42, att=0.15, rel=0.4)
    # curious pizzicato
    seq(S, pizz, [(0, "D3", 1, 0.55), (1.5, "A3", 1, 0.4), (3, "F#3", 1, 0.4)], b1, bt, pan=-0.3)
    seq(S, pizz, [(0, "D4", 1, 0.3), (2, "E4", 1, 0.25), (3.5, "A3", 1, 0.3)], b1, bt, pan=0.35)
    seq(S, pizz, [(0, "G2", 1, 0.55), (1, "D3", 1, 0.45), (2, "A2", 1, 0.5), (3, "A2", 1, 0.5)], b2, bt, pan=-0.3)
    # a tiny sparkle answering the "pon" in the garden
    S.mb(8.02, celesta("A6", 0.16), 0.3)
    S.mb(8.2, celesta("D7", 0.1), 0.4)
    # Haru leaps up: playful marimba run
    run = ["D5", "E5", "F#5", "A5", "B5", "A5", "B5", "D6"]
    seq(S, marimba, [(i * 0.5, nt, 0.5, 0.35 + 0.03 * i) for i, nt in enumerate(run)], b2, bt,
        pan=lambda b: -0.4 + 0.2 * b, rev=0.2)
    seq(S, marimba, [(3.5, "E5", 0.25, 0.3), (3.75, "F#5", 0.25, 0.35)], b2, bt, pan=0.3, rev=0.2)
    S.put(b1, sub("D2", 3.0, 0.3), 0, rev=0.05)
    S.put(b2, sub("G1", 1.5, 0.3), 0, rev=0.05)
    S.put(11.5, sub("A1", 1.5, 0.3), 0, rev=0.05)


# ------------------------------------------------------------------ s03
def s03(S):
    bt = 0.75
    bars = [13.0, 16.0, 19.0, 22.0]
    S.pad_chord(13.0, 16.0, ["D3", "F#3", "A3", "E4"], vel=0.45)
    S.pad_chord(16.0, 19.0, ["G3", "B3", "D4", "F#4"], vel=0.45)
    S.pad_chord(19.0, 22.0, ["E3", "G3", "B3", "D4"], vel=0.48, att=0.8)
    S.pad_chord(22.0, 23.5, ["B2", "D3", "F#3", "A3"], vel=0.48)
    S.pad_chord(23.5, 24.25, ["A2", "D3", "E3", "A3"], vel=0.45, att=0.3, rel=0.4)
    S.pad_chord(24.25, 25.0, ["A2", "C#3", "E3", "G3"], vel=0.45, att=0.2, rel=0.35)
    # pizz bass
    for b0, (r, f5) in zip(bars, [("D3", "A2"), ("G2", "D3"), ("E2", "B2"), ("B2", "F#2")]):
        seq(S, pizz, [(0, r, 1, 0.55), (2, f5, 1, 0.45)], b0, bt, pan=-0.25)
    seq(S, pizz, [(0, "A2", 1, 0.5), (1, "A2", 1, 0.4)], 23.5, bt, pan=-0.25)
    for b0, (r, f5) in zip(bars, [("D2", "D2"), ("G1", "G1"), ("E2", "E2"), ("B1", "B1")]):
        S.put(b0, sub(r, 2.9, 0.26), 0, rev=0.05)
    # playful marimba broken chords (bars 1-2)
    for b0, ch in [(13.0, ["D4", "A4", "F#4", "A4", "D5", "A4", "F#4", "A4"]),
                   (16.0, ["G4", "D5", "B4", "D5", "G5", "D5", "B4", "D5"])]:
        seq(S, marimba, [(i * 0.5, nt, 0.5, 0.2 if i % 2 else 0.27) for i, nt in enumerate(ch)], b0, bt,
            pan=0.35, rev=0.2)
    # harp, sadder (bars 3-4)
    for b0, ch in [(19.0, ["E3", "B3", "D4", "G4", "B4", "D5"]), (22.0, ["B2", "F#3", "A3", "D4", "F#4", "A4"])]:
        seq(S, harp, [(i * 0.5, nt, 1, 0.3) for i, nt in enumerate(ch)], b0, bt, pan=0.3, rev=0.4)
    # THEME on flute (bars 1-2)
    seq(S, flute, [(0, "F#5", 0.95, 0.30), (1, "E5", 0.45, 0.26), (1.5, "F#5", 0.45, 0.27), (2, "A5", 1.9, 0.30)],
        13.0, bt, pan=-0.05, rev=0.35)
    seq(S, flute, [(0, "B5", 0.95, 0.30), (1, "A5", 0.45, 0.26), (1.5, "F#5", 0.45, 0.26), (2, "E5", 1.9, 0.28)],
        16.0, bt, pan=-0.05, rev=0.35)
    # bars 3-4 (Kira: "...or I'll disappear") theme moves to soft, darker strings
    seq(S, violin, [(0, "D5", 1, 0.25), (1, "E5", 0.5, 0.22), (1.5, "F#5", 0.5, 0.23), (2, "A5", 1, 0.25),
                    (3, "B5", 1, 0.25)], 19.0, bt, pan=-0.1, rev=0.45)
    seq(S, violin, [(0, "A5", 2.8, 0.24), (3, "G5", 0.9, 0.18), (4, "F#5", 1, 0.18), (5, "E5", 1, 0.2)],
        22.0, bt, pan=-0.1, rev=0.45)


# ------------------------------------------------------------------ s04
def s04(S):
    bt = 0.5
    bars = [25.0, 27.0, 29.0, 31.0, 33.0, 35.0]
    chords = [(25.0, 27.0, ["D3", "A3", "D4", "F#4"], "D2"),
              (27.0, 29.0, ["G3", "D4", "F#4", "B4"], "G1"),
              (29.0, 30.0, ["E3", "B3", "D4", "G4"], "E2"),
              (30.0, 31.0, ["A3", "C#4", "E4", "A4"], "A1"),
              (31.0, 33.0, ["B3", "D4", "F#4", "B4"], "B1"),
              (33.0, 34.0, ["G3", "B3", "D4", "G4"], "G1"),
              (34.0, 35.0, ["G3", "Bb3", "D4", "G4"], "G1"),
              (35.0, 36.0, ["A3", "D4", "E4", "A4"], "A1"),
              (36.0, 36.95, ["A3", "C#4", "E4", "A4"], "A1")]
    for t0, t1, ch, bass in chords:
        S.pad_chord(t0, t1, ch, vel=0.5, att=0.12, rel=0.25 if t1 > 36.9 else 0.35)
        # driving bass eighths
        k = 0
        tt = t0
        while tt < t1 - 1e-6:
            nt = bass if k % 4 != 3 else bass[:-1] + str(int(bass[-1]) + 1)
            S.put(tt, pizz(nt, 0.62 if k % 2 == 0 else 0.45, t60=0.35), -0.15, rev=0.05)
            tt += bt / 2
            k += 1
        S.put(t0, sub(bass, t1 - t0 - 0.1, 0.3, att=0.02, rel=0.1), 0, rev=0.0)
        # spiccato string ostinato (eighths on upper chord tones)
        up = ch[1:]
        pat = [0, 1, 2, 1]
        tt, k = t0, 0
        while tt < t1 - 1e-6:
            S.put(tt, spiccato(up[pat[k % 4]] if pat[k % 4] < len(up) else up[-1], 0.4 if k % 2 == 0 else 0.28),
                  0.45 if k % 2 else -0.45, rev=0.2)
            tt += bt / 2
            k += 1
    # percussion
    for i, b0 in enumerate(bars):
        build = i >= 4
        # taiko "don . . don | don . don don"
        for b, v, f in [(0, 0.9, 60), (1.5, 0.45, 70), (2, 0.75, 60), (3, 0.55, 66), (3.5, 0.45, 70)]:
            S.put(b0 + b * bt, taiko(v * (1.1 if build else 1.0), f), -0.1, rev=0.25)
        for b in (1, 3):
            S.put(b0 + b * bt, ka(0.35), 0.3, rev=0.2)
        for k in range(16):
            v = 0.26 if k % 2 else 0.16
            S.put(b0 + k * bt / 4, shaker(v * (1.0 + 0.1 * i)), 0.55, rev=0.1)
    # final-bar taiko roll crescendo into the hush
    for k in range(14):
        t = 35.0 + k * 0.135
        S.put(t, taiko(0.25 + 0.05 * k, 72), 0.1 if k % 2 else -0.2, rev=0.25)
    # riser
    n = int(3.95 * SR)
    t = tvec(n)
    nz = RNG.standard_normal((2, n))
    fc = 300 * (6000 / 300) ** (t / t[-1])
    rs = svf(nz, fc, 1.5, "bp") * (t / t[-1]) ** 2.2
    S.put(33.0, fade(rs, 0.05, 0.05) * 0.12, rev=0.4)
    # melody: flute doubled by marimba an octave lower
    mel = [(0, "F#5", 1), (1, "E5", 0.5), (1.5, "F#5", 0.5), (2, "A5", 2),
           (4, "B5", 1), (5, "A5", 0.5), (5.5, "F#5", 0.5), (6, "E5", 2),
           (8, "D5", 1), (9, "E5", 0.5), (9.5, "F#5", 0.5), (10, "A5", 1), (11, "B5", 1),
           (12, "A5", 3.8),
           (16, "B5", 1), (17, "A5", 0.5), (17.5, "B5", 0.5), (18, "D6", 2),
           (20, "D6", 1), (21, "E6", 2.85)]
    seq(S, flute, [(b, nt, d * 0.95, 0.30) for b, nt, d in mel], 25.0, bt, pan=-0.05, rev=0.3, scoop=False)
    seq(S, marimba, [(b, nt[:-1] + str(int(nt[-1]) - 1), 1, 0.2) for b, nt, d in mel], 25.0, bt, pan=0.25, rev=0.2)


# ------------------------------------------------------------------ s05
def s05(S):
    # hush: Bm(add9) harmonics
    S.pad_chord(37.05, 40.2, ["B3", "D4", "F#4", "C#5"], vel=0.2, att=1.0, rel=0.8)
    n = int(3.6 * SR)
    t = tvec(n)
    har = (sine(hz("F#6"), n) + 0.5 * sine(hz("B6"), n)) * (0.7 + 0.3 * np.sin(2 * np.pi * 2.3 * t))
    S.put(37.1, har * env_adsr(n, 1.2, 0, 1, 1.0) * 0.012, 0.1, rev=0.9)
    # the idea: rising sparkle
    for dt, nt, p in [(39.3, "A5", -0.2), (39.45, "D6", 0.1), (39.6, "F#6", 0.3), (39.75, "A6", 0.5)]:
        S.mb(dt, celesta(nt, 0.22), p)
    # tender theme from 40.2 (bar 3.9 s)
    bt = 0.975
    S.pad_chord(40.2, 42.15, ["D3", "A3", "D4", "F#4"], vel=0.42, att=1.2)
    S.pad_chord(42.15, 44.1, ["F#3", "A3", "D4", "A4"], vel=0.46, att=0.6)
    S.pad_chord(44.1, 46.05, ["G3", "B3", "D4", "F#4"], vel=0.5, att=0.6)
    S.pad_chord(46.05, 48.05, ["F#3", "A3", "C#4", "E4"], vel=0.55, att=0.6, rel=0.5)
    for t0, t1, nt in [(40.2, 42.15, "D2"), (42.15, 44.1, "F#2"), (44.1, 46.05, "G2"), (46.05, 48.0, "F#2")]:
        S.put(t0, sub(nt, t1 - t0, 0.34, att=0.4, rel=0.5), 0, rev=0.05)
    arps = [(40.2, ["D3", "A3", "D4", "F#4", "A4", "D5", "F#5", "A5"]),
            (42.15, ["F#3", "A3", "D4", "F#4", "A4", "D5", "E5", "F#5"]),
            (44.1, ["G3", "D4", "F#4", "B4", "D5", "F#5", "A5", "B5"]),
            (46.05, ["F#3", "C#4", "E4", "A4", "C#5", "E5", "F#5", "A5"])]
    for t0, ch in arps:
        seq(S, harp, [(i * 0.25, nt, 1, 0.3) for i, nt in enumerate(ch)], t0, bt, pan=0.35, rev=0.45)
    seq(S, violin, [(0, "F#5", 1, 0.3), (1, "E5", 0.5, 0.27), (1.5, "F#5", 0.5, 0.28), (2, "A5", 2, 0.32),
                    (4, "B5", 1, 0.33), (5, "A5", 0.5, 0.3), (5.5, "F#5", 0.5, 0.3), (6, "E5", 1, 0.32),
                    (7, "C#5", 0.5, 0.3), (7.5, "E5", 0.5, 0.34)], 40.2, bt, pan=-0.1, rev=0.45)
    seq(S, violin, [(0, "F#4", 2, 0.16), (2, "A4", 2, 0.16), (4, "D5", 2, 0.18), (6, "C#5", 2, 0.2)],
        40.2, bt, pan=0.25, rev=0.45, att=0.5)


# ------------------------------------------------------------------ s06
def s06(S):
    bt = 0.6
    ch = [(48.0, 49.2, ["A3", "C#4", "E4", "B4"], "A1"),
          (49.2, 50.4, ["B3", "D#4", "F#4", "B4"], "B1"),
          (50.4, 51.6, ["G#3", "B3", "D#4", "F#4"], "G#1"),
          (51.6, 52.8, ["C#4", "E4", "G#4", "B4"], "C#2"),
          (52.8, 55.0, ["E3", "B3", "E4", "G#4", "B4", "F#5"], "E1")]
    for t0, t1, notes, bass in ch:
        last = t1 >= 55
        S.pad_chord(t0, t1, notes, vel=0.75 if not last else 0.9, att=0.25, rel=1.6 if last else 0.3)
        S.put(t0, sub(bass, t1 - t0, 0.55, att=0.05, rel=1.2 if last else 0.2), 0, rev=0.05)
        S.put(t0, sub(bass[:-1] + str(int(bass[-1]) + 1), t1 - t0, 0.3, att=0.05, rel=1.2 if last else 0.2), 0, rev=0.1)
        for nt in (notes[2:5] if last else notes[-3:]):
            S.choir_note(t0, t1, nt[:-1] + str(int(nt[-1]) + 1), 0.5 if not last else 0.62, "a",
                         att=0.35, rel=1.5 if last else 0.35)
    # melody: strings in octaves
    mel = [(0, "G#5", 1), (1, "F#5", 0.5), (1.5, "G#5", 0.5), (2, "B5", 2),
           (4, "C#6", 1), (5, "B5", 0.5), (5.5, "G#5", 0.5), (6, "F#5", 1), (7, "G#5", 0.5), (7.5, "B5", 0.5),
           (8, "E6", 3.6)]
    seq(S, violin, [(b, nt, d, 0.5) for b, nt, d in mel], 48.0, bt, pan=-0.1, rev=0.45, att=0.08, rel=0.4)
    seq(S, violin, [(b, nt[:-1] + str(int(nt[-1]) - 1), d, 0.34) for b, nt, d in mel], 48.0, bt, pan=0.2,
        rev=0.45, att=0.08, rel=0.4)
    seq(S, flute, [(8, "B6", 3.4, 0.12)], 48.0, bt, pan=0.0, rev=0.6, scoop=False)
    # percussion: big hits, roll into the burst
    for t, v in [(48.0, 0.9), (49.2, 0.8), (50.4, 0.9), (51.0, 0.6)]:
        S.put(t, taiko(v, 58, 0.6), -0.1, rev=0.35)
    for k in range(16):
        S.put(51.6 + k * 0.075, taiko(0.2 + 0.04 * k, 70, 0.25), 0.15 if k % 2 else -0.15, rev=0.3)
    S.put(52.8, taiko(1.25, 52, 1.0), 0, rev=0.5)
    # cymbal swell peaking at the burst
    n = int(4.2 * SR)
    t = tvec(n)
    up = 1.2
    env = np.where(t < up, (t / up) ** 3, np.exp(-(t - up) / 0.9))
    cy = hp(lp(RNG.standard_normal((2, n)), 13000), 3500) * env
    S.put(52.8 - up, fade(cy, 0.01, 0.2) * 0.13, rev=0.5)
    # harp glissando up to the peak
    gl = ["E4", "F#4", "G#4", "B4", "C#5", "E5", "F#5", "G#5", "B5", "C#6", "E6", "F#6", "G#6", "B6"]
    for i, nt in enumerate(gl):
        S.put(51.95 + i * 0.058, harp(nt, 0.3), -0.6 + 1.2 * i / len(gl), rev=0.5)
    # FM bells + glockenspiel sparkle cascade at the burst
    for dt, nt, p in [(0, "E5", -0.3), (0.09, "B5", 0.3), (0.18, "E6", -0.1), (0.3, "G#6", 0.4)]:
        S.put(52.8 + dt, fm_bell(hz(nt), 4.5, tau=1.8, rng=RNG) * 0.14, p, rev=0.6)
    casc = ["E7", "C#7", "B6", "G#6", "F#6", "E6", "C#6", "B5"]
    for i, nt in enumerate(casc):
        S.mb(53.1 + i * 0.16, celesta(nt, 0.2), 0.6 - 0.15 * i)


# ------------------------------------------------------------------ s07
def s07(S):
    S.pad_chord(55.0, 57.9, ["E3", "B3", "G#4"], vel=0.35, att=0.8, rel=1.0)
    S.pad_chord(57.9, 59.6, ["E3", "B3", "F#4", "G#4"], vel=0.28, att=0.6, rel=0.4)
    S.put(55.0, sub("E2", 4.3, 0.28, att=0.8, rel=0.6), 0, rev=0.05)
    bt = 0.45
    mel = [(0, "G#5", 1), (1, "F#5", 0.5), (1.5, "G#5", 0.5), (2, "B5", 2),
           (4, "C#6", 1), (5, "B5", 0.5), (5.5, "G#5", 0.5), (6, "F#5", 1), (7, "E5", 2)]
    for b, nt, d in mel:
        S.mb(55.2 + b * bt, musicbox(nt, 0.3), -0.1 + 0.03 * b)
    for b, nt in [(0, "E4"), (2, "B4"), (4, "A4"), (6, "B4"), (7, "E4")]:
        S.mb(55.2 + b * bt, musicbox(nt, 0.12), 0.2)
    S.put(55.2 + 7 * bt, fm_bell(hz("E5"), 4.0, index=1.6, tau=1.5, rng=RNG) * 0.12, 0.0, rev=0.6)
    S.put(55.2 + 7 * bt, fm_bell(hz("E4"), 4.0, index=1.2, tau=1.8, rng=RNG) * 0.06, 0.0, rev=0.6)


def render():
    S = Score()
    for fn in (s01, s02, s03, s04, s05, s06, s07):
        fn(S)
    # pad: automated low-pass (the "brightness" of the story arc)
    cut = curve([(0, 900), (7, 1300), (13, 1600), (19, 1300), (25, 2000), (33, 2300), (36.9, 3200),
                 (37.1, 700), (40.2, 900), (44, 1700), (48, 2400), (52.8, 5200), (55, 2200), (60, 1100)],
                S.pad.x.shape[1])
    pad = svf(S.pad.x, cut, 0.8, "lp")
    pad = svf(pad, cut * 1.6, 0.6, "lp")
    pad = hp(pad, 55)
    choir = sum(hp(formant_filter(S.choir[v].x, v), 150) for v in S.choir)
    mb = hp(S.mbox.x, 200)
    # conductor's dynamics (dB) — the overall arc of the story
    dyn = undb(curve([(0, -7), (7.0, -7), (7.6, -4.5), (13, -4.5), (13.6, -3), (24.6, -3), (25.0, 0),
                      (37.0, 0), (37.2, -1.5), (40.2, -1.5), (47.5, 0), (48.0, 1.0), (55.0, 1.0),
                      (55.9, -5.5), (60, -5.5)], S.dry.x.shape[1]))
    dry = (S.dry.x + pad + choir + mb) * dyn
    send = (S.send.x + 0.45 * pad + 0.6 * choir + 0.6 * mb) * dyn
    hall = make_ir(t60=2.8, predelay=0.03, seed=11, lf=1.2, hf=0.5)
    wet = reverb(hp(send, 180), hall, width=1.3)
    mix = dry + 0.55 * wet
    mix = hp(mix, 28)
    mix = mix[:, :N]
    # final fade (the music already decays; this guarantees a clean 0 at 60.000 s)
    fl = int(1.0 * SR)
    mix[:, -fl:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(fl) / fl)
    mix -= mix.mean(axis=1, keepdims=True)
    # stem level: integrated -20 LUFS, TP < -1.5
    mix *= undb(-20.0 - lufs(mix))
    if db(true_peak(mix)) > -1.5:
        mix = limiter(mix, -1.6)
    return mix


def main():
    mix = render()
    report("music", mix)
    print(f"  LUFS={lufs(mix):.2f} TP={db(true_peak(mix)):.2f} dBTP")
    print(write_wav(BUILD / "music.wav", mix))


if __name__ == "__main__":
    main()
