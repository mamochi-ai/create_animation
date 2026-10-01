"""Synthesized sound effects + ambience for 「あめふりロボ」 -> build/audio/sfx.wav
(48 kHz stereo, 60.000 s).  No samples: everything below is noise, oscillators and filters.

Ambience
  * rain bed 0-49.5 s: low "roar" + bright hiss (decorrelated L/R, slow gusts), thousands of
    individual drop splashes (Poisson, 6 resonance classes per channel), gutter trickle
    (Minnaert-bubble stream, left), awning drips 6-27 s.  Heavy -> lighter -> stops at 49.5.
  * rain drumming on the umbrella canopy 31.0-49.5: dense dull fabric "tup"s (low-passed
    taps + 230 Hz membrane resonance) overhead, rib-tip drips; leather "tok"s on the
    randoseru 43.6-49.5.
  * last drips 49.5-57, birds 52-60.
Cues (story/timeline.json "sfx"), robot beeps tuned to the score's key (F major, G major in s06).
"""
import _common  # noqa: F401
import numpy as np
import scipy.signal as ss

from dsp import (SR, N, BUILD, Bus, tvec, env_adsr, fade, curve, sine, saw, lp, hp, bp, eq, svf,
                 make_ir, reverb, write_wav, report, db, undb, load_timeline, hz)

RNG = np.random.default_rng(4242)
TT = tvec(N)


# ================================================================ helpers
def impulses(rate, amp_scale=1.0, n=N):
    """Poisson impulse train with time-varying rate (per second, array or scalar)."""
    p = np.broadcast_to(np.asarray(rate, dtype=float), (n,)) / SR
    hit = RNG.random(n) < p
    x = np.zeros(n)
    k = int(hit.sum())
    x[hit] = RNG.exponential(1.0, k) * RNG.choice([-1, 1], k)
    return x * amp_scale


def conv_classes(imp, kernels):
    """Spread an impulse train randomly over kernel classes and convolve."""
    cls = RNG.integers(0, len(kernels), len(imp))
    y = np.zeros(len(imp))
    for c, k in enumerate(kernels):
        y += ss.oaconvolve(np.where(cls == c, imp, 0.0), k)[:len(imp)]
    return y


def slow_noise(n, fc):
    """Unit-std low-passed noise (for gusts / modulations)."""
    x = lp(RNG.standard_normal(n), fc, 2)
    return x / (np.std(x) + 1e-12)


def kernel_splash(fc, tau, n_ms=40):
    n = int(n_ms / 1000 * SR)
    t = tvec(n)
    nz = bp(RNG.standard_normal(n), fc * 0.6, min(fc * 1.8, 20000))
    return fade(nz * np.exp(-t / tau), 0.0003, 0.003)


def plip(f0=900.0, amp=0.3, tau=0.025, rise=1.0):
    """Water drop into water: Minnaert bubble, sine with an upward pitch sweep."""
    n = int(0.12 * SR)
    t = tvec(n)
    f = f0 * (1 + rise * (1 - np.exp(-t / 0.03)))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / tau)
    y += 0.25 * hp(RNG.standard_normal(n), 3000) * np.exp(-t / 0.002)
    return amp * fade(y, 0.0005, 0.01)


def bubble_stream(rate_curve, fmin, fmax, amp):
    y = np.zeros(N + SR)
    p = rate_curve / SR
    idx = np.where(RNG.random(N) < p)[0]
    for i in idx:
        f0 = np.exp(RNG.uniform(np.log(fmin), np.log(fmax)))
        b = plip(f0, RNG.uniform(0.2, 1.0), tau=RNG.uniform(0.006, 0.02), rise=RNG.uniform(0.3, 1.2))
        y[i:i + len(b)] += b
    return amp * y[:N]


def thump(amp, f=80, tau=0.08):
    n = int(0.4 * SR)
    t = tvec(n)
    y = np.sin(2 * np.pi * np.cumsum(f * (1 + 0.6 * np.exp(-t / 0.02))) / SR) * np.exp(-t / tau)
    y += 0.5 * lp(RNG.standard_normal(n), 300) * np.exp(-t / 0.03)
    return amp * fade(y, 0.001, 0.05)


def clack(amp, pings=(2300, 3500), tau=0.05, low=110):
    """Metal latch / clamp: knock + bright burst + inharmonic ringing."""
    n = int(0.35 * SR)
    t = tvec(n)
    nz = RNG.standard_normal(n)
    y = bp(nz, 1500, 9000) * np.exp(-t / 0.004)
    y += 0.6 * np.sin(2 * np.pi * low * t) * np.exp(-t / 0.025)
    for k, f in enumerate(pings):
        y += 0.25 / (k + 1) * np.sin(2 * np.pi * f * t + RNG.uniform(0, 6)) * np.exp(-t / tau)
    return amp * fade(y, 0.0004, 0.05)


def servo(dur, f0, f1, amp, grit=0.3):
    """Small motor whine: saw with a pitch glide, nasal band-pass, gear grit."""
    n = int(dur * SR)
    t = tvec(n)
    x = t / dur
    f = f0 + (f1 - f0) * (0.5 - 0.5 * np.cos(np.pi * x))
    s = saw(f, n) + 0.5 * saw(2.01 * f, n, 0.3)
    s = bp(s, 500, 4000)
    s = eq(s, "peak", 1800, 6, 2)
    g = grit * bp(RNG.standard_normal(n), 2000, 8000) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * np.cumsum(f / 6) / SR)))
    env = env_adsr(n, 0.03, 0, 1, 0.06)
    return amp * fade((s * 0.5 + g * 0.3) * env, 0.003, 0.01)


def squeak(dur=0.09, f=2000.0, amp=0.3):
    """Rubber tread squeak: sine with fast stick-slip vibrato, saturated, band-limited."""
    n = int(dur * SR)
    t = tvec(n)
    ff = f * (0.92 + 0.16 * t / dur) * (1 + 0.04 * np.sin(2 * np.pi * 38 * t))
    y = np.tanh(2.5 * np.sin(2 * np.pi * np.cumsum(ff) / SR))
    y = bp(y, f * 0.7, f * 3.5)
    env = np.sin(np.pi * t / dur) ** 0.8
    return amp * fade(y * env, 0.002, 0.005)


def beep(f, dur=0.07, amp=0.2, sq=0.15):
    n = int((dur + 0.01) * SR)
    t = tvec(n)
    y = np.sin(2 * np.pi * f * t) + sq * np.sign(np.sin(2 * np.pi * f * t))
    y = lp(y, 7000)
    return amp * y * env_adsr(n, 0.003, 0, 1, 0.012, hold=dur - 0.003)


def chirp_beep(f0, f1, dur, amp):
    n = int(dur * SR)
    t = tvec(n)
    f = f0 * (f1 / f0) ** (t / dur) ** 1.5
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.1 * np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR))
    return amp * lp(y, 7000) * env_adsr(n, 0.005, 0, 1, 0.02)


def noise_sweep(dur, f0, f1, q=1.2, st=True):
    n = int(dur * SR)
    t = tvec(n)
    fc = f0 * (f1 / f0) ** (t / t[-1])
    nz = RNG.standard_normal((2, n) if st else n)
    return svf(nz, fc, q, "bp")


def rustle(dur, amp, lo=700, hi=6000, grains=45):
    n = int(dur * SR)
    t = tvec(n)
    nz = bp(RNG.standard_normal(n), lo, hi)
    g = np.zeros(n)
    for _ in range(int(dur * grains)):
        c = RNG.uniform(0, dur)
        w = RNG.uniform(0.005, 0.03)
        g += RNG.uniform(0.2, 1) * np.exp(-0.5 * ((t - c) / w) ** 2)
    y = fade(nz * g * env_adsr(n, 0.02, 0, 1, dur * 0.4), 0.003, 0.02)
    return amp * y / np.max(np.abs(y))


def crushed(x, bits=4, hold=12):
    y = np.repeat(x[::hold], hold)[:len(x)]
    q = 2 ** (bits - 1)
    return np.round(np.clip(y, -1, 1) * q) / q


# ================================================================ creature sounds
def mew(dur=0.55, f0=((0, 620), (0.12, 980), (0.3, 920), (1.0, 600)),
        F1=((0, 450), (0.25, 1100), (0.7, 900), (1.0, 550)),
        F2=((0, 2000), (0.25, 1700), (0.7, 1300), (1.0, 1000)), amp=0.3):
    """Kitten meow: harmonic source with an f0 contour, time-varying formants (m-i-a-ow),
    nasal onset, a little vibrato/jitter and breath."""
    n = int(dur * SR)
    t = tvec(n)
    x = t / dur
    k = lambda pts: np.interp(x, [p[0] for p in pts], [p[1] for p in pts])
    f = k(f0) * (1 + 0.012 * np.sin(2 * np.pi * 9 * t) + 0.004 * lp(RNG.standard_normal(n), 40) * 8)
    ph = np.cumsum(f) / SR
    src = np.zeros(n)
    for h in range(1, 14):
        src += (h ** -0.9) * np.sin(2 * np.pi * h * ph) * (h * f < 16000)
    y = 1.0 * svf(src, k(F1), 4.0, "bp") + 0.7 * svf(src, k(F2), 6.0, "bp") + 0.25 * svf(src, 3800, 5.0, "bp")
    nasal = np.clip(x / 0.12, 0, 1)
    y = nasal * y + (1 - nasal) * lp(src, 700) * 0.3
    y += 0.03 * bp(RNG.standard_normal(n), 3000, 7000) * env_adsr(n, 0.05, 0, 1, 0.1)
    env = env_adsr(n, 0.04, 0.1, 0.85, dur * 0.35)
    return amp * fade(y * env, 0.003, 0.02)


def bird_chirp(amp):
    """Sparrow-ish 'chip' / 'tsee-wee' phrase."""
    out = []
    kind = RNG.integers(0, 3)
    for j in range(RNG.integers(2, 5)):
        d = RNG.uniform(0.04, 0.09)
        n = int(d * SR)
        t = tvec(n)
        f0 = RNG.uniform(3200, 5200)
        if kind == 0:
            f = f0 * (1 + 0.35 * np.sin(np.pi * t / d))
        elif kind == 1:
            f = f0 * (1.4 - 0.5 * t / d)
        else:
            f = f0 * (0.8 + 0.5 * t / d) * (1 + 0.05 * np.sin(2 * np.pi * 70 * t))
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) ** 1.2
        out.append(y)
        out.append(np.zeros(int(RNG.uniform(0.03, 0.08) * SR)))
    return amp * fade(np.concatenate(out), 0.001, 0.005)


# ================================================================ beds
def rain_beds():
    st = np.zeros((2, N))
    # overall rain intensity (1 = heavy)
    inten = curve([(0, 1.0), (6, 0.9), (6.5, 0.78), (27, 0.75), (38, 0.62), (44, 0.5), (47.5, 0.35),
                   (49.5, 0.08), (50.5, 0.0), (60, 0.0)])
    gust = np.clip(1 + 0.15 * slow_noise(N, 0.35), 0.65, 1.35)
    for c in range(2):
        nz = RNG.standard_normal(N)
        roar = bp(nz, 120, 1200) * 0.9
        hiss = bp(nz, 1800, 11000)
        hiss = eq(hiss, "peak", 4500, 3, 0.8)
        bed = (roar * inten ** 1.4 + hiss * inten) * gust * 0.05
        # individual drops (splashes on pavement / puddles / shop roofs)
        rate = 90 * inten
        kern = [kernel_splash(RNG.uniform(1500, 7000), RNG.uniform(0.002, 0.006)) for _ in range(6)]
        drops = conv_classes(impulses(rate), kern) * 0.05
        # a few close bright puddle plinks
        pl = bubble_stream(6 * inten, 700, 2400, 0.035)
        st[c] = bed + drops + pl
    # gutter trickle: bubble stream + gurgle, left, slightly distant
    trick_rate = curve([(0, 45), (49.5, 40), (52, 20), (55, 6), (57, 0), (60, 0)])
    tr_ = bubble_stream(trick_rate, 350, 1300, 0.045)
    gurgle = lp(RNG.standard_normal(N), 900) * (0.5 + 0.5 * slow_noise(N, 6)).clip(0, 1.5)
    tr_ = lp(tr_ + 0.01 * gurgle * (trick_rate / 45), 5000)
    st += np.stack([tr_ * 0.85, tr_ * 0.35])
    return st, inten


def canopy_drum():
    """Rain drumming on the stretched umbrella fabric (31.0-49.5)."""
    lvl = curve([(0, 0), (30.8, 0), (31.05, 1.0), (38, 0.95), (44, 0.75), (47.5, 0.5), (49.3, 0.12),
                 (49.6, 0.0), (60, 0)])
    st = np.zeros((2, N))
    kern = []
    for _ in range(6):
        n = int(0.05 * SR)
        t = tvec(n)
        tap = lp(bp(RNG.standard_normal(n), 250, 2200), 1800) * np.exp(-t / RNG.uniform(0.004, 0.009))
        memb = np.sin(2 * np.pi * RNG.uniform(190, 280) * t) * np.exp(-t / 0.02) * 0.5
        kern.append(fade(tap + memb, 0.0003, 0.005))
    for c in range(2):
        imp = impulses(170 * lvl)
        y = conv_classes(imp, kern)
        y = eq(y, "peak", 230, 5, 1.5)        # fabric membrane resonance
        y = eq(y, "peak", 700, 2, 1.0)
        y = eq(y, "highshelf", 5000, -8)
        st[c] = y * 0.034
    # sheet of patter (dense small drops, soft): filtered noise AM'd by the drop density
    pat = bp(RNG.standard_normal((2, N)), 400, 2500) * lvl * 0.010
    st += pat
    # rib-tip drips falling off the canopy edge onto the pavement
    for c, pn in ((0, -1), (1, 1)):
        st[c] += bubble_stream(3.0 * lvl, 600, 1600, 0.05)
    return st


def randoseru_rain():
    lvl = curve([(0, 0), (43.65, 0), (43.9, 1.0), (47, 0.7), (49.3, 0.15), (49.6, 0), (60, 0)])
    kern = []
    for _ in range(4):
        n = int(0.03 * SR)
        t = tvec(n)
        f = RNG.uniform(800, 1400)
        kern.append(fade(np.sin(2 * np.pi * f * t) * np.exp(-t / 0.004) +
                         0.4 * bp(RNG.standard_normal(n), 2000, 7000) * np.exp(-t / 0.0015), 0.0002, 0.003))
    y = conv_classes(impulses(28 * lvl), kern) * 0.045
    return np.stack([y * 0.7, y * 1.0])


# ================================================================ cues
def cues(bus):
    put = bus.add
    # 6.4 kitten mew (beside Mina, left of center)
    put(6.4, mew(0.55, amp=0.22), -0.35)
    # awning drips (6-27): little plips around her, slow
    t = 6.2
    while t < 27:
        put(t, plip(RNG.uniform(800, 1500), RNG.uniform(0.04, 0.09)), RNG.uniform(-0.5, 0.5))
        t += RNG.uniform(0.35, 0.9)
    # 9.8 drip lands on her hat: close plip + soft "tup" on the hat brim
    put(9.8, plip(1150, 0.22, tau=0.03, rise=0.8), -0.1)
    n = int(0.1 * SR)
    tt = tvec(n)
    put(9.8, fade(lp(bp(RNG.standard_normal(n), 300, 2500), 1500) * np.exp(-tt / 0.012), 0.0005, 0.01) * 0.25, -0.1)
    # 10.6-11.3 Robo rolls in from the right: tread rumble + motor + rhythmic squeak
    d = 0.95
    n = int(d * SR)
    tt = tvec(n)
    pan = 0.95 - 0.7 * np.clip(tt / 0.7, 0, 1)
    rumble = lp(RNG.standard_normal(n), 350) * (0.6 + 0.4 * np.sin(2 * np.pi * 14 * tt)) * 0.25
    motor = servo(d, 260, 300, 0.12, grit=0.2)
    body = (rumble + motor) * env_adsr(n, 0.08, 0, 1, 0.25, hold=0.7)
    th = (pan + 1) * np.pi / 4
    bus.add(10.55, np.stack([body * np.cos(th), body * np.sin(th)]) * 1.41)
    for i in range(4):
        tq = 10.6 + i * 0.175
        put(tq, squeak(0.085, 2050 + 120 * (i % 2), 0.11), 0.9 - 0.6 * i / 3)
    # 11.3 stop: brake squeak + clunk + beep (C6, in F major)
    put(11.27, squeak(0.12, 1800, 0.12), 0.3)
    put(11.3, clack(0.25, pings=(1250, 1900), low=95), 0.3)
    put(11.36, beep(hz("C6"), 0.06, 0.07), 0.3)
    # head turn servo just before speaking
    put(11.38, servo(0.16, 400, 520, 0.06), 0.3)
    # 15.5 scan whirr: servo + scanning beam hum, sweeping across her
    put(15.5, servo(0.35, 300, 700, 0.07), 0.25)
    n = int(1.3 * SR)
    tt = tvec(n)
    hum = (np.sin(2 * np.pi * hz("F6") * tt) * (0.6 + 0.4 * np.sin(2 * np.pi * 24 * tt)) +
           0.4 * np.sin(2 * np.pi * hz("C6") * tt))
    sweep = svf(RNG.standard_normal(n), 2000 + 2500 * np.sin(np.pi * tt / 1.3), 4, "bp") * 0.3
    beam = (hum * 0.6 + sweep) * env_adsr(n, 0.15, 0, 1, 0.35) * 0.05
    bus.add(15.75, np.stack([beam * np.cos((0.6 - tt / 1.3 + 1) * np.pi / 4), beam * np.sin((0.6 - tt / 1.3 + 1) * np.pi / 4)]) * 1.41)
    put(17.15, beep(hz("A5"), 0.05, 0.05), 0.25)
    put(17.25, beep(hz("C6"), 0.05, 0.05), 0.25)
    # 19.7 head hatch servo -> 20.2 hatch clack + tiny umbrella pop
    put(19.7, servo(0.45, 500, 900, 0.05), 0.25)
    put(20.15, clack(0.18, pings=(2600, 3900), tau=0.03, low=180), 0.25)
    n = int(0.15 * SR)
    tt = tvec(n)
    pop = lp(RNG.standard_normal(n), 2500) * np.exp(-tt / 0.006) * 0.5 + \
        np.sin(2 * np.pi * np.cumsum(900 * np.exp(-tt / 0.03) + 300) / SR) * np.exp(-tt / 0.02)
    put(20.2, fade(pop, 0.0003, 0.01) * 0.2, 0.25)
    put(20.24, rustle(0.18, 0.06, 1500, 8000, 80), 0.25)       # tiny fabric flick
    # 21.6 Robo's "? ?" screen: two questioning up-blips + processing chatter
    put(21.75, chirp_beep(hz("C5"), hz("A5"), 0.16, 0.07), 0.25)
    put(22.05, chirp_beep(hz("C5"), hz("C6"), 0.18, 0.07), 0.25)
    pent = ["C6", "D6", "F6", "G6", "A6"]
    for i in range(10):
        put(22.45 + i * 0.075, beep(hz(pent[RNG.integers(0, 5)]), 0.035, 0.025), 0.25)
    # 27.0-30.8 transformation: panel split, servo whirrs, clacks on the beat, 8 ribs telescope, locks
    bt = 0.475
    put(27.0, clack(0.35, pings=(900, 1450, 2300), tau=0.08, low=70), 0.0)
    put(27.0, thump(0.25, 60, 0.12), 0.0)
    hs = noise_sweep(0.6, 6000, 2500, 0.8) * env_adsr(int(0.6 * SR), 0.005, 0, 1, 0.5) * 0.06
    bus.add(27.02, hs)                                        # hydraulic hiss
    for b in range(1, 8):
        put(27.0 + b * bt, clack(0.16 + 0.02 * b, pings=(1500 + 150 * b, 2400 + 200 * b), low=90), (-1) ** b * 0.4)
        put(27.0 + b * bt - 0.3, servo(0.28, 350 + 40 * b, 700 + 70 * b, 0.06), (-1) ** b * 0.3)
    for i in range(8):                                        # ribs telescoping out: shk-tink x8
        tr0 = 29.15 + i * 0.19
        sl = noise_sweep(0.09, 1500, 6000, 2.0, st=False) * np.sin(np.pi * tvec(int(0.09 * SR)) / 0.09) * 0.12
        put(tr0, sl, -0.8 + 1.6 * i / 7)
        put(tr0 + 0.09, clack(0.08, pings=(3000 + 150 * i, 4600), tau=0.03, low=300), -0.8 + 1.6 * i / 7)
    put(30.55, clack(0.3, pings=(800, 1300), tau=0.06, low=70), 0.0)   # "ka-chunk" final lock
    put(30.62, clack(0.22, pings=(1100, 1700), tau=0.05, low=80), 0.0)
    # 30.8 canopy opens: air whoosh into a big fabric whump
    ws = noise_sweep(0.45, 3500, 500, 0.7) * np.sin(np.pi * tvec(int(0.45 * SR)) / 0.45) ** 2 * 0.12
    bus.add(30.4, ws)
    n = int(0.8 * SR)
    tt = tvec(n)
    wh = svf(RNG.standard_normal(n), 220 * np.exp(-tt / 0.1) + 60, 0.8, "lp") * np.exp(-tt / 0.18) * 1.2
    wh += np.sin(2 * np.pi * np.cumsum(70 * np.exp(-tt / 0.15) + 40) / SR) * np.exp(-tt / 0.22) * 0.8
    wh += bp(RNG.standard_normal(n), 300, 2500) * np.exp(-tt / 0.05) * 0.3  # fabric snap
    bus.add(30.8, np.stack([wh, wh]) * 0.3)
    # 34.6 electric sparks: crackling bursts (and a few more while it glitches in s05)
    for ts, dd, a in [(34.6, 0.5, 0.2), (35.25, 0.18, 0.12), (36.1, 0.3, 0.13), (37.05, 0.15, 0.1),
                      (38.1, 0.25, 0.12), (39.7, 0.2, 0.09), (41.5, 0.3, 0.1)]:
        put(ts, sparks(dd, a), RNG.uniform(0.0, 0.5))
    # 38.6 robot glitch: crushed static bursts + stuttered beeps (mostly in the line's pauses)
    for ts, dd in [(38.25, 0.3), (39.63, 0.4), (40.47, 0.38)]:
        put(ts, glitch(dd, 0.08), 0.3)
    n = int(4.3 * SR)
    stat = crushed(RNG.standard_normal(n) * 0.5, 3, 30) * env_adsr(n, 0.2, 0, 1, 0.6) * 0.006
    put(38.0, hp(stat, 1500), 0.3)                           # faint screen static
    n = int(0.9 * SR)
    tt = tvec(n)
    pdn = np.sin(2 * np.pi * np.cumsum(1400 * np.exp(-tt / 0.3) + 120) / SR) * env_adsr(n, 0.01, 0, 1, 0.3) * 0.05
    put(41.4, pdn, 0.3)                                      # power-down whine
    # 43.5 randoseru: rustle, strap buckle clinks, leather creak, set down on Robo's head
    put(43.45, rustle(0.75, 0.2, 500, 7000, 60), -0.15)
    put(43.6, clack(0.05, pings=(3300, 5100), tau=0.04, low=600), -0.2)
    put(43.85, clack(0.04, pings=(3600, 5400), tau=0.03, low=600), 0.0)
    n = int(0.35 * SR)
    tt = tvec(n)
    stick = (np.sin(2 * np.pi * 32 * tt) > 0.6).astype(float)
    creak = svf(stick + 0.05 * RNG.standard_normal(n), 340 + 80 * tt / 0.35, 8, "bp") * np.sin(np.pi * tt / 0.35) * 0.15
    put(43.75, creak, 0.05)
    put(44.1, thump(0.08, 140, 0.04), 0.15)
    # 46.4 warm power-up hum (F major, rises an octave into place, glows, fades)
    n = int(2.6 * SR)
    tt = tvec(n)
    rise = 2 ** (-1 + np.clip(tt / 0.9, 0, 1) ** 0.6)
    hum = np.zeros(n)
    for h, a in [(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15), (6, 0.06)]:
        hum += a * np.sin(2 * np.pi * np.cumsum(hz("F3") * h * rise) / SR)
    hum *= (1 + 0.08 * np.sin(2 * np.pi * 4.5 * tt)) * env_adsr(n, 0.4, 0, 1, 1.2) * 0.05
    put(46.4, lp(hum, 2500), 0.25)
    put(47.25, beep(hz("F6"), 0.08, 0.035), 0.25)
    # 49.5 rain stops: last drips with decaying rate
    t = 49.5
    while t < 57.5:
        x = (t - 49.5) / 8
        put(t, plip(RNG.uniform(700, 1500), RNG.uniform(0.04, 0.1) * (1 - 0.6 * x), tau=0.03), RNG.uniform(-0.7, 0.7))
        t += 0.12 + 1.8 * x ** 1.3 + RNG.uniform(0, 0.25)
    # 51.0 sun breaks: rising chime cascade (G major pentatonic) + airy swell
    pent = ["G5", "A5", "B5", "D6", "E6", "G6", "A6", "B6", "D7", "E7", "G7"]
    for i, nt in enumerate(pent):
        n = int(1.4 * SR)
        tt = tvec(n)
        f = hz(nt)
        y = (np.sin(2 * np.pi * f * tt) + 0.2 * np.sin(2 * np.pi * f * 2.76 * tt) * np.exp(-tt / 0.1)) * np.exp(-tt / 0.5)
        put(51.0 + i * 0.09, fade(y, 0.001, 0.05) * 0.03, -0.8 + 1.6 * i / len(pent))
    sw = noise_sweep(2.2, 2500, 9000, 1.5) * np.sin(np.pi * tvec(int(2.2 * SR)) / 2.2) ** 2 * 0.02
    bus.add(50.8, sw)
    # birds 52-60, light
    t = 52.3
    while t < 59.3:
        put(t, bird_chirp(RNG.uniform(0.012, 0.025)), RNG.choice([-0.85, -0.6, 0.6, 0.85]))
        t += RNG.uniform(0.8, 1.8)
    # 55.8 happy kitten mew (rising, "mya~ah")
    put(55.8, mew(0.3, f0=((0, 700), (0.4, 1050), (1, 1150)), F1=((0, 500), (0.4, 1100), (1, 1000)),
                  F2=((0, 2100), (1, 1700)), amp=0.18), -0.35)
    put(56.12, mew(0.42, f0=((0, 900), (0.3, 1200), (1, 800)), amp=0.18), -0.35)
    # 58.3 robot happy beep: pi-po-pi (G major: B6 D7 G7)
    for i, nt in enumerate(["B6", "D7", "G7"]):
        put(58.3 + i * 0.1, beep(hz(nt), 0.065, 0.06), 0.25)


def sparks(dur, amp):
    n = int(dur * SR)
    t = tvec(n)
    dens = 1500 * (0.3 + np.abs(slow_noise(n, 25)))
    imp = (RNG.random(n) < dens / SR) * RNG.exponential(1, n) * RNG.choice([-1, 1], n)
    k = hp(RNG.standard_normal(int(0.002 * SR)), 3000) * np.exp(-tvec(int(0.002 * SR)) / 0.0004)
    crack = ss.oaconvolve(imp, k)[:n]
    buzz = bp(saw(120.0 * (1 + 0.05 * np.sin(2 * np.pi * 9 * t)), n), 800, 5000) * \
        (slow_noise(n, 30) > 0.3)
    y = crack * 0.8 + buzz * 0.25 + 0.15 * hp(RNG.standard_normal(n), 5000) * np.exp(-t / 0.05)
    y = fade(y * env_adsr(n, 0.002, 0, 1, dur * 0.4), 0.0005, 0.02)
    return amp * y / np.max(np.abs(y))


def glitch(dur, amp):
    """Digital stutter: grains of crushed noise and a stuck square beep, gated irregularly."""
    n = int(dur * SR)
    y = np.zeros(n)
    i = 0
    while i < n:
        g = int(RNG.uniform(0.02, 0.05) * SR)
        kind = RNG.integers(0, 3)
        tt = tvec(min(g, n - i))
        if kind == 0:
            seg = crushed(RNG.standard_normal(len(tt)) * 0.6, 3, int(RNG.integers(8, 40)))
        elif kind == 1:
            f = hz(["C6", "F6", "A5"][RNG.integers(0, 3)])
            seg = np.sign(np.sin(2 * np.pi * f * tt)) * 0.5
        else:
            seg = np.zeros(len(tt))
        reps = RNG.integers(1, 3)
        for r in range(reps):
            j = i + r * len(tt)
            L = min(len(tt), n - j)
            if L > 0:
                y[j:j + L] += fade(seg[:L], 0.001, 0.002)
        i += len(tt) * reps + int(RNG.uniform(0, 0.02) * SR)
    y = lp(y, 9000)
    return amp * y / (np.max(np.abs(y)) + 1e-9)


# ================================================================ main
def render():
    rain, inten = rain_beds()
    can = canopy_drum()
    ran = randoseru_rain()
    bus = Bus()
    cues(bus)
    fx = bus.out()
    outdoor = make_ir(t60=0.7, predelay=0.012, seed=41, lf=0.8, hf=0.5, density_ms=5)
    wet = reverb(hp(fx, 250), outdoor, width=1.2)
    mix = rain + can + ran + fx + 0.18 * wet
    mix = hp(mix, 30, 2)
    mix = fade(mix, 0.0, 0.05)
    return mix


def main():
    tl = load_timeline()
    print("  cues:", ", ".join(f"{c['t']}:{c['name']}" for c in tl["sfx"]))
    mix = render()
    report("sfx", mix)
    print(write_wav(BUILD / "sfx.wav", mix))


if __name__ == "__main__":
    main()
