"""Synthesized sound effects + ambience -> build/audio/sfx.wav (48 kHz stereo, 60.000 s).

Every cue in story/timeline.json "sfx" is rendered at its time; the ambience
beds (night wind / distant insects 0-13 s, garden crickets 13-56 s, dawn
birds 55-60 s) and the fūrin wind-chime tinkles (7-13 s) are added around them.
"""
import numpy as np

from dsp import (SR, N, BUILD, Bus, tvec, env_adsr, fade, curve, sine, lp, hp, bp, svf, partials,
                 make_ir, reverb, write_wav, report, db, undb, load_timeline, hz)

RNG = np.random.default_rng(777)


# ------------------------------------------------------------------ elements
def ping(f, tau=0.06, amp=1.0):
    n = int(min(tau * 6, 1.0) * SR)
    t = tvec(n)
    y = np.sin(2 * np.pi * f * t + RNG.uniform(0, 6)) * np.exp(-t / tau)
    y += 0.3 * np.sin(2 * np.pi * f * 2.71 * t) * np.exp(-t / (tau * 0.4))
    return amp * fade(y, 0.001, 0.005)


def sparkles(bus, t0, t1, rate, fmin=4000, fmax=11000, amp=0.05, pan=None, decay=None, taus=(0.03, 0.12)):
    """Poisson-distributed glittering pings between t0 and t1."""
    t = t0
    while True:
        t += RNG.exponential(1.0 / rate)
        if t >= t1:
            break
        x = (t - t0) / max(t1 - t0, 1e-3)
        a = amp * (np.exp(-x * decay) if decay else 1.0) * RNG.uniform(0.3, 1.0)
        f = np.exp(RNG.uniform(np.log(fmin), np.log(fmax)))
        p = pan(x) + RNG.uniform(-0.25, 0.25) if pan else RNG.uniform(-0.9, 0.9)
        bus.add(t, ping(f, RNG.uniform(*taus), a), float(np.clip(p, -1, 1)))


def noise_sweep(dur, f0, f1, q=1.2, st=True):
    n = int(dur * SR)
    t = tvec(n)
    fc = f0 * (f1 / f0) ** (t / t[-1])
    nz = RNG.standard_normal((2, n) if st else n)
    return svf(nz, fc, q, "bp")


def cricket_track(f, t0, t1, period, pulses, amp, jitter=0.15, pulse_len=0.013, gap=0.02, lp_hz=None):
    """Field-cricket chirps (sine carrier, pulsed) as a mono 60 s track."""
    env = np.zeros(N)
    pl = int(pulse_len * SR)
    w = np.sin(np.pi * np.arange(pl) / pl) ** 2
    t = t0 + RNG.uniform(0, period)
    while t < t1:
        for k in range(pulses):
            i = int((t + k * (pulse_len + gap)) * SR)
            if i + pl < N:
                env[i:i + pl] += w * RNG.uniform(0.75, 1.0)
        t += period * (1 + RNG.uniform(-jitter, jitter))
    fm = f * (1 + 0.004 * np.sin(2 * np.pi * 0.37 * tvec(N)))
    y = (sine(fm, N) + 0.15 * sine(2 * fm, N)) * env
    if lp_hz:
        y = lp(y, lp_hz)
    return amp * y


def bell_cricket(f, t0, t1, amp):
    """Suzumushi-like 'riiin': fast-trilled tone bursts."""
    env = np.zeros(N)
    t = t0 + RNG.uniform(0, 1)
    while t < t1:
        d = RNG.uniform(0.35, 0.6)
        n = int(d * SR)
        tt = tvec(n)
        e = (0.5 + 0.5 * np.sin(2 * np.pi * 48 * tt)) ** 2 * env_adsr(n, 0.04, 0, 1, 0.12)
        i = int(t * SR)
        L = min(n, N - i)
        env[i:i + L] += e[:L]
        t += d + RNG.uniform(0.5, 1.4)
    return amp * sine(f, N) * env


def rustle(dur, amp):
    n = int(dur * SR)
    t = tvec(n)
    nz = bp(RNG.standard_normal(n), 700, 6000)
    g = np.zeros(n)
    for _ in range(int(dur * 45)):  # many tiny fabric grains
        c = RNG.uniform(0, dur)
        w = RNG.uniform(0.005, 0.03)
        g += RNG.uniform(0.2, 1) * np.exp(-0.5 * ((t - c) / w) ** 2)
    return amp * fade(nz * g * env_adsr(n, 0.02, 0, 1, dur * 0.4), 0.003, 0.02)


def thump(amp, f=75, tau=0.09):
    n = int(0.4 * SR)
    t = tvec(n)
    y = np.sin(2 * np.pi * np.cumsum(f * (1 + 0.6 * np.exp(-t / 0.02))) / SR) * np.exp(-t / tau)
    y += 0.5 * lp(RNG.standard_normal(n), 300) * np.exp(-t / 0.03)
    return amp * fade(y, 0.001, 0.05)


def footstep(amp, kind="stone", scuff=False):
    n = int(0.35 * SR)
    t = tvec(n)
    nz = RNG.standard_normal(n)
    if kind == "stone":
        # straw sandal on stone: soft slap + knock + grit
        slap = bp(nz, 900, 4500) * np.exp(-t / RNG.uniform(0.008, 0.014))
        f = RNG.uniform(150, 210)
        knock = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.025) * 0.6
        grit = hp(nz, 3500) * np.exp(-t / 0.035) * RNG.uniform(0.12, 0.25)
        y = slap + knock + grit
        if scuff:
            y += bp(RNG.standard_normal(n), 1500, 7000) * env_adsr(n, 0.03, 0, 1, 0.15, hold=0.08) * 0.35
    else:  # wooden floor
        f = RNG.uniform(110, 150)
        y = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.05) + 0.4 * bp(nz, 400, 2500) * np.exp(-t / 0.012)
    return amp * fade(y, 0.0015, 0.05)


def glass_chime(f, amp, tau=1.8):
    n = int(tau * 4 * SR)
    return amp * partials(f, n, [(1, 1, tau), (2.76, 0.4, tau * 0.45), (5.4, 0.18, tau * 0.2),
                                 (8.93, 0.06, tau * 0.1)], RNG, att=0.0008)


def furin(amp):
    """Fūrin: glass bell hit 1-3 times by its clapper."""
    f = 2350.0
    out = np.zeros(int(3.5 * SR))
    t = 0.0
    for _ in range(RNG.integers(1, 4)):
        y = partials(f * RNG.uniform(0.998, 1.002), int(3.0 * SR),
                     [(1, 1, 0.8), (2.43, 0.5, 0.4), (3.9, 0.25, 0.2), (5.6, 0.12, 0.1)], RNG, att=0.0005)
        i = int(t * SR)
        out[i:i + len(y)] += y[:len(out) - i] * RNG.uniform(0.5, 1.0)
        t += RNG.uniform(0.09, 0.22)
    return amp * fade(out, 0.0005, 0.3)


def bird_syllable(kind):
    if kind == "sparrow":   # 'chun' — short down-sweep with harmonics
        d = RNG.uniform(0.05, 0.08)
        n = int(d * SR)
        t = tvec(n)
        f = 4300 - 1600 * (t / d) ** 0.7 + 250 * np.sin(2 * np.pi * 60 * t)
        ph = np.cumsum(f) / SR
        y = np.sin(2 * np.pi * ph) + 0.35 * np.sin(4 * np.pi * ph)
    else:                   # whistle — up/down glide with fast trill
        d = RNG.uniform(0.09, 0.18)
        n = int(d * SR)
        t = tvec(n)
        a, b = RNG.uniform(2200, 3000), RNG.uniform(3200, 4600)
        f = a + (b - a) * np.sin(np.pi * t / d) + 120 * np.sin(2 * np.pi * RNG.uniform(25, 40) * t)
        y = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return y * np.sin(np.pi * np.arange(n) / n) ** 1.5


# ------------------------------------------------------------------ build
def render():
    tl = load_timeline()
    cue = {c["name"]: c["t"] for c in tl["sfx"]}
    dry = Bus()
    out_send = Bus()     # outdoor reverb (garden / shrine)
    sky_send = Bus()     # big sky reverb

    def put(t, sig, pan=0.0, gain=1.0, rev=0.2, sky=0.0):
        dry.add(t, sig, pan, gain)
        if rev:
            out_send.add(t, sig, pan, gain * rev)
        if sky:
            sky_send.add(t, sig, pan, gain * sky)

    # ---- ambience: wind (whole film, level follows the location)
    wn = RNG.standard_normal((2, N))
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * tvec(N) + np.array([[0.0], [1.3]])) \
        * (0.7 + 0.3 * np.sin(2 * np.pi * 0.23 * tvec(N)))
    wind = svf(lp(wn, 900, 1), 250 + 300 * gust[0], 0.9, "bp") * gust
    wlev = curve([(0, -44), (7, -44), (7.5, -52), (13, -52), (13.5, -47), (25, -47), (37, -44), (48, -43),
                  (55, -46), (60, -48)])
    dry.add(0, wind * undb(wlev) * 6)
    # ---- distant insects 0-13 (night, soft, filtered)
    distant = (cricket_track(4300, 0, 13.5, 0.9, 3, 1.0, lp_hz=3500)
               + cricket_track(4900, 0, 13.5, 1.3, 4, 0.8, lp_hz=3500))
    dry.add(0, np.stack([distant * 0.7, distant]) * undb(-50) * fade(np.ones(N), 1.5, 0.0))
    out_send.add(0, np.stack([distant, distant * 0.7]) * undb(-46))
    # ---- garden crickets bed from 13.0
    bed = np.zeros((2, N))
    for f, per, pul, amp, pan in [(4400, 0.62, 3, 1.0, -0.6), (4750, 0.8, 4, 0.8, 0.5), (5200, 0.45, 2, 0.55, 0.15),
                                  (3900, 1.1, 5, 0.6, -0.2), (4600, 0.7, 3, 0.45, 0.85)]:
        tr = cricket_track(f, cue["crickets_bed_start"] - 0.3, 57.0, per, pul, amp)
        th = (pan + 1) * np.pi / 4
        bed += np.stack([tr * np.cos(th), tr * np.sin(th)])
    for f, pan, amp in [(4100, -0.4, 0.5), (4050, 0.6, 0.35)]:
        tr = bell_cricket(f, 13.2, 57.0, amp)
        th = (pan + 1) * np.pi / 4
        bed += np.stack([tr * np.cos(th), tr * np.sin(th)])
    blev = curve([(0, -80), (12.7, -80), (13.4, -40), (25, -40), (25.5, -45), (37, -45), (37.5, -42),
                  (48, -44), (53, -50), (56.5, -70), (57, -90), (60, -90)])
    bed = bed * undb(blev) * 2.2
    dry.add(0, bed)
    out_send.add(0, bed * 0.6)

    # ---- shooting star 2.2: whoosh across the sky, sputter, dive
    t0 = cue["shooting_star_whoosh"]
    d = 2.0
    n = int(d * SR)
    t = tvec(n)
    ws = noise_sweep(d, 3800, 900, 1.6)
    amp = np.where(t < 0.35, (t / 0.35) ** 2, 1.0) * np.exp(-np.maximum(t - 0.9, 0) / 0.45)
    sput = 1 - 0.55 * np.clip((t - 0.8) / 0.1, 0, 1) * np.clip((1.3 - t) / 0.1, 0, 1) \
        * (0.5 + 0.5 * np.sin(2 * np.pi * 13 * t))
    panc = np.clip(-0.9 + 1.5 * t / 1.2, -0.9, 0.7)
    th = (panc + 1) * np.pi / 4
    whoosh = ws * amp * sput * np.stack([np.cos(th), np.sin(th)]) * 1.4
    whistle = np.sin(2 * np.pi * np.cumsum(3000 * (0.55 ** (t / d))) / SR) * amp * sput * 0.05
    whoosh += np.stack([whistle * np.cos(th), whistle * np.sin(th)])
    put(t0, fade(whoosh, 0.02, 0.2) * 0.22, rev=0.1, sky=0.5)
    sparkles(dry, t0 + 0.05, t0 + 1.6, 55, amp=0.035, pan=lambda x: -0.9 + 1.6 * x, decay=1.5)
    sparkles(sky_send, t0 + 0.05, t0 + 1.6, 25, amp=0.03, pan=lambda x: -0.9 + 1.6 * x, decay=1.5)

    # ---- 7.6 star pass shimmer
    t0 = cue["star_pass_shimmer"]
    d = 0.7
    sh = noise_sweep(d, 6000, 2500, 2.5)
    tt = tvec(sh.shape[1])
    th = (np.clip(-0.7 + 1.4 * tt / d, -1, 1) + 1) * np.pi / 4
    sh = sh * np.sin(np.pi * tt / d) ** 2 * np.stack([np.cos(th), np.sin(th)])
    put(t0 - 0.15, sh * 0.12, rev=0.2, sky=0.3)
    sparkles(dry, t0 - 0.1, t0 + 0.5, 60, amp=0.03, pan=lambda x: -0.7 + 1.4 * x, decay=1.0)

    # ---- 8.0 soft cute "pon" in the garden (heard from the attic)
    t0 = cue["soft_impact_pon"]
    n = int(0.6 * SR)
    t = tvec(n)
    f = 330 + 420 * np.exp(-t / 0.035)
    pon = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.11) + 0.6 * np.sin(2 * np.pi * 85 * t) * np.exp(-t / 0.07)
    put(t0, lp(fade(pon, 0.001, 0.05), 3500) * 0.3, pan=0.15, rev=0.4)
    sparkles(dry, t0 + 0.02, t0 + 0.6, 20, amp=0.012, decay=2)

    # ---- fūrin tinkles 7-13 (away from Haru's line at 8.6-9.7)
    for tf, p, a in [(7.25, 0.55, 0.05), (10.35, 0.5, 0.045), (11.55, 0.6, 0.035), (12.5, 0.5, 0.04)]:
        put(tf, furin(a), p, rev=0.35)

    # ---- 9.9 Haru jumps up: cloth rustle + thump, then runs off across the wooden floor
    t0 = cue["haru_jumps_up"]
    put(t0, rustle(0.4, 0.2), -0.05, rev=0.15)
    put(t0 + 0.28, thump(0.35), 0.0, rev=0.15)
    for k, tf in enumerate([10.55, 10.85, 11.13, 11.42, 11.7, 11.98]):
        put(tf, lp(footstep(0.3 * 0.8 ** k, "wood"), 3000 - 300 * k), 0.1 + 0.12 * k, rev=0.25)
    put(10.5, rustle(1.2, 0.05), 0.3, rev=0.15)

    # ---- 25-37 running footsteps up stone steps, ~3 steps/s
    ts, te = cue["footsteps_running_start"], cue["footsteps_running_stop"]
    tf, k = ts + 0.12, 0
    while tf < te - 0.2:
        a = 0.26 * RNG.uniform(0.8, 1.0) * (1.08 if k % 2 == 0 else 0.92)
        put(tf, footstep(a, "stone"), -0.12 if k % 2 else 0.12, rev=0.18)
        tf += 1 / 3 + RNG.uniform(-0.018, 0.018)
        k += 1
    put(te - 0.12, footstep(0.3, "stone", scuff=True), 0.0, rev=0.2)
    put(te - 0.35, rustle(0.5, 0.06), 0.1, rev=0.1)

    # ---- 40.2 lantern ignite: match scratch, whoomp, crackle, flame roar
    t0 = cue["lantern_ignite_whoomp"]
    n = int(0.18 * SR)
    t = tvec(n)
    put(t0 - 0.32, bp(RNG.standard_normal(n), 2000, 7000) * env_adsr(n, 0.01, 0.05, 0.6, 0.1) * 0.12, 0.2, rev=0.2)
    n = int(1.4 * SR)
    t = tvec(n)
    body = svf(RNG.standard_normal(n), 150 + 1600 * np.exp(-t / 0.18) * (1 - np.exp(-t / 0.03)), 0.8, "lp")
    whoomp = body * (1 - np.exp(-t / 0.03)) * np.exp(-t / 0.35) * 1.6
    whoomp += np.sin(2 * np.pi * np.cumsum(55 + 30 * np.exp(-t / 0.2)) / SR) * (1 - np.exp(-t / 0.04)) * np.exp(-t / 0.3) * 0.6
    put(t0, fade(whoomp, 0.005, 0.2) * 0.35, 0.05, rev=0.3)
    # crackle (decaying rate) + flame hiss that fades as the lantern lifts away
    tc = t0 + 0.05
    while tc < 48.6:
        x = tc - t0
        tc += RNG.exponential(1 / (4 + 22 * np.exp(-x / 1.2)))
        m = int(RNG.uniform(0.001, 0.004) * SR)
        cr = bp(RNG.standard_normal(m + 64), 1200, 6000)[:m] * np.hanning(m)
        put(tc, cr * RNG.uniform(0.03, 0.12) * (1 if tc < 48 else 0.4), RNG.uniform(-0.2, 0.3), rev=0.15)
    n = int((49.5 - t0) * SR)
    t = tvec(n)
    flame = lp(hp(RNG.standard_normal((2, n)), 250), 2200) * (0.8 + 0.2 * np.sin(2 * np.pi * 1.7 * t))
    fl_env = env_adsr(n, 0.3, 0, 1, 1.5, hold=(48.0 - t0))
    put(t0, flame * fl_env * 0.018, rev=0.2)

    # ---- 48.0 lantern rise chimes: ascending E-major pentatonic glass chimes
    t0 = cue["lantern_rise_chimes"]
    notes = ["B5", "C#6", "E6", "F#6", "G#6", "B6", "C#7", "E7"]
    for i, nt in enumerate(notes):
        put(t0 + 0.2 + i * 0.55, glass_chime(hz(nt), 0.025 + 0.003 * i, tau=1.4), -0.3 + 0.1 * i, rev=0.3, sky=0.4)
    sparkles(dry, t0 + 0.2, t0 + 4.6, 6, amp=0.012, fmin=5000)

    # ---- 52.8 star burst: rising suck-in, glittering explosion, soft low bloom
    t0 = cue["star_burst_sparkle"]
    d = 0.6
    up = noise_sweep(d, 500, 7000, 1.8)
    tt = tvec(up.shape[1])
    put(t0 - d, up * (tt / d) ** 3 * 0.18, rev=0.1, sky=0.4)
    n = int(2.2 * SR)
    t = tvec(n)
    fw = hp(RNG.standard_normal((2, n)), 2500) * np.exp(-t / 0.35) * (1 - np.exp(-t / 0.004))
    put(t0, fade(fw, 0.001, 0.3) * 0.12, rev=0.2, sky=0.6)
    put(t0, thump(0.35, 48, 0.5), 0, rev=0.1, sky=0.3)
    sparkles(dry, t0, t0 + 2.8, 140, amp=0.05, fmin=3000, fmax=12000, decay=2.2, taus=(0.04, 0.2))
    sparkles(sky_send, t0, t0 + 3.2, 70, amp=0.05, fmin=3000, fmax=12000, decay=2.0)

    # ---- 55.0 dawn birds
    t0 = cue["dawn_birds"]
    for kind, pan, amp, start, gap in [("sparrow", -0.55, 0.05, t0 + 0.3, 1.1), ("whistle", 0.6, 0.035, t0 + 0.9, 1.6),
                                       ("sparrow", 0.2, 0.03, t0 + 1.6, 1.4)]:
        tb = start
        while tb < 59.4:
            for s in range(RNG.integers(2, 5)):
                y = lp(bird_syllable(kind), 7000)
                put(tb, y * amp * RNG.uniform(0.7, 1.0), pan + RNG.uniform(-0.1, 0.1), rev=0.3)
                tb += len(y) / SR + RNG.uniform(0.05, 0.12)
            tb += gap * RNG.uniform(0.7, 1.3)

    # ---- reverbs & master
    out_ir = make_ir(t60=0.9, predelay=0.012, seed=3, lf=1.0, hf=0.5, density_ms=5)
    sky_ir = make_ir(t60=2.6, predelay=0.04, seed=4, lf=0.9, hf=0.6)
    mix = dry.x + 0.5 * reverb(out_send.x, out_ir) + 0.5 * reverb(sky_send.x, sky_ir, width=1.4)
    mix = hp(mix, 25)[:, :N]
    fl = int(1.0 * SR)
    mix[:, -fl:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(fl) / fl)
    mix -= mix.mean(axis=1, keepdims=True)
    return mix


def main():
    mix = render()
    report("sfx", mix)
    print(write_wav(BUILD / "sfx.wav", mix))


if __name__ == "__main__":
    main()
