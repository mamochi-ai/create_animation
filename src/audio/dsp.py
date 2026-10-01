"""Shared DSP toolkit for the ほしのおとしもの audio pipeline.

Everything here is plain numpy/scipy (+ numba for the few per-sample loops,
with a pure-python fallback).  All signals are float64 at SR = 48 kHz.
Mono signals are 1-D arrays, stereo signals are shape (2, n).
"""
import json
from pathlib import Path

import numpy as np
import scipy.signal as ss
import scipy.io.wavfile as wavfile

try:  # numba makes the per-sample loops fast; fall back to python if missing
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

SR = 48000
DUR = 60.0
N = int(round(SR * DUR))
import os
REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("FILM_DIR", REPO)).resolve()
BUILD = ROOT / "build" / "audio"
TWO_PI = 2.0 * np.pi


def load_timeline():
    return json.loads((ROOT / "story" / "timeline.json").read_text())


def load_voices():
    return json.loads((ROOT / "assets" / "voice" / "voices.json").read_text())


# ----------------------------------------------------------------- pitch
_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi(name):
    if isinstance(name, (int, float, np.integer, np.floating)):
        return float(name)
    letter, i, acc = name[0].upper(), 1, 0
    while i < len(name) and name[i] in "#b":
        acc += 1 if name[i] == "#" else -1
        i += 1
    return 12 * (int(name[i:]) + 1) + _NOTE[letter] + acc


def hz(name):
    return 440.0 * 2.0 ** ((midi(name) - 69) / 12.0)


def tvec(n):
    return np.arange(n) / SR


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def undb(d):
    return 10 ** (d / 20.0)


# ----------------------------------------------------------------- envelopes
def env_adsr(n, a=0.01, d=0.1, s=0.8, r=0.2, hold=None):
    """Click-free ADSR.  Raised-cosine attack and release; release starts at
    `hold` seconds (default: so that it ends exactly at the last sample)."""
    t = tvec(n)
    if hold is None:
        hold = max(n / SR - r, 0.0)
    e = np.full(n, float(s))
    a = max(a, 1e-4)
    m = t < a
    e[m] = 0.5 - 0.5 * np.cos(np.pi * t[m] / a)
    if d > 0:
        m = (t >= a) & (t < a + d)
        x = (t[m] - a) / d
        e[m] = s + (1 - s) * (1 - x) ** 2
    rel = np.ones(n)
    r = max(r, 1e-4)
    m = t >= hold
    x = np.clip((t[m] - hold) / r, 0, 1)
    rel[m] = 0.5 + 0.5 * np.cos(np.pi * x)
    return e * rel


def fade(x, fi=0.002, fo=0.01):
    """Raised-cosine fade in/out (seconds) on the last axis, in place copy."""
    x = np.array(x, dtype=float, copy=True)
    n = x.shape[-1]
    ni, no = min(int(fi * SR), n), min(int(fo * SR), n)
    if ni > 1:
        x[..., :ni] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(ni) / ni)
    if no > 1:
        x[..., n - no:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, no + 1) / no)
    return x


def curve(keys, n=N):
    """Piecewise-linear automation curve from [(t, value), ...]."""
    ks = np.array(keys, dtype=float)
    return np.interp(tvec(n), ks[:, 0], ks[:, 1])


# ----------------------------------------------------------------- stereo bus
def pan_st(mono, pan=0.0):
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.stack([mono * np.cos(th) * np.sqrt(2), mono * np.sin(th) * np.sqrt(2)])


class Bus:
    """A stereo mix buffer with a tail beyond the nominal length."""

    def __init__(self, n=N, tail=8.0):
        self.n = n
        self.x = np.zeros((2, n + int(tail * SR)))

    def add(self, t, sig, pan=0.0, gain=1.0):
        sig = np.asarray(sig, dtype=float)
        if sig.ndim == 1:
            sig = pan_st(sig, pan)
        i0 = int(round(t * SR))
        if i0 < 0:
            sig, i0 = sig[:, -i0:], 0
        L = min(sig.shape[1], self.x.shape[1] - i0)
        if L > 0:
            self.x[:, i0:i0 + L] += gain * sig[:, :L]

    def out(self):
        return self.x[:, :self.n]


# ----------------------------------------------------------------- oscillators
def phase(f, n, ph0=0.0):
    if np.isscalar(f):
        return ph0 + np.arange(n) * (f / SR)
    return ph0 + np.cumsum(np.asarray(f, dtype=float)[:n]) / SR


def sine(f, n, ph0=0.0):
    return np.sin(TWO_PI * phase(f, n, ph0))


def saw(f, n, ph0=0.0):
    """Band-limited (polyBLEP) sawtooth; f scalar or per-sample array."""
    f = np.broadcast_to(np.asarray(f, dtype=float), (n,))
    dt = f / SR
    p = np.mod(ph0 + np.cumsum(dt), 1.0)
    y = 2 * p - 1
    m = p < dt
    tt = p[m] / dt[m]
    y[m] -= tt + tt - tt * tt - 1
    m = p > 1 - dt
    tt = (p[m] - 1) / dt[m]
    y[m] -= tt * tt + tt + tt + 1
    return y


def noise(n, rng):
    return rng.standard_normal(n)


# ----------------------------------------------------------------- filters
def _sos(kind, fc, order=2):
    return ss.butter(order, fc, kind, fs=SR, output="sos")


def lp(x, fc, order=2):
    return ss.sosfilt(_sos("lowpass", min(fc, 0.45 * SR), order), x, axis=-1)


def hp(x, fc, order=2):
    return ss.sosfilt(_sos("highpass", fc, order), x, axis=-1)


def bp(x, lo, hi, order=2):
    return ss.sosfilt(_sos("bandpass", [lo, min(hi, 0.45 * SR)], order), x, axis=-1)


def _biquad(kind, f0, gain_db, q):
    A = 10 ** (gain_db / 40)
    w = TWO_PI * f0 / SR
    al = np.sin(w) / (2 * q)
    c = np.cos(w)
    if kind == "peak":
        b = [1 + al * A, -2 * c, 1 - al * A]
        a = [1 + al / A, -2 * c, 1 - al / A]
    elif kind == "lowshelf":
        sq = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
        a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
    elif kind == "highshelf":
        sq = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
        a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    else:
        raise ValueError(kind)
    return np.array(b) / a[0], np.array(a) / a[0]


def eq(x, kind, f0, gain_db, q=0.707):
    b, a = _biquad(kind, f0, gain_db, q)
    return ss.lfilter(b, a, x, axis=-1)


@njit(cache=True)
def _svf(x, fc, q, mode):
    y = np.empty_like(x)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / q
    for i in range(x.shape[0]):
        f = fc[i]
        if f > 0.45 * 48000.0:
            f = 0.45 * 48000.0
        g = np.tan(np.pi * f / 48000.0)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = k * v1
        else:
            y[i] = x[i] - k * v1 - v2
    return y


def svf(x, fc, q=0.707, mode="lp"):
    """Time-varying TPT state-variable filter (fc may be a per-sample array)."""
    m = {"lp": 0, "bp": 1, "hp": 2}[mode]
    x = np.asarray(x, dtype=float)
    fcs = np.broadcast_to(np.asarray(fc, dtype=float), (x.shape[-1],)).copy()
    if x.ndim == 1:
        return _svf(x, fcs, float(q), m)
    return np.stack([_svf(np.ascontiguousarray(c), fcs, float(q), m) for c in x])


# ----------------------------------------------------------------- Karplus-Strong
@njit(cache=True)
def _ks(exc, n, L, rho, damp):
    y = np.zeros(n)
    ne = exc.shape[0]
    for i in range(n):
        v = exc[i] if i < ne else 0.0
        m = i - L
        if m >= 1.0:
            mi = int(m)
            fr = m - mi
            a = y[mi] * (1.0 - fr) + y[mi + 1] * fr if mi + 1 < i else y[mi]
            b = y[mi - 1] * (1.0 - fr) + y[mi] * fr
            v += rho * ((1.0 - damp) * a + damp * b)
        y[i] = v
    return y


def pluck(f, dur, t60=1.0, damp=0.35, bright=4000.0, rng=None, pos=0.13):
    """Karplus-Strong plucked string.  t60 = decay time of the fundamental."""
    rng = rng or np.random.default_rng(0)
    n = int(dur * SR)
    per = SR / f
    ne = max(int(per), 4)
    exc = lp(rng.uniform(-1, 1, ne), min(bright, 0.45 * SR), 1)
    # pluck-position comb (removes some harmonics -> more string-like)
    d = max(int(pos * per), 1)
    exc = exc - np.concatenate([np.zeros(d), exc[:-d]])[:ne]
    rho = 0.001 ** (1.0 / (f * t60))
    y = _ks(exc, n, per - damp, rho, damp)
    y = hp(y, 30, 1)
    return fade(y, 0.0015, 0.03)


# ----------------------------------------------------------------- tuned percussion
def partials(f, n, parts, rng=None, att=0.002):
    """Additive decaying partials: parts = [(ratio, amp, tau_seconds), ...]."""
    rng = rng or np.random.default_rng(1)
    t = tvec(n)
    y = np.zeros(n)
    for r, a, tau in parts:
        if f * r < 0.45 * SR:
            y += a * np.sin(TWO_PI * f * r * t + rng.uniform(0, TWO_PI)) * np.exp(-t / tau)
    return fade(y, att, 0.03)


def fm_bell(f, dur=4.0, ratio=1.4, index=2.2, tau=1.6, itau=0.5, rng=None):
    """Chowning-style FM bell with a decaying modulation index."""
    n = int(dur * SR)
    t = tvec(n)
    mod = index * np.exp(-t / itau) * np.sin(TWO_PI * f * ratio * t)
    y = np.sin(TWO_PI * f * t + mod) * np.exp(-t / tau)
    y += 0.25 * np.sin(TWO_PI * f * 2.76 * t) * np.exp(-t / (tau * 0.3))   # strike tone
    y += 0.2 * np.sin(TWO_PI * f * 0.5 * t) * np.exp(-t / (tau * 1.4))    # hum
    return fade(y, 0.001, 0.2)


# ----------------------------------------------------------------- reverb
def make_ir(t60=2.5, predelay=0.02, seed=0, lf=1.25, hf=0.45, early=10, density_ms=12.0):
    """Stereo synthetic room IR: decorrelated noise with frequency-dependent
    exponential decay + sparse early reflections.  Energy-normalised."""
    rng = np.random.default_rng(seed)
    L = int((t60 * 1.15 + predelay) * SR)
    t = tvec(L)
    ir = np.zeros((2, L))
    pd = int(predelay * SR)
    for c in range(2):
        nz = rng.standard_normal(L - pd)
        lo = lp(nz, 450)
        hi = hp(nz, 4500)
        mid = nz - lo - hi
        tt = t[:L - pd]
        dec = lambda T: np.exp(-6.908 * tt / T)
        tail = 1.1 * lo * dec(t60 * lf) + mid * dec(t60) + 0.8 * hi * dec(t60 * hf)
        tail *= 1 - np.exp(-tt / (density_ms / 1000))           # diffuse build-up
        ir[c, pd:] = tail * 0.6
        for k in range(early):
            ti = int(rng.uniform(0.004, 0.07) * SR)
            if ti < L:
                ir[c, ti] += rng.choice([-1, 1]) * rng.uniform(0.4, 1.0) * np.exp(-ti / SR / 0.05) * 4
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    return ir


def reverb(x, ir, width=1.0):
    """Convolve a stereo (or mono) signal with a stereo IR; output same length."""
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    n = x.shape[1]
    y = np.stack([ss.oaconvolve(x[c], ir[c])[:n] for c in range(2)])
    if width != 1.0:
        mid = 0.5 * (y[0] + y[1])
        side = 0.5 * (y[0] - y[1]) * width
        y = np.stack([mid + side, mid - side])
    return y


# ----------------------------------------------------------------- dynamics
@njit(cache=True)
def _follow(x, att, rel):
    y = np.empty_like(x)
    s = 0.0
    for i in range(x.shape[0]):
        v = x[i]
        c = att if v > s else rel
        s = v + c * (s - v)
        y[i] = s
    return y


def follower(x, attack=0.005, release=0.1):
    """Peak-ish envelope follower with separate attack/release (seconds)."""
    a = np.exp(-1.0 / (attack * SR))
    r = np.exp(-1.0 / (release * SR))
    return _follow(np.abs(np.asarray(x, dtype=float)), a, r)


def smooth_gain(g, attack, release, rate=SR):
    """One-pole smoothing of a gain curve (lower = attack, higher = release)."""
    a = np.exp(-1.0 / (attack * rate))
    r = np.exp(-1.0 / (release * rate))
    g = np.asarray(g, dtype=float)
    # _follow is 'rises with att' so run on the negated curve
    return -_follow_signed(-g, a, r)


@njit(cache=True)
def _follow_signed(x, att, rel):
    y = np.empty_like(x)
    s = x[0]
    for i in range(x.shape[0]):
        v = x[i]
        c = att if v > s else rel
        s = v + c * (s - v)
        y[i] = s
    return y


def compress(x, thresh_db=-20.0, ratio=3.0, attack=0.005, release=0.12, knee=6.0, makeup_db=0.0):
    """Feed-forward soft-knee compressor (mono or linked stereo)."""
    x = np.asarray(x, dtype=float)
    det = np.max(np.abs(np.atleast_2d(x)), axis=0)
    lvl = db(follower(det, attack, release))
    over = lvl - thresh_db
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    return x * undb(-gr + makeup_db)


def deess(x, f0=6500.0, thresh_db=-30.0, ratio=4.0):
    """Split-band de-esser: compress only the band above f0."""
    hi = hp(x, f0, 2)
    lo = x - hi
    env = db(follower(hi, 0.002, 0.06))
    gr = np.maximum(env - thresh_db, 0) * (1 - 1 / ratio)
    return lo + hi * undb(-gr)


@njit(cache=True)
def _release(g, r):
    y = np.empty_like(g)
    s = 1.0
    for i in range(g.shape[0]):
        v = g[i]
        if v < s:
            s = v
        else:
            s = v + r * (s - v)
        y[i] = s
    return y


def true_peak(x, os=4):
    x = np.atleast_2d(x)
    return max(np.max(np.abs(ss.resample_poly(c, os, 1))) for c in x)


def limiter(x, ceiling_db=-1.3, lookahead=0.004, release=0.12, os=4):
    """Look-ahead brick-wall limiter working on 4x-oversampled peaks."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[1]
    up = np.max(np.abs(np.stack([ss.resample_poly(c, os, 1)[:n * os] for c in x])), axis=0)
    pk = up.reshape(n, os).max(axis=1)
    ceil = undb(ceiling_db)
    req = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    W = max(int(lookahead * SR), 1)
    # forward-looking minimum over W, then W-long moving average (never exceeds req at the peak)
    from scipy.ndimage import minimum_filter1d
    m = minimum_filter1d(req, size=W, origin=-(W // 2))
    k = np.ones(W) / W
    s = np.convolve(np.concatenate([np.full(W - 1, m[0]), m]), k, mode="valid")
    g = _release(s, np.exp(-1.0 / (release * SR)))
    g = np.minimum(g, s)
    return x * g


def lufs(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(np.atleast_2d(x).T)


# ----------------------------------------------------------------- I/O
def read_wav(path):
    sr, x = wavfile.read(str(path))
    assert sr == SR, (path, sr)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    else:
        x = x.astype(np.float64)
    return x.T if x.ndim == 2 else x


def write_wav(path, x, bits=32, seed=0):
    """Write (2, n) or (n,) float audio.  bits=16 -> PCM with TPDF dither,
    bits=32 -> IEEE float."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=float)
    data = x.T if x.ndim == 2 else x
    if bits == 16:
        rng = np.random.default_rng(seed)
        d = (rng.random(data.shape) - rng.random(data.shape))  # TPDF, +-1 LSB
        q = np.clip(np.round(data * 32767.0 + d), -32768, 32767).astype(np.int16)
        wavfile.write(str(path), SR, q)
    else:
        wavfile.write(str(path), SR, data.astype(np.float32))
    return path


def report(name, x):
    x = np.atleast_2d(x)
    pk = np.max(np.abs(x))
    print(f"[{name}] len={x.shape[1] / SR:.3f}s peak={db(pk):.2f} dBFS "
          f"rms={db(np.sqrt(np.mean(x ** 2))):.2f} dBFS dc={np.mean(x):+.2e}")
