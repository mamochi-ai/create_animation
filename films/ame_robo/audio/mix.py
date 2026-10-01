"""Final mix for 「あめふりロボ」 (adapted from src/audio/mix.py) -> build/audio/mix.wav (48 kHz stereo 16-bit PCM, exactly 60.000 s).

  * music is side-chain ducked by the dialogue stem: -8 dB broadband (deepening
    to at most -12 dB where the score is loud, so lines stay ~9 dB on top) plus an
    extra -3 dB dip of the 1-4 kHz speech band, 120 ms lookahead, smooth
    attack (~0.15 s) / release (~0.7 s); short pauses inside a line are held.
  * SFX get a gentle -3 dB duck under dialogue.
  * loudness normalised to -16 LUFS integrated (ITU-R BS.1770 via pyloudnorm),
    then a look-ahead true-peak limiter keeps it below -1 dBTP.
"""
import _common  # noqa: F401
import numpy as np
from scipy.ndimage import maximum_filter1d

from dsp import (SR, N, BUILD, read_wav, write_wav, report, db, undb, lufs, true_peak, limiter,
                 smooth_gain, bp, fade, load_timeline, load_voices)

TARGET_LUFS = -16.0
TP_CEIL = -1.0
MUSIC_GAIN_DB = 0.0
SFX_GAIN_DB = 0.0
DIA_GAIN_DB = 0.0
DUCK_DB = -8.0
DUCK_BAND_DB = -3.0
SFX_DUCK_DB = -5.0


def fit(x):
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    out = np.zeros((2, N))
    L = min(N, x.shape[1])
    out[:, :L] = x[:, :L]
    return out


def _ctrl_rms(x, win):
    """Short-term RMS in dB at the 1 kHz control rate."""
    hop = SR // 1000
    m = np.mean(np.atleast_2d(x) ** 2, axis=0)
    w = int(win * SR)
    return db(np.sqrt(np.convolve(m, np.ones(w) / w, "same")[::hop]))


def duck_curve(dia, depth_db, bed=None, want_db=9.0, max_db=-12.0, lookahead=0.12, attack=0.15,
               release=0.7, hold=0.45, thresh_db=-46.0):
    """Linear gain curve (N,) that drops by depth_db while dialogue is present.
    With `bed` given, the depth deepens (down to max_db) where the bed would
    otherwise sit less than want_db below the dialogue."""
    hop = SR // 1000
    act = (_ctrl_rms(dia, 0.03) > thresh_db).astype(float)
    act = maximum_filter1d(act, size=int(hold * 1000))            # bridge short pauses
    depth = np.full(len(act), float(depth_db))
    if bed is not None:
        dl = maximum_filter1d(_ctrl_rms(dia, 0.3), size=1500)       # line level (held)
        bl = maximum_filter1d(_ctrl_rms(bed, 0.3), size=600)
        need = bl - (dl - want_db)                                  # dB of reduction needed
        depth = -np.clip(need, -depth_db, -max_db)
    k = int(lookahead * 1000)
    g_db = act * depth
    g_db = np.concatenate([g_db[k:], np.zeros(k)])                  # start ducking a bit early
    act = np.concatenate([act[k:], np.zeros(k)])
    g_db = smooth_gain(g_db, attack / 3, release / 3, rate=1000)
    g = np.interp(np.arange(N) / hop, np.arange(len(g_db)), g_db)
    return undb(g), act


def main():
    music = fit(read_wav(BUILD / "music.wav")) * undb(MUSIC_GAIN_DB)
    sfx = fit(read_wav(BUILD / "sfx.wav")) * undb(SFX_GAIN_DB)
    dia = fit(read_wav(BUILD / "dialogue.wav")) * undb(DIA_GAIN_DB)

    g_music, act = duck_curve(dia, DUCK_DB, bed=music + sfx)
    g_band, _ = duck_curve(dia, DUCK_BAND_DB)
    g_sfx, _ = duck_curve(dia, SFX_DUCK_DB)
    band = bp(music, 1000, 4000, 1)
    music_d = (music + band * (g_band - 1)) * g_music
    sfx_d = sfx * g_sfx

    mix = music_d + sfx_d + dia
    gain = TARGET_LUFS - lufs(mix)
    print(f"  pre-limit TP at target loudness {db(true_peak(mix * undb(gain))):.2f} dBTP")
    ceil = -1.4
    for _ in range(4):                      # limit, re-measure, refine make-up gain
        lim = limiter(mix * undb(gain), ceil, lookahead=0.004, release=0.15)
        err = TARGET_LUFS - lufs(lim)
        if db(true_peak(lim)) > TP_CEIL - 0.1:
            ceil -= 0.2
        elif abs(err) < 0.05:
            break
        gain += err
    print(f"  loudness make-up {gain:+.2f} dB")
    mix = lim
    # guarantee exact silence at the very end and zero DC
    mix -= mix.mean(axis=1, keepdims=True)
    mix = fade(mix, 0.0, 0.05)
    L = lufs(mix)
    tp = db(true_peak(mix))
    report("mix", mix)
    print(f"  integrated {L:.2f} LUFS, true peak {tp:.2f} dBTP (limiter ceiling {ceil} dB)")
    # stats of the side-chain
    print(f"  music ducked {np.mean(act > 0) * 100:.0f}% of the time, depth {DUCK_DB} dB (adaptive to {db(g_music.min()):.1f} dB) "
          f"(+{DUCK_BAND_DB} dB 1-4 kHz)")
    tl, vo = load_timeline(), {v["id"]: v["dur"] for v in load_voices()}
    bed = music_d + sfx_d
    print("  per line: dialogue-to-(ducked music+sfx) ratio, broadband / speech band 300-4k Hz")
    dia_b, bed_b = bp(dia, 300, 4000), bp(bed, 300, 4000)
    rms = lambda x, a, b: db(np.sqrt(np.mean(x[:, a:b] ** 2)))
    for ln in tl["lines"]:
        a, b = int(ln["start"] * SR), int((ln["start"] + vo[ln["id"]]) * SR)
        print(f"    {ln['id']:9s} {rms(dia, a, b) - rms(bed, a, b):+5.1f} dB / "
              f"{rms(dia_b, a, b) - rms(bed_b, a, b):+5.1f} dB")
    print(write_wav(BUILD / "mix.wav", mix, bits=16))


if __name__ == "__main__":
    main()
