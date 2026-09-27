"""Objective checks for the audio stems (we can't listen, so we measure).

    python3 src/audio/analyze.py [music sfx dialogue mix]

Prints per-scene RMS / loudness, peak, true-peak, DC, click detector, and the
dialogue-vs-music ratio during each line; writes spectrogram PNGs to build/audio/.
"""
import sys

import numpy as np
import scipy.signal as ss

from dsp import SR, BUILD, read_wav, db, lufs, true_peak, load_timeline, load_voices, hp


def section_table(x, tl):
    rows = []
    for s in tl["scenes"]:
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        seg = x[:, a:b]
        rows.append((s["id"], db(np.sqrt(np.mean(seg ** 2))), db(np.max(np.abs(seg)))))
    return rows


def clicks(x, thresh=12.0):
    """Count samples whose 2nd difference is far above the local HF level."""
    m = x.mean(axis=0)
    d2 = np.abs(np.diff(m, 2))
    loc = ss.medfilt(d2[::48], 21).repeat(48)[:len(d2)] + 1e-5
    return int(np.sum(d2 > thresh * loc + 0.02))


def spectrogram_png(x, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = x.mean(axis=0)
    f, t, S = ss.spectrogram(m, SR, nperseg=2048, noverlap=1536)
    fig, ax = plt.subplots(2, 1, figsize=(16, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax[0].pcolormesh(t, f, 10 * np.log10(S + 1e-14), shading="auto", vmin=-130, vmax=-40, cmap="magma")
    ax[0].set_yscale("symlog", linthresh=500)
    ax[0].set_ylim(30, 20000)
    ax[0].set_ylabel("Hz")
    ax[0].set_title(title)
    w = int(0.1 * SR)
    r = np.sqrt(np.convolve(m ** 2, np.ones(w) / w, "same"))
    tt = np.arange(len(m)) / SR
    ax[1].plot(tt[::240], db(r[::240]), lw=0.8)
    ax[1].set_ylim(-70, 0)
    ax[1].set_ylabel("RMS dB (100ms)")
    for s in load_timeline()["scenes"]:
        for a in ax:
            a.axvline(s["start"], color="c", lw=0.6, alpha=0.7)
    ax[1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main(names):
    tl = load_timeline()
    stems = {}
    for nm in names:
        p = BUILD / f"{nm}.wav"
        if not p.exists():
            continue
        x = np.atleast_2d(read_wav(p))
        stems[nm] = x
        print(f"== {nm}: {x.shape[1] / SR:.3f}s ch={x.shape[0]} peak={db(np.max(np.abs(x))):.2f} dBFS "
              f"TP={db(true_peak(x)):.2f} dBTP LUFS={lufs(x):.2f} DC={np.mean(x):+.1e} "
              f"clicks={clicks(x)} last10ms={db(np.max(np.abs(x[:, -480:]))):.1f} dB")
        print("   " + "  ".join(f"{sid}:{r:6.1f}" for sid, r, _ in section_table(x, tl)))
        spectrogram_png(x, BUILD / f"{nm}.png", f"{nm}.wav")
    if "dialogue" in stems and ("music_ducked" in stems or "mix" in stems):
        pass
    if "dialogue" in stems and "music" in stems:
        d, m = stems["dialogue"], stems["music"]
        vo = {v["id"]: v["dur"] for v in load_voices()}
        print("   dialogue vs (un-ducked) music RMS per line:")
        for ln in tl["lines"]:
            a, b = int(ln["start"] * SR), int((ln["start"] + vo[ln["id"]]) * SR)
            print(f"     {ln['id']}: dia {db(np.sqrt(np.mean(d[:, a:b] ** 2))):6.1f}  "
                  f"music {db(np.sqrt(np.mean(m[:, a:b] ** 2))):6.1f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["music", "sfx", "dialogue", "mix"])
