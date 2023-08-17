from pathlib import Path
from itertools import compress
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from textgrid import TextGrid
import scienceplots

plt.style.use(["science", "ieee", "no-latex", "high-vis"])

root = Path(__file__).parent.parent.parent.absolute()

phoneme = "OY"

textgrids = list((root / "raw" / "stimuli").glob("*.TextGrid"))
spectrograms = list((root / "results" / "spectrograms").glob("*_spg.mat"))
textgrids.sort(), spectrograms.sort()

aligned = np.load(root / "results" / "aligned" / f"{phoneme}_spectrograms.npy").mean(
    axis=-1
)
warping = np.load(root / "results" / "aligned" / f"{phoneme}_warping.npy")

unaligned = []
# get the envelope for each phoneme utterance
for t, s in zip(textgrids, spectrograms):
    mat = loadmat(s)
    spectrogram, Fs = mat["spectrogram"], mat["Fs"][0][0]
    phoneme_grid = TextGrid.fromFile(t)[0]
    for p in phoneme_grid:
        if p.mark[:2] == phoneme:
            start, stop = round(p.minTime * Fs), round(p.maxTime * Fs)
            unaligned.append(spectrogram[start:stop, :].mean(axis=1))

# reject outliers
lengths = np.asarray([len(s) for s in unaligned])
mask = np.abs(lengths - lengths.mean()) < 2 * lengths.std()
unaligned = list(compress(unaligned, mask))
print(f"removed {len(lengths)-len(unaligned)} outliers")

# pad to same length so that each phoneme starts and ends with 0
max_len = max([len(s) for s in unaligned])
for i, env in enumerate(unaligned):
    diff = max_len - len(env)
    unaligned[i] = np.concatenate([np.zeros(1), env, np.zeros((diff + 1))])
unaligned = np.stack(unaligned)
# set all negative values to 0
unaligned = unaligned.clip(min=0)
idx = [2, 30, 55, 84, 92]
# idx = np.random.choice(len(aligned), 6, replace=False)  # curves to plot
times = np.arange(aligned.shape[-1]) * 1 / Fs
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
for i in idx:
    ax[0, 0].plot(times, unaligned[i], linestyle="-")
    ax[1, 0].plot(times, aligned[i], linestyle="-")
    ax[1, 1].plot(np.linspace(0, 1, warping.shape[-1]), warping[i], linestyle="-")
ax[0, 1].hist(lengths[mask] * 1 / Fs, bins=100, color="black", alpha=0.7)
ax[0, 0].plot(times, unaligned.mean(axis=0), color="black", linestyle="--")
ax[1, 0].plot(times, aligned.mean(axis=0), color="black", linestyle="--")
ax[1, 1].plot([0, 1], [0, 1], color="black", linestyle="--")

ax[0, 0].set(xlabel="Time [s]", ylabel="Amplitude [a.u.]")
ax[0, 1].set(xlabel="Duration [s]", ylabel="Number of occurrences")
ax[1, 0].set(xlabel="Time [s]", ylabel="Amplitude [a.u.]")
ax[1, 1].set(xlabel="Input time [a.u.]", ylabel="Output time [a.u.]")

for ax, label in zip(ax.flatten(), ["A", "B", "C", "D"]):
    ax.text(
        -0.02,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )


fig.savefig(root / "results" / "plots" / "warping.png", dpi=300)
plt.close("all")
