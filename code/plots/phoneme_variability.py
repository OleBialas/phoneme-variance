from pathlib import Path
import json
import numpy as np
import pandas as pd
from mtrf.model import TRF
from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
import scienceplots
from adjustText import adjust_text

plt.style.use(["science", "ieee", "no-latex", "high-vis"])

root = Path(__file__).parent.parent.absolute()
raw = read_raw_brainvision(
    root / "raw" / "sub-001" / "eeg" / "sub-001_task-listening_run-01_eeg.vhdr"
)
raw = raw.set_montage(make_standard_montage("biosemi128"))

phonemes = json.load(open(root / "code" / "phonemes.json"))
channels = json.load(open(root / "code" / "trf_parameters.json"))["channels"]
mask = np.repeat(False, 128)
mask[channels] = True

ipa_codes = [p["ipa"] for p in phonemes.values()]
is_vowel = np.asarray([p["manner"] == "vowel" for p in phonemes.values()])
# divide consonants into voiced and unvoiced
is_voiced = np.asarray(
    [p["manner"] != "vowel" and p["voicing"] == "voiced" for p in phonemes.values()]
)
is_voiceless = np.asarray(
    [p["manner"] != "vowel" and p["voicing"] == "voiceless" for p in phonemes.values()]
)

label_phonemes = ["ʌ", "i", "æ", "ɛ", "oʊ", "ʊ", "t"]
labels = []
for i in ipa_codes:
    if i in label_phonemes:
        labels.append(i)
    else:
        labels.append("")

# Two subplots, one with scatterplot of temporal variance vs spectral variance and one with image plot of weights
fig, ax = plt.subplot_mosaic([["A", "B"], ["A", "C"]], figsize=(5, 10))

# phoneme weight subplot
trfs = []
for trfile in (root / "results" / "trfs").glob("*.trf"):
    trf = TRF()
    trf.load(trfile)
    trfs.append(trf)
avg_trf = np.mean(trfs)
w = avg_trf.weights[-39:, :, 86]  # Channel C23, Fz like
img = ax["A"].imshow(w, aspect=3)
xtick_labels = np.arange(0, 0.4, 0.05)
xticks = [np.argmin(np.abs(avg_trf.times - t)) for t in xtick_labels]
ax["A"].set_xticks(xticks, (xtick_labels * 1e3).astype(int))
ax["A"].set_xlabel("Time [ms]")
ax["A"].set_yticks(range(w.shape[0]), ipa_codes)

cbar = fig.colorbar(img, ticks=[-0.025, 0.025], shrink=0.2)
cbar.ax.set_yticklabels(["-", "+"])  # vertically oriented colorbar
cbar.set_label("Weight [a.u.]", labelpad=-5)
ax["A"].minorticks_off()
ax["A"].tick_params(axis="x", which="both", top=False, bottom=True)
ax["A"].tick_params(axis="y", which="both", right=False, left=True)

# correlation map subplot
maps = []
for corr in (root / "results" / "correlations").glob("*_corrmap.npy"):
    maps.append(np.load(corr))
img, _ = plot_topomap(np.mean(maps, axis=0), raw.info, axes=ax["B"], show=False)
cbar = fig.colorbar(img, ticks=[0, 0.05], shrink=0.2)
cbar.set_label("Accuracy [r]", labelpad=-5)

# phoneme termporal vs spectral variance
df = pd.read_csv(root / "results" / "variance_per_phoneme.csv")
tmp_var = df.temporal_variance / df.temporal_variance.max()
amp_var = df.amplitude_variance / df.amplitude_variance.max()
size = (df["count"] / df["count"].max()) * 30

ax["C"].scatter(  # vowels
    tmp_var[is_vowel],
    amp_var[is_vowel],
    s=size[is_vowel],
    facecolors="none",
    edgecolors="C3",
    label="vowels",
    alpha=0.7,
)

ax["C"].scatter(  # voiced consonants
    tmp_var[is_voiced],
    amp_var[is_voiced],
    s=size[is_voiced],
    color='C4'
    label="voiced",
    alpha=0.7,
)


ax["C"].scatter(  #voiceless consonants
    tmp_var[is_voiceless],
    amp_var[is_voiceless],
    s=size[is_voiceless],
    facecolors="none",
    edgecolors="C4",
    label="voiceless",
    alpha=0.7,
)

texts = []
for i, txt in enumerate(labels):
    if not txt == "":
        texts.append(
            ax["C"].text(tmp_var[i] + 0.01, amp_var[i] + 0.01, txt, fontsize=8)
        )
adjust_text(texts[:-1])


ax["C"].set_aspect("equal")
ax["C"].set(
    xticks=[0, 1],
    xlabel="Temporal variability [a.u.]",
    yticks=[0, 1],
    ylabel="Amplitude variability [a.u.]",
)
lgnd = ax["C"].legend(handletextpad=-0.6, loc="upper right", borderpad=-0.3)
lgnd.legend_handles[0]._sizes = [10]
lgnd.legend_handles[1]._sizes = [10]

plt.subplots_adjust(hspace=-0.7)

for label, x, y in zip(["A", "B", "C"], [-0.1, 0, 0], [1.05, 1.05, 1.15]):
    ax[label].text(
        x,
        y,
        label,
        transform=ax[label].transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )

fig.savefig(root / "results" / "plots" / "phoneme_variability.png", dpi=300)
