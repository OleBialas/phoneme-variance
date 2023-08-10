from pathlib import Path
import json
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scienceplots

plt.style.use(["science", "ieee", "no-latex", "high-vis"])

root = Path(__file__).parent.parent.parent.absolute()


phonemes = json.load(open(root / "code" / "phonemes.json"))
ipa_codes = [p["ipa"] for p in phonemes.values()]
is_vowel = np.asarray([p["manner"] == "vowel" for p in phonemes.values()])
vowels = np.asarray(list(phonemes.keys()))[is_vowel]
consonants = np.asarray(list(phonemes.keys()))[~is_vowel]

df_phonemes = pd.read_csv(root / "results" / "phoneme_weights_predictions.csv")
df_phonemes.weight -= df_phonemes.weight.mean()
df_phonemes.weight /= df_phonemes.weight.std()
df_vowels = df_phonemes[df_phonemes.phoneme.isin(vowels)]
df_consonants = df_phonemes[~df_phonemes.phoneme.isin(vowels)]

df_post = pd.read_csv(root / "results" / "posterior_samples.csv")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

divider = make_axes_locatable(ax[0])
ax = np.append(ax, divider.append_axes("right", size="100%", pad=0.1))

colors = ["C0", "C1", "C2"]
labels = ["temporal var.", "amplitude var.", "occurrence"]
patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
ax[2].legend(handles=patches, loc=(0, 0.8))

[a.minorticks_off() for a in ax]
ax[0].set(
    xlim=(-0.15, 0.23),
    ylim=(0, 19),
    ylabel="Density [a.u.]",
    xlabel="\u03B2-Weight [a.u.]",
)
ax[0].xaxis.set_label_coords(1.0, -0.1)
ax[2].set(xlim=(0.47, 0.75), ylim=(0, 19))

# hide the spines between ax[0] and ax[2]
ax[0].spines["right"].set_visible(False)
ax[2].spines["left"].set_visible(False)
ax[0].yaxis.tick_left()
ax[2].yaxis.tick_right()
ax[2].yaxis.set_ticklabels([])

d = 0.010  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax[0].transAxes, color="k", clip_on=False)
ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax[2].transAxes)  # switch to the bottom axes
ax[2].plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax[2].plot((-d, +d), (-d, +d), **kwargs)

density = gaussian_kde(df_post.amplitude)
x = np.linspace(0, 0.22, 200)
y = density(x)
ax[0].plot(x, density(x), color="C0", linestyle="-", linewidth=2)
ax[0].vlines(x[np.argmax(y)], ymin=0, ymax=y.max(), linestyle="--", color="C0")

density = gaussian_kde(df_post.phase)
x = np.linspace(-0.15, 0.1, 200)
y = density(x)
ax[0].plot(x, density(x), color="C1", linestyle="-", linewidth=2)
ax[0].vlines(x[np.argmax(y)], ymin=0, ymax=y.max(), linestyle="--", color="C1")

density = gaussian_kde(df_post["count"])
x = np.linspace(0.5, 0.76, 200)
y = density(x)
ax[2].plot(x, density(x), color="C2", linestyle="-", linewidth=2)
ax[2].vlines(x[np.argmax(y)], ymin=0, ymax=y.max(), linestyle="--", color="C2")

ax[1].scatter(df_vowels.weight, df_vowels.weight_sim, color="C3", label="vowels")
ax[1].scatter(
    df_consonants.weight, df_consonants.weight_sim, color="C4", label="consonants"
)

ax[1].set(
    xlim=(-2, 5.5), ylim=(-2, 5.5), xlabel="Observed weight", ylabel="Predicted weight"
)
ax[1].plot([-2, 5], [-2, 5], color="black", linestyle="--")
ax[1].set_aspect("auto")


u = 2.5  # x-position of the center
v = 2.5  # y-position of the center
a = 3.1  # radius on the x-axis
b = 0.8  # radius on the y-axis
t = np.linspace(0, 2 * np.pi, 100)
ax[1].plot(u + a * np.cos(t), v + b * np.sin(t), color="k")
ax[1].annotate(text="ʌ", xy=(0, 3.3), fontsize=16)
ax[1].legend(loc="lower right")

for a, label, x, y in zip(ax[0:2], ["A", "B"], [0, 0], [1.05, 1.05]):
    a.text(
        x,
        y,
        label,
        transform=a.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )

var_explained = sum(df_phonemes.weight - df_phonemes.weight_sim)**2)/len(df_phonemes)
r2 = (df_phonemes.weight.var()-var_explained)/df_phonemes.weight.var()
print(f'R^2={r2}')

fig.savefig(root / "results" / "plots" / "weight_model.png", dpi=300)
