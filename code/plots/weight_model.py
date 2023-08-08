from pathlib import Path
import json
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
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
ax = np.append(ax, divider.append_axes("right", size="100%", pad=0))

density = gaussian_kde(df_post.amplitude)
xs = np.linspace(0, 0.22, 200)
plt.plot(xs, density(xs), color="C0", linestyle="-")

density = gaussian_kde(df_post.phase)
xs = np.linspace(-0.1, 0.1, 200)
plt.plot(xs, density(xs), color="C1", linestyle="-")

density = gaussian_kde(df_post["count"])
xs = np.linspace(0.5, 0.76, 200)
plt.plot(xs, density(xs), color="C1", linestyle="-")


ax[1].scatter(df_vowels.weight, df_vowels.weight_sim, color="C3")
ax[1].scatter(df_consonants.weight, df_consonants.weight_sim, color="C4")
ax[1].set(xlim=(-2, 5.5), ylim=(-2, 5.5))
ax[1].plot([-2, 5], [-2, 5], color="black", linestyle="--")
ax[1].set_aspect("auto")

fig.savefig(root / "results" / "plots" / "weight_model.png", dpi=300)
