from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

root = Path(__file__).parent.parent.absolute()
plt.style.use(["science", "no-latex"])
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
bar_width = 0.35

phoneme_dict = json.load(open(root / "code" / "phoneme_codes.json"))
phoneme_codes = np.asarray(list(phoneme_dict.keys()))
phoneme_labels = np.asarray(list(phoneme_dict.values()))

df = pd.read_csv(root / "results" / "variance_per_phoneme.csv")
df.temporal_variance /= df.temporal_variance.max()
df.amplitude_variance /= df.amplitude_variance.max()

x = np.linspace(0, 2 * (len(phoneme_codes) - 1), len(phoneme_codes))
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.bar(x, df.amplitude_variance, bar_width, color=colors[0])
ax1.bar(x + 0.5, df.temporal_variance, bar_width, color=colors[1])
ax2.bar(x + 1.0, df["count"], bar_width, color=colors[2])

ax1.set_xticks(x + 0.25, phoneme_labels, minor=False)
ax1.set_xlim(-0.5, x.max() + 2)
ax1.set_ylabel("Variance [a.u.]")
ax2.set_ylabel("Phoneme count")

patch1 = mpatches.Patch(color=colors[0], label="Amplitude variance")
patch2 = mpatches.Patch(color=colors[1], label="Temporal Variance")
patch3 = mpatches.Patch(color=colors[2], label="Count")
ax1.legend(handles=[patch1, patch2, patch3], fontsize="small", loc=(0.6, 0.8))

ax1.tick_params(axis="x", which="both", top=False)
ax1.tick_params(axis="x", which="minor", bottom=False)

fig.savefig(
    root / "results" / "plots" / "variance_py_phoneme.png", dpi=300, bbox_inches="tight"
)
