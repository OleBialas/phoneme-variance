from pathlib import Path
import json
import numpy as np
import pandas as pd
from mtrf.model import TRF
from matplotlib import pyplot as plt

vowels = [
    "IY",
    "OW",
    "AE",
    "AA",
    "EH",
    "ER",
    "IH",
    "UH",
    "AH",
    "UW",
    "EY",
    "AY",
    "OY",
    "AW",
]

root = Path(__file__).parent.parent.absolute()
phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phoneme_codes.json")).values())
)

# Two subplots, one with scatterplot of temporal variance vs spectral variance and one with image plot of weights
fig, ax = plt.subplot_mosaic([["A", "B"], ["A", "C"]], figsize=(5, 10))
trfs = []
for trfile in (root / "results" / "trfs").glob("*.trf"):
    trf = TRF()
    trf.load(trfile)
    trfs.append(trf)
avg_trf = np.mean(trfs)
w = avg_trf.weights[-39:, :, 86]  # Channel C23, Fz like

# phoneme weight subplot
ax["A"].imshow(w)
xtick_labels = np.arange(0, 0.4, 0.05)
xticks = [np.argmin(np.abs(avg_trf.times - t)) for t in xtick_labels]
ax["A"].set_xticks(xticks, (xtick_labels * 1e3).astype(int))
ax["A"].set_xlabel("Time [ms]")
ax["A"].set_yticks(range(w.shape[0]), phoneme_codes)

# correlation map

# phoneme termporal vs spectral variance
