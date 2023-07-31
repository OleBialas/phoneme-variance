from pathlib import Path
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

# Two subplots, one with scatterplot of temporal variance vs spectral variance and one with image plot of weights
fig, ax = plt.subplot_mosaic([["A", "B"], ["A", "C"]])
trfs = []
for trfile in (root / "results" / "trfs").glob("*.trf"):
    trf = TRF()
    trf.load(trfile)
    trfs.append(trf)
avg_trf = np.mean(trfs)
w = avg_trf.weights[-39:, :, 86]  # Channel C23, Fz like

ax["A"].imshow(w)
tick_labels = np.arange(0, 0.4, 0.05)
ticks = [np.argmin(np.abs(avg_trf.times - t)) for t in tick_labels]
ax["A"].set_xticks(ticks, (tick_labels * 1e3).astype(int))
