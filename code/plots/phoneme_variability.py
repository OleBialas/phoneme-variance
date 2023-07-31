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

trfs = []
for trfile in (root / "results" / "trfs").glob("*.trf"):
    trf = TRF()
    trf.load(trfile)
    trfs.append(trf)
avg_trf = np.mean(trfs)
w = avg_trf.weights[17:, :, 86]  # Channel C23, Fz like
