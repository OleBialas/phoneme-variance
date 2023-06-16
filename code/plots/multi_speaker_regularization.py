#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress
from matplotlib import pyplot as plt

plt.style.use("science")


def line(a, b, x):
    return a + b * x


root = Path(__file__).parent.parent.absolute()

var = pd.read_csv(root / "results" / "variance_per_speaker.csv")["temporal_variance"]

# load the correlation coefficients for each subject
files = list((root / "results" / "trfs").glob("sub-1[0-9][0-9]_correlations.npy"))
data = np.zeros((len(files), 14, 2))
for i, fi in enumerate(files):
    data[i, :, :] = np.load(fi)

slopes = []
for d in data:
    diff = (d[:, 1] - d[:, 0]) / (d[:, 1] + d[:, 0])
    plt.scatter(var, diff)
    b, a = linregress(var, diff)[:2]
    slopes.append(b)
    x = np.linspace(var.min(), var.max(), 10)
    plt.plot(x, line(a, b, x))
