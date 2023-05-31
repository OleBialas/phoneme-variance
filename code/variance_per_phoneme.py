""" compute spectral and temporal variance for each phoneme for the single speaker audio """

from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).parent.parent.absolute()

phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phoneme_codes.json")).keys())
)

tmp_var, amp_var, count = [], [], []
tmp_var, amp_var, count = [], [], []
for pc in phoneme_codes:
    warping = np.load(
        root / "results" / "aligned" / "single_speaker" / f"{pc}_warping.npy"
    )
    spectrograms = np.load(
        root / "results" / "aligned" / "single_speaker" / f"{pc}_spectrograms.npy"
    )
    # compute the variance across utterances, then average across time (and bands)
    tmp_var.append(warping.var(axis=0).mean())
    amp_var.append(spectrograms.var(axis=1).mean())
    count.append(spectrograms.shape[0])
for pc in phoneme_codes:
    warping = np.load(
        root / "results" / "aligned" / "single_speaker" / f"{pc}_warping.npy"
    )
    spectrograms = np.load(
        root / "results" / "aligned" / "single_speaker" / f"{pc}_spectrograms.npy"
    )
    # compute the variance across utterances, then average across time (and bands)
    tmp_var.append(warping.var(axis=0).mean())
    amp_var.append(spectrograms.var(axis=1).mean())
    count.append(spectrograms.shape[0])

# save as csv
data = np.stack([tmp_var, amp_var, count, phoneme_codes])
df = pd.DataFrame(
    data=data.T, columns=["temporal_variance", "amplitude_variance", "count", "phoneme"]
)
df.to_csv(root / "results" / "variance_per_phoneme.csv")
