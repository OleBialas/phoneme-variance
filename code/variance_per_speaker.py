""" compute spectral and temporal variance for each phoneme for the single speaker audio """

from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).parent.parent.absolute()

phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phoneme_codes.json")).keys())
)
audios = list((root / "results" / "aligned" / "multi_speakers").glob("audio*"))
audios.sort()
r = np.load(root / "results" / "multi_speakers_correlations.npy")
n_subjects = r.shape[-1]
df = pd.DataFrame(
    columns=[
        "tmp_var",
        "amp_var",
        "audio",
        "subject",
        "r_s",
        "r_fs",
    ]
)
for ia, a in enumerate(audios):
    amp_var, tmp_var = [], []
    for pc in phoneme_codes:
        if (a / f"{pc}_warping.npy").exists():
            wrp = np.load(a / f"{pc}_warping.npy")
            spg = np.load(a / f"{pc}_spectrograms.npy")
            amp_var.append(np.abs(spg - spg.mean(axis=0)).mean(axis=(1, 2)))
            tmp_var.append(np.abs(wrp - np.linspace(0, 1, wrp.shape[-1])).mean(axis=1))

    for isub in range(n_subjects):
        df.loc[len(df)] = [
            np.concatenate(tmp_var).mean(),
            np.concatenate(amp_var).mean(),
            ia + 1,
            isub + 1,
            r[0, ia, isub],
            r[1, ia, isub],
        ]

df.to_csv(root / "results" / "variance_per_speaker.csv")
