from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
from mtrf.model import TRF

root = Path(__file__).parent.parent.absolute()
phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phonemes.json")).keys())
)
chs = json.load(open(root / "code" / "trf_parameters.json"))["channels"]

# first, compute the average phoneme weight for each subject and average spectral distance
# and temporal warping across all utterances of the same phoneme
amplitude_var, phase_var, occurrences = [], [], []
for ip, phoneme in enumerate(phoneme_codes):
    spg = np.load(root / "results" / "aligned" / f"{phoneme}_spectrograms.npy")
    # spectral variance -> mean difference of utterances to the average phoneme
    amplitude_var.append(np.abs(spg - spg.mean(axis=0)).mean())
    # temporal variance -> mean difference of warping to the diagonal
    wrp = np.load(root / "results" / "aligned" / f"{phoneme}_warping.npy")
    diagonal = np.linspace(0, 1, wrp.shape[-1])
    phase_var.append(np.abs(wrp - diagonal).mean())
    occurrences.append(spg.shape[0])

df = pd.DataFrame(
    columns=[
        "subject",
        "phoneme",
        "weight",
        "amplitude_var",
        "phase_var",
        "occurrences",
        "correlation",
    ]
)

trf_files = list((root / "results" / "trfs").glob("*.trf"))
corr_files = list((root / "results" / "correlations").glob("*.npy"))
trf_files.sort(), corr_files.sort()
for tf, cf in zip(trf_files, corr_files):
    subject = int(re.findall(r"\d+", tf.name)[0])
    trf = TRF()
    trf.load(tf)
    correlation = np.load(cf)[chs].mean()
    weights = np.abs(trf.weights[-39:, :, chs]).mean(axis=(1, 2))
    data = np.zeros(
        (len(weights)),
        dtype={
            "names": (
                "subject",
                "weight",
                "phoneme",
                "amplitude_var",
                "phase_var",
                "occurrences",
                "correlation",
            ),
            "formats": ("U10", "f8", "U10", "f8", "f8", "i8", "f8"),
        },
    )
    data["amplitude_var"] = amplitude_var
    data["phase_var"] = phase_var
    data["occurrences"] = occurrences
    data["subject"] = np.repeat(subject, len(weights))
    data["weight"] = weights
    data["phoneme"] = phoneme_codes
    data["correlation"] = correlation
    df = pd.concat([df, pd.DataFrame(data)])

df.to_csv(root / "results" / f"phoneme_weights.csv")
