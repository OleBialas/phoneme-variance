from pathlib import Path
import json
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
svar, tvar, count = [], [], []
for ip, phoneme in enumerate(phoneme_codes):
    spg = np.load(root / "results" / "aligned" / f"{phoneme}_spectrograms.npy")
    # spectral variance -> mean difference of utterances to the average phoneme
    svar.append(np.abs(spg - spg.mean(axis=0)).mean())
    # temporal variance -> mean difference of warping to the diagonal
    wrp = np.load(root / "results" / "aligned" / f"{phoneme}_warping.npy")
    diagonal = np.linspace(0, 1, wrp.shape[-1])
    tvar.append(np.abs(wrp - diagonal).mean())
    count.append(spg.shape[0])

df = pd.DataFrame(columns=["subject", "phoneme", "weight", "svar", "tvar", "count"])

trf_files = list((root / "results" / "trfs").glob("*.trf"))
trf_files.sort()
for tf in trf_files:
    subject = int(re.findall(r"\d+", tf.name)[0])
    trf = TRF()
    trf.load(tf)
    weights = np.abs(trf.weights[-39:, :, chs]).mean(axis=(1, 2))
    data = np.zeros(
        (len(weights)),
        dtype={
            "names": ("subject", "weight", "phoneme", "svar", "tvar", "count"),
            "formats": ("U10", "f8", "U10", "f8", "f8", "i8"),
        },
    )
    data["svar"] = svar
    data["tvar"] = tvar
    data["count"] = count
    data["subject"] = np.repeat(subject, len(weights))
    data["weight"] = weights
    data["phoneme"] = phoneme_codes
    df = pd.concat([df, pd.DataFrame(data)])

df.to_csv(root / "results" / f"phoneme_weights.csv")
