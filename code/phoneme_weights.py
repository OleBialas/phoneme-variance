from pathlib import Path
import json
import pandas as pd
import numpy as np
from mtrf.model import TRF

root = Path(__file__).parent.parent.absolute()
phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phoneme_codes.json")).keys())
)
cfg = json.load(open(root / "code" / "trf_parameters.json"))

chs = [106, 107, 108, 115, 116, 117, 54, 55, 56, 61, 62, 63]

# first, compute the average phoneme weight for each subject and average spectral distance
# and temporal warping across all utterances of the same phoneme
svar, tvar, count = [], [], []
directory = root / "results" / "aligned" / "single_speaker"
for ip, phoneme in enumerate(phoneme_codes):
    spg = np.load(directory / f"{phoneme}_spectrograms.npy")
    # spectral variance -> mean difference of utterances to the average phoneme
    svar.append(np.abs(spg - spg.mean(axis=0)).mean())
    # temporal variance -> mean difference of warping to the diagonal
    wrp = np.load(directory / f"{phoneme}_warping.npy")
    diagonal = np.linspace(0, 1, wrp.shape[-1])
    tvar.append(np.abs(wrp - diagonal).mean())
    count.append(spg.shape[0])

for model in ["pho", "spg_pho", "spg_pho_ons"]:
    df = pd.DataFrame(columns=["subject", "phoneme", "weight", "svar", "tvar", "count"])

    trf_files = list((root / "results" / "trfs").glob(f"sub-0[0-9][0-9]_{model}.trf"))
    trf_files.sort()
    for tf in trf_files:
        subject = tf.name.split("_")[0]
        trf = TRF()
        trf.load(tf)
        if model == "pho":
            weights = np.abs(trf.weights[:, :, chs]).mean(axis=(1, 2))
        elif model == "spg_pho":
            weights = np.abs(trf.weights[16:, :, chs]).mean(axis=(1, 2))
        elif model == "spg_pho_ons":
            weights = np.abs(trf.weights[16:, :, chs]).mean(axis=(1, 2))

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

    df.to_csv(root / "results" / f"phoneme_weights_{model}.csv")
