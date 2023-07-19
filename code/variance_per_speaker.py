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
min_n = 20  # minimum number of utterances neccessary for a phoneme to be considered

n_phonemes = []
amp_var_speaker, tmp_var_speaker, n_phonemes_speaker = [], [], []
for a in audios:
    amp_var, tmp_var = [], []
    for pc in phoneme_codes:
        if (a / f"{pc}_warping.npy").exists():
            warping = np.load(a / f"{pc}_warping.npy")
            spectrograms = np.load(a / f"{pc}_spectrograms.npy")
            if spectrograms.shape[0] >= min_n:
                # spectral variance -> mean difference of utterances to the average phoneme
                amp_var.append(np.abs(spectrograms - spectrograms.mean(axis=0)).mean())
                # temporal variance -> mean difference of warping to the diagonal
                diagonal = np.linspace(0, 1, warping.shape[-1])
                tmp_var.append(np.abs(warping - diagonal).mean())
    amp_var_speaker.append(np.mean(amp_var))
    tmp_var_speaker.append(np.mean(tmp_var))
    n_phonemes_speaker.append(len(amp_var))

correlations = np.load(root / "results" / "multi_speakers_correlations.npy")
delta_r = correlations[1] - correlations[0]

df = pd.DataFrame(columns=["temp_var", "amp_var", "audio_nr", "subject_nr", "delta_r"])

for isub in range(delta_r.shape[-1]):
    sub_delta_r = delta_r[:, isub]
    sub_data = pd.DataFrame(
        {
            "temp_var": tmp_var_speaker,
            "amp_var": amp_var_speaker,
            "audio_nr": np.arange(1, len(amp_var_speaker) + 1),
            "subject_nr": isub + 1,
            "delta_r": sub_delta_r,
        }
    )
    df = pd.concat([df, sub_data])

df.to_csv(root / "results" / "variance_per_speaker.csv")
