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

amp_var_speaker, tmp_var_speaker, n_phonemes_speaker = [], [], []
for a in audios:
    amp_var, tmp_var = [], []
    for pc in phoneme_codes:
        if (a / f"{pc}_warping.npy").exists():
            warping = np.load(a / f"{pc}_warping.npy")
            spectrograms = np.load(a / f"{pc}_spectrograms.npy")
            tmp_var.append(warping.var(axis=0).mean())
            amp_var.append(spectrograms.var(axis=1).mean())
    amp_var_speaker.append(np.mean(amp_var))
    tmp_var_speaker.append(np.mean(tmp_var))
    n_phonemes_speaker.append(len(amp_var))

audio_nr = list(range(1, len(audios) + 1))
data = np.stack([tmp_var_speaker, amp_var_speaker, audio_nr])
df = pd.DataFrame(
    data=data.T, columns=["temporal_variance", "amplitude_variance", "audio"]
)
df.to_csv(root / "results" / "variance_per_speaker.csv")
