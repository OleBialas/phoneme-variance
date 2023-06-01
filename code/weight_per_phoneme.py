from pathlib import Path
import json
import numpy as np
import pandas as pd
from mtrf.model import TRF
from mne.channels import make_standard_montage

root = Path(__file__).parent.parent.absolute()
montage = make_standard_montage("biosemi128")

phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phoneme_codes.json")).keys())
)
cfg = json.load(open(root / "code" / "trf_parameters.json"))

trfs = list((root / "results" / "trfs").glob("*sub-0*_spg_pho_ons.trf"))
trfs.sort()

weight, subject_id, phoneme = [], [], []
for it, t in enumerate(trfs):
    trf = TRF()
    trf.load(t)
    phoneme_trfs = trf.to_mne_evoked(montage)[16:-1]
    for ip, p in enumerate(phoneme_trfs):
        weight.append(np.abs(p.data).mean())
        subject_id.append(it)
        phoneme.append(phoneme_codes[ip])
data = np.stack([subject_id, weight, phoneme])

df = pd.DataFrame(data=data.T, columns=["subject_id", "weight", "phoneme"])
df.to_csv(root / "results" / "weight_per_phoneme.csv")
