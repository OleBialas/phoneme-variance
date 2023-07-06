from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
import textgrid
from mtrf.model import TRF, cross_validate

root = Path(__file__).parent.parent.absolute()
montage = make_standard_montage("biosemi128")
positions = np.stack([ch for ch in montage.get_positions()["ch_pos"].values()])
mat_files = list((root / "results" / "spectrograms" / "single_speaker").glob("*.mat"))
mat_files.sort()

# get spectograms and phoneme stick functions
stimulus = []
for m in mat_files:
    mat = loadmat(m)
    env, fs = mat["spectrogram"].mean(axis=1), mat["Fs"][0][0]
    stimulus.append((env - env.mean()) / env.std())

trfs = []
subjects = list((root / "raw").glob("sub-0*"))
for subject in subjects:
    response = []
    recordings = list((subject / "eeg").glob("*_eeg.vhdr"))
    channels = list((subject / "eeg").glob("*_channels.tsv"))
    recordings.sort(), channels.sort()
    for rec, chs in zip(recordings, channels):
        raw = read_raw_brainvision(rec, verbose=False, preload=True)
        raw.set_montage(montage)
        bads = (pd.read_csv(chs, sep="\t").status == "bad").tolist()
        raw.info["bads"] = [
            ch for ch, bad in zip(raw.info["ch_names"], bads) if bad is True
        ]
        raw = raw.interpolate_bads()
        raw = raw.filter(1, 20)
        raw = raw.resample(fs)
        raw = raw.set_eeg_reference("average")
        raw, fs = raw.get_data().T, raw.info["sfreq"]
        response.append((raw - raw.mean(axis=0)) / raw.std(axis=0))
    for i, (s, r) in enumerate(zip(stimulus, response)):
        if len(s) > len(r):
            stimulus[i] = stimulus[i][: len(r)]
        if len(r) > len(s):
            response[i] = response[i][: len(s)]

    trf = TRF()
    trf.train(stimulus, response, fs, -0.1, 0.4, 10)
    trfs.append(trf)

avg_trf = np.mean(trfs)
avg_trf = avg_trf.to_mne_evoked(montage)[0]
