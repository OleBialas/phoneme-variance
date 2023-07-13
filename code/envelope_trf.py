""" Sanity check: compute envelope TRF and plot average time course, topography and distribution of correlation coefficients """

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
import textgrid
from mtrf.model import TRF, cross_validate

root = Path(__file__).parent.parent.absolute()
montage = make_standard_montage("biosemi128")
cfg = json.load(open(root / "code" / "trf_parameters.json"))
cfg["lambda"] = 10000
positions = np.stack([ch for ch in montage.get_positions()["ch_pos"].values()])

for data_set, subject_id in zip(
    ["single_speaker", "multi_speakers"], ["sub-0*", "sub-1*"]
):
    mat_files = list((root / "results" / "spectrograms" / data_set).glob("*.mat"))
    subjects = list((root / "raw").glob(subject_id))
    mat_files.sort(), subjects.sort()

    trfs, correlations = [], []
    for subject in subjects:
        stimulus, response = [], []
        for m in mat_files:
            mat = loadmat(m)
            spg, fs = mat["spectrogram"], mat["Fs"][0][0]
            stimulus.append((spg - spg.mean(axis=0)) / spg.std(axis=0))

        recordings = list((subject / "eeg").glob("*_eeg.vhdr"))
        channels = list((subject / "eeg").glob("*_channels.tsv"))
        recordings.sort(), channels.sort()
        for rec, chs in zip(recordings, channels):
            raw = read_raw_brainvision(rec, verbose=False, preload=True)
            raw = raw.set_montage(montage)
            bads = (pd.read_csv(chs, sep="\t").status == "bad").tolist()
            raw.info["bads"] = [
                ch for ch, bad in zip(raw.info["ch_names"], bads) if bad is True
            ]
            raw = raw.interpolate_bads()
            raw = raw.filter(1, 20)
            raw = raw.resample(fs)
            raw = raw.set_eeg_reference("average")
            raw, info = raw.get_data().T, raw.info
            raw = (raw - raw.mean(axis=0)) / raw.std(axis=0)
            response.append(raw)

        for i, (s, r) in enumerate(zip(stimulus, response)):
            if len(s) > len(r):
                stimulus[i] = stimulus[i][: len(r)]
            if len(r) > len(s):
                response[i] = response[i][: len(s)]

        idx = np.random.choice(len(stimulus), len(stimulus), replace=False)
        trf = TRF()
        trf.train(
            [stimulus[i] for i in idx[:-1]],
            [response[i] for i in idx[:-1]],
            fs,
            cfg["tmin"],
            cfg["tmax"],
            cfg["lambda"],
        )
        _, r, _ = trf.predict(stimulus[idx[-1]], response[idx[-1]], average=False)
        correlations.append(r)
        trfs.append(trf)

plot_topomap(np.mean(correlations, axis=0), info)
avg_trf = np.mean(trfs)
avg_trf = avg_trf.to_mne_evoked(montage)[0]
avg_trf.plot_joint(times=[0.08, 0.165])
