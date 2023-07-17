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
fs = 128
cfg = json.load(open(root / "code" / "trf_parameters.json"))
montage = make_standard_montage("biosemi128")
positions = np.stack([ch for ch in montage.get_positions()["ch_pos"].values()])
phoneme_feature_mapping = json.load(
    open(root / "code" / "phoneme_feature_mapping_1.json")
)
phonetic_feature_names = list(phoneme_feature_mapping.keys())
phoneme_codes = np.unique(np.concatenate(list(phoneme_feature_mapping.values())))
regularization = np.logspace(-1, 5, 10)

text_grids = list((root / "raw" / "stimuli" / "single_speaker").glob("*.TextGrid"))
text_grids.sort(), mat_files.sort()

# get spectograms and phoneme stick functions
phonetic_features = []
for t, m in zip(text_grids, mat_files):
    mat = loadmat(m)
    phoneme_grid = textgrid.TextGrid.fromFile(t)[0]
    phonetic_features.append(
        np.zeros((spectrograms[-1].shape[0], len(phonetic_feature_names)))
    )
    for p in phoneme_grid:
        if p.mark[:2] in phoneme_codes:
            start = round(p.minTime * fs)
            stop = round(p.maxTime * fs)
            for i, p_list in enumerate(phoneme_feature_mapping.values()):
                if p.mark[:2] in p_list:
                    phonetic_features[-1][start:stop, i] = 1

stimulus = phonetic_features
subjects = list((root / "raw").glob("sub-0*"))
trfs = []
for subject in subjects:
    response = []
    recordings = list((subject / "eeg").glob("*_eeg.vhdr"))
    channels = list((subject / "eeg").glob("*_channels.tsv"))
    recordings.sort(), channels.sort()
    for rec, chs, stm in zip(recordings, channels, stimulus):
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
    trf.train(
        stimulus,
        response,
        fs,
        cfg["tmin"],
        cfg["tmax"],
        cfg["lambda"],
    )
    trfs.append(trf)
