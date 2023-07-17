"""
For each subject and each audio segment, predict the eeg using a trf  with spectrogram+onsets and one with
spectrigram+onsets+phonemes trained on all other segments.
"""
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
cfg = json.load(open(root / "code" / "trf_parameters.json"))
montage = make_standard_montage("biosemi128")
positions = np.stack([ch for ch in montage.get_positions()["ch_pos"].values()])
phoneme_feature_mapping = json.load(
    open(root / "code" / "phoneme_feature_mapping_1.json")
)
phonetic_feature_names = list(phoneme_feature_mapping.keys())
phoneme_codes = np.unique(np.concatenate(list(phoneme_feature_mapping.values())))
regularization = np.logspace(-1, 5, 10)

text_grids = list((root / "raw" / "stimuli" / "multi_speakers").glob("*.TextGrid"))
mat_files = list((root / "results" / "spectrograms" / "multi_speakers").glob("*.mat"))
text_grids.sort(), mat_files.sort()

# get spectograms and phoneme stick functions
phonetic_features, spectrograms = [], []
for t, m in zip(text_grids, mat_files):
    mat = loadmat(m)
    spg, fs = mat["spectrogram"], mat["Fs"][0][0]
    spectrograms.append((spg - spg.mean(axis=0)) / spg.std(axis=0))
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
onsets = []
for s in spectrograms:
    o = np.diff(s.mean(axis=1, keepdims=True), prepend=np.zeros((1, 1)), axis=0).clip(
        min=0
    )
    onsets.append((o - o.mean()) / o.std())
stimulus_s = []
for s, o in zip(spectrograms, onsets):
    stimulus_s.append(np.concatenate([s, o], axis=1))
stimulus_fs = []
for s, o, p in zip(spectrograms, onsets, phonetic_features):
    stimulus_fs.append(np.concatenate([s, o, p], axis=1))

subjects = list((root / "raw").glob("sub-1*"))
# one correlation coefficient per model, segment and subject
correlations = np.zeros((2, len(stimulus_s), len(subjects)))
for isub, subject in enumerate(subjects):
    print(f"Loading data from {subject.name}")
    response = []
    recordings = list((subject / "eeg").glob("*_eeg.vhdr"))
    channels = list((subject / "eeg").glob("*_channels.tsv"))
    recordings.sort(), channels.sort()
    for rec, chs, stm in zip(recordings, channels, stimulus_s):
        raw = read_raw_brainvision(rec, verbose=False, preload=True)
        raw.set_montage(montage)
        bads = (pd.read_csv(chs, sep="\t").status == "bad").tolist()
        raw.info["bads"] = [
            ch for ch, bad in zip(raw.info["ch_names"], bads) if bad is True
        ]
        raw = raw.interpolate_bads(verbose=False)
        raw = raw.filter(1, 20, verbose=False)
        raw = raw.resample(fs, verbose=False)
        raw = raw.set_eeg_reference("average", verbose=False)
        raw, fs = raw.get_data().T[: len(s)], raw.info["sfreq"]
        response.append((raw - raw.mean(axis=0)) / raw.std(axis=0))

    for istim, stimulus in enumerate([stimulus_s, stimulus_fs]):
        if istim == 0:
            print("Fitting acoustic model")
        elif istim == 1:
            print("Fitting acoustic-phonetic model")
        for ires in range(len(response)):
            print(f"predicting segment {ires}")
            response_val, stimulus_val = response[ires], stimulus[ires]
            response_train = response[:ires] + response[ires + 1 :]
            stimulus_train = stimulus[:ires] + stimulus[ires + 1 :]
        trf = TRF()
        trf.train(
            stimulus_train,
            response_train,
            fs,
            cfg["tmin"],
            cfg["tmax"],
            regularization,
            verbose=False,
        )
        _, r, _ = trf.predict(stimulus_val, response_val, average=False)
        correlations[istim, ires, isub] = r[cfg["channels"]].mean()

np.save(root / "results" / "multi_speakers_correlations.npy", correlations)
