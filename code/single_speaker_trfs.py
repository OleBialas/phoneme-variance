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
mne.set_log_level(verbose=False)
montage = make_standard_montage("biosemi128")
positions = np.stack([ch for ch in montage.get_positions()["ch_pos"].values()])
cfg = json.load(open(root / "code" / "trf_parameters.json"))
phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phoneme_codes.json")).keys())
)
text_grids = list((root / "raw" / "stimuli" / "single_speaker").glob("*TextGrid"))
mat_files = list((root / "results" / "spectrograms" / "single_speaker").glob("*.mat"))
text_grids.sort(), mat_files.sort()

# get spectograms and phoneme stick functions
phonemes, spectrograms = [], []
for t, m in zip(text_grids, mat_files):
    mat = loadmat(m)
    spg, fs = mat["spectrogram"], mat["Fs"][0][0]
    spectrograms.append((spg - spg.mean(axis=0)) / spg.std(axis=0))
    phoneme_grid = textgrid.TextGrid.fromFile(t)[0]
    phonemes.append(np.zeros((spectrograms[-1].shape[0], len(phoneme_codes))))
    for p in phoneme_grid:
        if p.mark[:2] in phoneme_codes:
            idx = np.where(np.asarray(phoneme_codes) == p.mark[:2])[0][0]
            start = round(p.minTime * fs)
            stop = round(p.maxTime * fs)
            phonemes[-1][start:stop, idx] = 1
onsets = []
for s in spectrograms:
    o = np.diff(s.mean(axis=1, keepdims=True), prepend=np.zeros((1, 1)), axis=0).clip(
        min=0
    )
    onsets.append((o - o.mean()) / o.std())
stimulus = []
for s, o, p in zip(spectrograms, onsets, phonemes):
    stimulus.append(np.concatenate([s, o, p], axis=1))

subjects = list((root / "raw").glob("sub-0*"))
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
    r, _ = trf.test(
        stimulus,
        response,
        fs,
        cfg["tmin"],
        cfg["tmax"],
        cfg["lambda"],
    )
    trf.save(root / "results" / "trfs" / f"{subject.name}.trf")
    np.save(root / "results" / f"{subject.name}_corrmap.npy", r)
