from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import textgrid
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
from mne import set_log_level
from mtrf.model import TRF

root = Path(__file__).parent.parent.absolute()

# sort phonemes by number of occurrences then iterate trough
# them in steps of 2 and separate them into high and low variability
df = pd.read_csv(root / "results" / "variance_per_phoneme.csv")
idx = np.argsort(df["count"])[::-1]
df = df.iloc[idx]
variant_phonemes = [
    0,
    3,
    6,
    7,
    9,
    11,
    13,
    16,
    18,
    20,
    21,
    22,
    23,
    26,
    27,
    28,
    31,
    32,
    36,
]
invariant_phonemes = [
    1,
    2,
    4,
    5,
    8,
    10,
    12,
    14,
    15,
    17,
    19,
    24,
    25,
    29,
    30,
    33,
    34,
    35,
    37,
    38,
]

text_grids = list((root / "raw" / "stimuli").glob("*TextGrid"))
mat_files = list((root / "results" / "spectrograms").glob("*.mat"))
subjects = list((root / "raw").glob("sub-0*"))
text_grids.sort(), mat_files.sort()

set_log_level(verbose=False)
montage = make_standard_montage("biosemi128")
positions = np.stack([ch for ch in montage.get_positions()["ch_pos"].values()])
cfg = json.load(open(root / "code" / "trf_parameters.json"))

r_all_subjects = np.zeros((len(subjects), 2))
# get spectograms and phoneme stick functions
for pc_i, (phoneme_codes, phoneme_name) in enumerate(
    zip(
        [
            df.iloc[variant_phonemes].phoneme,
            df.iloc[invariant_phonemes].phoneme,
        ],
        ["variant phonemes", "invariant phonemes"],
    )
):
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
        o = np.diff(
            s.mean(axis=1, keepdims=True), prepend=np.zeros((1, 1)), axis=0
        ).clip(min=0)
        onsets.append((o - o.mean()) / o.std())
    stimulus = []
    for s, o, p in zip(spectrograms, onsets, phonemes):
        stimulus.append(np.concatenate([s, o, p], axis=1))

    for sub_i, subject in enumerate(subjects):
        print(f"Subject {subject.name}, {phoneme_name} model")
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
            if len(raw.info["bads"]) > 0:
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
        r, _ = trf.train(
            stimulus, response, fs, cfg["tmin"], cfg["tmax"], cfg["lambdas"]
        )
        r_all_subjects[sub_i, pc_i] = r
        trf.save(root / "results" / "trfs" / f"{subject.name}_variant.trf")
np.save(
    root / "results" / "correlations" / "variant_invariant_corr.npy", r_all_subjects
)
