"""
For each subject and each audio segment, predict the eeg using a trf  with spectrogram+onsets and one with
spectrigram+onsets+phonemes trained on all other segments.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from mne.io import read_raw_fif
import textgrid
import slab
from mtrf.model import TRF, cross_validate

root = Path(__file__).parent.parent.absolute()
cfg = json.load(open(root / "code" / "trf_parameters.json"))
phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phoneme_codes.json")).keys())
)

text_grids = list((root / "raw" / "stimuli" / "multi_speaker").glob("*TextGrid"))
spectrograms = list(
    (root / "results" / "spectrograms" / "multi_speaker").glob("*_spg.wav")
)
text_grids.sort(), spectrograms.sort()

regularization = np.logspace(-1, 5, 10)

# get spectograms and phoneme stick functions
pho, spg = [], []
for t, s in zip(text_grids, spectrograms):
    sound = slab.Sound(s)
    fs, sound = sound.samplerate, sound.data
    sound = (sound - sound.mean(axis=0)) / sound.std(axis=0)
    spg.append(sound)
    phoneme_grid = textgrid.TextGrid.fromFile(t)[0]
    pho.append(np.zeros((spg[-1].shape[0], len(phoneme_codes))))
    for p in phoneme_grid:
        if p.mark[:2] in phoneme_codes:
            idx = np.where(np.asarray(phoneme_codes) == p.mark[:2])[0][0]
            start = round(p.minTime * fs)
            stop = round(p.maxTime * fs)
            pho[-1][start:stop, idx] = 1


# compute acoustic onsets by half-wave rectification of the envelope
ons = []
for s in spg:
    o = np.diff(s.mean(axis=1, keepdims=True), prepend=np.zeros((1, 1)), axis=0).clip(
        min=0
    )
    ons.append((o - o.mean()) / o.std())
# stack spectrogram and phonemes into one vector
spg_ons = []
for s, o in zip(spg, ons):
    spg_ons.append(np.concatenate([s, o], axis=1))
# also add phonemes
spg_ons_pho = []
for s, p in zip(spg_ons, pho):
    spg_ons_pho.append(np.concatenate([s, p], axis=1))

subjects = list((root / "preprocessed").glob("sub-1*"))
for sub in subjects:
    runs = list(sub.glob("*.fif"))
    runs.sort()
    responses = []
    for irun, run in enumerate(runs):
        raw = read_raw_fif(run, verbose=False)
        responses.append(raw.get_data().T[0 : spg[irun].shape[0], :])

    r_coefs = np.zeros((len(runs), 2))
    for ifeat, (features, bands, feat_names) in enumerate(
        zip([spg_ons, spg_ons_pho], [(16, 1), (16, 1, 39)], ["spg_ons", "spg_ons_pho"])
    ):
        # find best lambda
        trf = TRF()
        trf.train(
            features,
            responses,
            raw.info["sfreq"],
            cfg["tmin"],
            cfg["tmax"],
            regularization,
            bands=bands,
        )
        best_reg = trf.regularization
        np.save(
            root / "results" / "trfs" / f"{sub.name}_{feat_names}_regularization.npy",
            best_reg,
        )

        # test the model on each segment, train it on the other ones using
        # the best values for regularization
        for iseg, (feat_test, resp_test) in enumerate(zip(features, responses)):
            feat_train = features[:iseg] + features[iseg + 1 :]
            resp_train = responses[:iseg] + responses[iseg + 1 :]
            trf = TRF()
            trf._train(
                feat_train,
                resp_train,
                raw.info["sfreq"],
                cfg["tmin"],
                cfg["tmax"],
                best_reg,
            )
            _, r, _ = trf.predict(feat_test, resp_test, average=False)
            r_coefs[iseg, ifeat] = r[cfg["channels"]].mean()
            trf.save(
                root
                / "results"
                / "trfs"
                / f"{sub.name}_{feat_names}_{str(iseg+1).zfill(2)}.trf"
            )
    np.save(root / "results" / "trfs" / f"{sub.name}_correlations.npy", r_coefs)
