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


for data_set, subject_id in zip(
    ["single_speaker", "multi_speakers"], ["sub-0*", "sub-1*"]
):

    text_grids = list((root / "raw" / "stimuli" / data_set).glob("*TextGrid"))
    spectrograms = list(
        (root / "results" / "spectrograms" / data_set).glob("*_spg.wav")
    )
    text_grids.sort(), spectrograms.sort()

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
    # stack spectrogram and phonemes into one vector
    spg_pho = []
    for s, p in zip(spg, pho):
        spg_pho.append(np.concatenate([s, p], axis=1))
    # also add the onsets
    ons = []
    for s in spg:
        o = np.diff(
            s.mean(axis=1, keepdims=True), prepend=np.zeros((1, 1)), axis=0
        ).clip(min=0)
        ons.append((o - o.mean()) / o.std())
    spg_pho_ons = []
    for s, o in zip(spg_pho, ons):
        spg_pho_ons.append(np.concatenate([o, s], axis=1))

    # compute spg and spg+pho trf for each subject
    subjects = list((root / "preprocessed").glob(subject_id))
    for sub in subjects:
        responses = []
        runs = list(sub.glob("*.fif"))
        runs.sort()
        for irun, run in enumerate(runs):
            raw = read_raw_fif(run, verbose=False)
            responses.append(raw.get_data().T[0 : spg[irun].shape[0], :])

        trf_pho = TRF()
        trf_pho.train(
            pho,
            responses,
            raw.info["sfreq"],
            cfg["tmin"],
            cfg["tmax"],
            cfg["lambda"],
        )
        trf_pho.save(root / "results" / "trfs" / f"{sub.name}_spg.trf")
        del trf_pho

        trf_spg = TRF()
        trf_spg.train(
            spg,
            responses,
            raw.info["sfreq"],
            cfg["tmin"],
            cfg["tmax"],
            cfg["lambda"],
        )
        trf_spg.save(root / "results" / "trfs" / f"{sub.name}_spg.trf")
        del trf_spg

        trf_spg_pho = TRF()
        trf_spg_pho.train(
            spg_pho,
            responses,
            raw.info["sfreq"],
            cfg["tmin"],
            cfg["tmax"],
            cfg["lambda"],
        )
        trf_spg_pho.save(root / "results" / "trfs" / f"{sub.name}_spg_pho.trf")
        del trf_spg_pho

        trf_spg_pho_ons = TRF()
        trf_spg_pho_ons.train(
            spg_pho_ons,
            responses,
            raw.info["sfreq"],
            cfg["tmin"],
            cfg["tmax"],
            cfg["lambda"],
        )
        trf_spg_pho_ons.save(root / "results" / "trfs" / f"{sub.name}_spg_pho_ons.trf")
