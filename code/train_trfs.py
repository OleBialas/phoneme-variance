from pathlib import Path
import json
import numpy as np
import pandas as pd
from mne.io import read_raw_fif
import textgrid
from mtrf.model import TRF, cross_validate

root = Path(__file__).parent.parent.absolute()
cfg = json.load(open(root / "code" / "trf_parameters.json"))
phoneme_codes = pd.read_csv(root / "code" / "phonemes.csv")["ARPAbet"].to_list()


for data_set, subject_id in zip(
    ["single_speaker", "multi_speaker"], ["sub-0*", "sub-1*"]
):

    text_grids = list((root / "raw" / "stimuli" / data_set).glob("*TextGrid"))
    spectrograms = list((root / "raw" / "stimuli" / data_set).glob("*spg.npy"))
    text_grids.sort(), spectrograms.sort()

    # get spectograms and phoneme stick functions
    pho, spg = [], []
    for t, s in zip(text_grids, spectrograms):
        spg.append(np.load(s, allow_pickle=True).item()["signal"])
        fs = np.load(s, allow_pickle=True).item()["fs"]
        phoneme_grid = textgrid.TextGrid.fromFile(t)[0]
        pho.append(np.zeros((spg[-1].shape[0], len(phoneme_codes))))
        for p in phoneme_grid:
            if p.mark[:2] in phoneme_codes:
                idx = np.where(np.asarray(phoneme_codes) == p.mark[:2])[0][0]
                start = round(p.minTime * fs)
                stop = round(p.maxTime * fs)
                pho[-1][start:stop, idx] = 1

    # compute spg and spg+pho trf for each subject
    subjects = list((root / "preprocessed").glob(subject_id))
    for isub, sub in enumerate(subjects):
        responses = []
        runs = list(sub.glob("*.fif"))
        runs.sort()
        for irun, run in enumerate(runs):
            eeg = read_raw_fif(run, verbose=False)
            eeg = eeg.get_data().T

            print(eeg.shape[0] - spg[irun].shape[0])

            responses.append(raw.get_data().T[0 : spg[irun].shape[0], :])

        trf = TRF()
        trf.train(spg, responses, raw.info["sfreq"], tmin, tmax, regularization)
        trf.save(root / "results" / f"{sub.name}_phoneme.trf")
