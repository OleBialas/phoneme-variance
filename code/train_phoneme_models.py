"""
For each subject, find the best lambda, then train one phoneme TRF using all the data
"""
from pathlib import Path
import numpy as np
from mne.io import read_raw_fif
from mtrf.model import TRF, cross_validate
from phonemes import get_stick_functions

root = Path(__file__).parent.parent.absolute()

regularization = np.logspace(-2, 6, 8)
splits = 10
tmin, tmax = 0, 0.35

subjects = list((root / "preprocessed").glob("sub-0*"))
subjects.sort()
audios = list((root / "raw" / "stimuli" / "single_speaker").glob("*TextGrid"))
audios.sort()

for isub, sub in enumerate(subjects):
    responses = []
    runs = list(sub.glob("*.fif"))
    runs.sort()
    for irun, run in enumerate(runs):
        raw = read_raw_fif(run)
        if irun == 0 and isub == 0:
            stimuli = get_stick_functions(audios, raw.info["sfreq"])
        responses.append(raw.get_data().T[0 : stimuli[irun].shape[0], :])
    trf = TRF()
    trf.fit(stimuli, responses, raw.info["sfreq"], tmin, tmax, regularization, k=splits)
    trf.save(root / "results" / f"{sub.name}_phoneme.trf")
