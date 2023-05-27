from pathlib import Path
import slab
import numpy as np
from scipy.io import loadmat

root = Path(__file__).parent.parent.absolute()

envelopes = list((root / "raw" / "stimuli" / "single_speaker").glob("*_env.mat"))
spectrograms = list((root / "raw" / "stimuli" / "single_speaker").glob("*_spg.mat"))

for e, s in zip(envelopes, spectrograms):
    mat = loadmat(e)
    envelope = slab.Sound(mat["envelope"][0], samplerate=mat["Fs"][0][0])
    envelope = envelope.resample(128)
    envelope.data[envelope.data < 0] = 0
    mat = loadmat(s)
    spectrogram = slab.Sound(mat["spectrogram"], samplerate=mat["Fs"][0][0])
    spectrogram.resample(128)
