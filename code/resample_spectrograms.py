from pathlib import Path
import numpy as np
from scipy.io import loadmat
import slab

root = Path(__file__).parent.parent.absolute()

spectrograms = list(
    (root / "results" / "spectrograms" / "single_speaker").glob("*_spg.mat")
)
for s in spectrograms:
    mat = loadmat(s)
    spg, fs = mat["spectrogram"], mat["Fs"][0][0]
    spg = slab.Sound(spg, samplerate=fs)
    spg = spg.filter(int(128 / 3), kind="lp")
    spg = spg.resample(128)
    spg.write(s.parent / f'{s.name.split(".")[0]}.wav')

spectrograms = list(
    (root / "results" / "spectrograms" / "multi_speakers").glob("*_spg.mat")
)
for s in spectrograms:
    mat = loadmat(s)
    spg, fs = mat["spectrogram"], mat["Fs"][0][0]
    spg = slab.Sound(spg, samplerate=fs)
    spg = spg.filter(int(128 / 3), kind="lp")
    spg = spg.resample(128)
    spg.write(s.parent / f'{s.name.split(".")[0]}.wav')
