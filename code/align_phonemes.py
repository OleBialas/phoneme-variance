from pathlib import Path
import json
import numpy as np
import slab
import skfda
from skfda.preprocessing.registration import FisherRaoElasticRegistration
import textgrid

root = Path(__file__).parent.parent.absolute()

phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phoneme_codes.json")).keys())
)

# for single speaker data align the phonemes across all files
textgrids = list((root / "raw" / "stimuli" / "single_speaker").glob("*.TextGrid"))
spectrograms = list(
    (root / "results" / "spectrograms" / "single_speaker").glob("*_spg.wav")
)
textgrids.sort(), spectrograms.sort()

# get the envelope for each phoneme utterance
for pc in phoneme_codes:
    phoneme_spectrograms = []
    for t, s in zip(textgrids, spectrograms):
        spectrogram = slab.Sound(s)
        phoneme_grid = textgrid.TextGrid.fromFile(t)[0]
        for p in phoneme_grid:
            if p.mark[:2] == pc:
                idx = np.where(np.asarray(phoneme_codes) == p.mark[:2])[0][0]
                start = round(p.minTime * spectrogram.samplerate)
                stop = round(p.maxTime * spectrogram.samplerate)
                phoneme_spectrograms.append(spectrogram.data[start:stop, :])

    # reject outliers
    lengths = np.asarray([len(s) for s in phoneme_spectrograms])
    mask = (
        (lengths.mean() + 2 * lengths.std())
        > lengths
        > (lengths.mean() - 2 * lengths.std())
    )

# pad to same length so that each phoneme starts and ends with 0

# compute warping function

# apply warping to spectrogram

# save aligned spectrograms and warping

# for mutli speaker data align the phonemes for each file
