from pathlib import Path
import json json
import numpy as np
import skfda
from skfda.preprocessing.registration import FisherRaoElasticRegistration
import textgrid

root = Path(__file__).parent.parent.absolute()

phoneme_codes = np.asarray(
    list(json.load(open(root / "code" / "phoneme_codes.json")).keys())
)

# for single speaker data align the phonemes across all files
textgrids = list((root/'raw'/'stimuli'/'single_speaker').glob('*.TextGrid'))

pg


# get the envelope for each phoneme utterance

# reject outliers

# pad to same length so that each phoneme starts and ends with 0

# compute warping function

# apply warping to spectrogram

# save aligned spectrograms and warping

# for mutli speaker data align the phonemes for each file


