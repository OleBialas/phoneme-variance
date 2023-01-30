from pathlib import Path
import json
import numpy as np
import textgrid

root = Path(__file__).parent.parent.absolute()


def get_stick_functions(text_grids, samplerate):
    phoneme_codes = np.asarray(
        list(json.load(open(root / "code" / "phoneme_codes.json")).keys())
    )
    stick_functions = []
    for tg in text_grids:
        phoneme_grid = textgrid.TextGrid.fromFile(tg)[0]
        nsamples = round(phoneme_grid[-1].maxTime * samplerate)
        stick_function = np.zeros((nsamples, len(phoneme_codes)))
        for p in phoneme_grid:
            if p.mark[:2] in phoneme_codes:
                idx = np.where(phoneme_codes == p.mark[:2])[0][0]
                start = round(p.minTime * samplerate)
                stop = round(p.maxTime * samplerate)
                stick_function[start:stop, idx] = 1
        stick_functions.append(stick_function)
    return stick_functions
