from pathlib import Path
import json
from itertools import compress
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
    mask = np.abs(lengths - lengths.mean()) < 2 * lengths.std()
    phoneme_spectrograms = list(compress(phoneme_spectrograms, mask))
    print(f"removed {len(lengths)-len(phoneme_spectrograms)} outliers")

    # pad to same length so that each phoneme starts and ends with 0
    max_len = max([len(s) for s in phoneme_spectrograms])
    for ipho, pho_spg in enumerate(phoneme_spectrograms):
        diff = max_len - len(pho_spg)
        phoneme_spectrograms[ipho] = np.concatenate(
            [
                np.zeros((1, pho_spg.shape[-1])),
                pho_spg,
                np.zeros((diff + 1, pho_spg.shape[-1])),
            ]
        )
    phoneme_spectrograms = np.stack(phoneme_spectrograms)
    # set all negative values to 0
    phoneme_spectrograms = phoneme_spectrograms.clip(min=0)

    # compute warping function on envelopes
    phoneme_envelopes = phoneme_spectrograms.mean(axis=-1)
    phoneme_envelopes = skfda.FDataGrid(phoneme_envelopes)
    elastic_registration = FisherRaoElasticRegistration()
    elastic_registration.fit_transform(phoneme_envelopes)

    # apply warping to spectrogram bands by band
    phoneme_spectrograms_aligned = np.zeros(phoneme_spectrograms.shape)
    for iband in range(phoneme_spectrograms.shape[-1]):
        band = skfda.FDataGrid(phoneme_spectrograms[:, :, iband])
        band = elastic_registration.transform(band)
        phoneme_spectrograms_aligned[:, :, iband] = band.data_matrix.squeeze()

    # save aligned spectrograms and warping
    outdir = root / "results" / "aligned" / "single_speaker"
    np.save(outdir / f"{pc}_spectrograms.npy", phoneme_spectrograms_aligned)
    np.save(
        outdir / f"{pc}_warping.npy",
        elastic_registration.warping_.data_matrix.squeeze(),
    )

# for mutli speaker data align the phonemes for each file
textgrids = list((root / "raw" / "stimuli" / "multi_speakers").glob("*.TextGrid"))
spectrograms = list(
    (root / "results" / "spectrograms" / "multi_speakers").glob("*_spg.wav")
)
textgrids.sort(), spectrograms.sort()

for t, s in zip(textgrids, spectrograms):
    spectrogram = slab.Sound(s)
    phoneme_grid = textgrid.TextGrid.fromFile(t)[0]
    for pc in phoneme_codes:  # get the envelope for each phoneme utterance
        phoneme_spectrograms = []
        for p in phoneme_grid:
            if p.mark[:2] == pc:
                idx = np.where(np.asarray(phoneme_codes) == p.mark[:2])[0][0]
                start = round(p.minTime * spectrogram.samplerate)
                stop = round(p.maxTime * spectrogram.samplerate)
                phoneme_spectrograms.append(spectrogram.data[start:stop, :])
        if len(phoneme_spectrograms) > 2:
            # reject outliers
            lengths = np.asarray([len(s) for s in phoneme_spectrograms])
            mask = np.abs(lengths - lengths.mean()) < 2 * lengths.std()
            phoneme_spectrograms = list(compress(phoneme_spectrograms, mask))
            print(f"removed {len(lengths)-len(phoneme_spectrograms)} outliers")

            # pad to same length so that each phoneme starts and ends with 0
            max_len = max([len(s) for s in phoneme_spectrograms])
            for ipho, pho_spg in enumerate(phoneme_spectrograms):
                diff = max_len - len(pho_spg)
                phoneme_spectrograms[ipho] = np.concatenate(
                    [
                        np.zeros((1, pho_spg.shape[-1])),
                        pho_spg,
                        np.zeros((diff + 1, pho_spg.shape[-1])),
                    ]
                )
            phoneme_spectrograms = np.stack(phoneme_spectrograms)
            # set all negative values to 0
            phoneme_spectrograms = phoneme_spectrograms.clip(min=0)

            # compute warping function on envelopes
            phoneme_envelopes = phoneme_spectrograms.mean(axis=-1)
            phoneme_envelopes = skfda.FDataGrid(phoneme_envelopes)
            elastic_registration = FisherRaoElasticRegistration()
            elastic_registration.fit_transform(phoneme_envelopes)

            # apply warping to spectrogram bands by band
            phoneme_spectrograms_aligned = np.zeros(phoneme_spectrograms.shape)
            for iband in range(phoneme_spectrograms.shape[-1]):
                band = skfda.FDataGrid(phoneme_spectrograms[:, :, iband])
                band = elastic_registration.transform(band)
                phoneme_spectrograms_aligned[:, :, iband] = band.data_matrix.squeeze()

            # save aligned spectrograms and warping
            outdir = (
                root / "results" / "aligned" / "multi_speakers" / s.name.split("_")[0]
            )
            if not outdir.exists():
                outdir.mkdir()
            np.save(outdir / f"{pc}_spectrograms.npy", phoneme_spectrograms_aligned)
            np.save(
                outdir / f"{pc}_warping.npy",
                elastic_registration.warping_.data_matrix.squeeze(),
            )
