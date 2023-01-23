from pathlib import Path
import numpy as np
from mne.io import read_raw_brainvision
from mne.preprocessing import ICA, read_ica, corrmap
from pyprep.ransac import find_bad_by_ransac
from mne.channels import make_standard_montage

root = Path(__file__).parent.parent.absolute()

reference = read_ica(root / "code" / "blink_reference-ica.fif")
montage = make_standard_montage("biosemi128")
positions = np.stack([ch for ch in montage.get_positions()["ch_pos"].values()])

for subfolder in (root / "raw").glob("sub*"):
    out_folder = root / "preprocessed" / subfolder.name
    if not out_folder.exists():
        out_folder.mkdir()
    for run in (subfolder / "eeg").glob("*_eeg.vhdr"):
        raw = read_raw_brainvision(run, preload=True)
        if int(subfolder.name.split("-")[-1]) >= 100:
            raw.drop_channels(
                [
                    "EXG1",
                    "EXG2",
                    "EXG3",
                    "EXG4",
                    "EXG5",
                    "EXG6",
                    "EXG7",
                    "EXG8",
                    "Status",
                ]
            )
        raw.set_montage(montage)
        raw.filter(1, 20)
        raw.resample(128)
        bads, _ = find_bad_by_ransac(
            raw.get_data(),
            raw.info["sfreq"],
            np.asarray(raw.info["ch_names"]),
            positions,
            exclude=[],
            corr_thresh=0.85,
        )
        raw.info["bads"] = bads
        raw.interpolate_bads()
        ica = ICA(n_components=0.99)
        ica.fit(raw)
        corrmap(
            [reference, ica],
            template=(0, component[0]),
            label="blinks",
            plot=False,
            threshold=0.8,
        )
        ica.apply(raw, exclude=ica.labels_["blinks"])
        raw.save(out_folder / (run.name.split("_eeg")[0] + "-raw.fif"), overwrite=True)
        ica.save(out_folder / (run.name.split("_eeg")[0] + "-ica.fif"), overwrite=True)
