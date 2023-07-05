from pathlib import Path
import numpy as np
import pandas as pd
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
from pyprep.ransac import find_bad_by_ransac
from mne_bids import BIDSPath, mark_channels

root = Path(__file__).parent.parent.absolute()
montage = make_standard_montage("biosemi128")
positions = np.stack([ch for ch in montage.get_positions()["ch_pos"].values()])
subfolders = list((root / "raw").glob("sub*"))
subfolders.sort()
n_bads = []

for subfolder in subfolders:
    n_runs = len(list((subfolder / "eeg").glob("*_eeg.vhdr")))
    for irun in range(1, n_runs + 1):
        bids_path = BIDSPath(
            subject=subfolder.name.split("-")[1], run=irun, root=subfolder.parent
        )
        # remove old annotations by setting all channels to 'good'
        channels = pd.read_csv(
            subfolder
            / "eeg"
            / f"{subfolder.name}_task-listening_run-{str(irun).zfill(2)}_channels.tsv",
            sep="\t",
        )
        # channels.status = "good"
        channels.status_description = "n/a"
        channels.to_csv(
            subfolder
            / "eeg"
            / f"{subfolder.name}_task-listening_run-{str(irun).zfill(2)}_channels.tsv",
            columns=[
                "name",
                "type",
                "units",
                "description",
                "sampling_frequency",
                "status",
                "status_description",
            ],
            sep="\t",
            index=False,
        )
        raw = read_raw_brainvision(
            subfolder
            / "eeg"
            / f"{subfolder.name}_task-listening_run-{str(irun).zfill(2)}_eeg.vhdr",
            preload=True,
            verbose=False,
        )
        raw = raw.filter(1, 20, n_jobs=4, verbose=False)
        raw = raw.resample(64, n_jobs=4, verbose=False)
        bads, _ = find_bad_by_ransac(
            raw.get_data(),
            raw.info["sfreq"],
            np.asarray(raw.info["ch_names"]),
            positions,
            exclude=[],
        )
        n_bads.append(len(bads))
        if len(bads) > 0:
            mark_channels(bids_path, ch_names=bads, status="bad")
print(n_bads)
