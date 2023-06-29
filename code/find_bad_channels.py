#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
from pyprep.ransac import find_bad_by_ransac
from mne_bids import BIDSPath, mark_channels

root = Path(__file__).parent.parent.absolute()
montage = make_standard_montage("biosemi128")
positions = np.stack([ch for ch in montage.get_positions()["ch_pos"].values()])
subfolders = list((root / "raw").glob("sub*"))
subfolders.sort()
for subfolder in subfolders:
    runs = list((subfolder / "eeg").glob("*_eeg.vhdr"))
    runs.sort()
    for irun, run in enumerate(runs):
        path = BIDSPath(subject=subfolder.name.split('-')[1], run=irun+1, root=subfolder.parent)
        raw = read_raw_brainvision(run, preload=True, verbose=False)
        raw.filter(1, 20)
        bads, _ = find_bad_by_ransac(
            raw.get_data(),
            raw.info["sfreq"],
            np.asarray(raw.info["ch_names"]),
            positions,
            exclude=[],
        )
        mark_channels(bids_path, ch_names=bads, status='bad')
