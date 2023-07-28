from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
from mne.viz import plot_topomap

root = Path(__file__).parent.parent.parent.absolute()
raw = read_raw_brainvision(
    root / "raw" / "sub-001" / "eeg/sub-001_task-listening_run-01_eeg.vhdr"
)
montage = make_standard_montage("biosemi128")
raw = raw.set_montage(montage)
info = raw.info
del raw

fig, ax = plt.subplots(1, 2)

subject_ids = [list((root / "raw").glob("sub-0*")), list((root / "raw").glob("sub-1*"))]
names = ["Old man and the sea", "Multiple stories"]

for i, (subjects, name) in enumerate(zip(subject_ids, names)):
    bad_counts = np.zeros(128)
    for isub, subject in enumerate(subjects):
        channels = list((subject / "eeg").glob("*_channels.tsv"))
        for chs in channels:
            bads = (pd.read_csv(chs, sep="\t").status == "bad").tolist()
            bad_counts += bads
    bad_counts /= len(subjects) * len(channels)
    im, _ = plot_topomap(bad_counts, info, axes=ax[i], show=False)
    ax[i].set_title(name)

fig.colorbar(im, ax=list(ax), fraction=0.1)
fig.suptitle("% segments for which channels are bad")
plt.show()
