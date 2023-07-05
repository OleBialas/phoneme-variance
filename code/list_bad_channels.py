from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).parent.parent.absolute()

subjects = list((root / "raw").glob("sub*"))
subjects.sort()

n_bads = []
for subject in subjects:
    runs = list((subject / "eeg").glob("*.tsv"))
    runs.sort()
    for run in runs:
        channels = pd.read_csv(run, sep="\t")
        n_bads.append(sum(channels.status == "bad"))
