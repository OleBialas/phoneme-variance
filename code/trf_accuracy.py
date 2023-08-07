from pathlib import Path
import json
import numpy as np

root = Path(__file__).parent.parent.absolute()

channels = json.load(open(root / "code" / "trf_parameters.json"))["channels"]

corr_files = list((root / "results" / "correlations").glob("*"))

correlation = []
for f in corr_files:
    correlation.append(np.load(f)[channels].mean())

correlation = np.asarray(correlation)
print(f"Mean r: {correlation.mean()}, SD: {correlation.std()}")
