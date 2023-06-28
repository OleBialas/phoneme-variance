#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt

root = Path(__file__).parent.parent.absolute()

for feature_matrix in ["1", "2"]:
    mapping = json.load(
        open(root / "code" / f"phonetic_features_{feature_matrix}.json")
    )

    features = list(mapping.keys())
    phonemes = np.unique([s for sublist in list(mapping.values()) for s in sublist])

    matrix = np.zeros((len(features), len(phonemes)))

    for i_p, p in enumerate(phonemes):
        for i_f, f in enumerate(features):
            if p in mapping[f]:
                matrix[i_f, i_p] = 1

    plt.figure()
    plt.imshow(matrix)
    plt.xticks(np.arange(len(phonemes)), phonemes)
    plt.yticks(np.arange(len(features)), features)
    plt.show()
