import random

import torch

from data_loaders.embedding_data import EmbeddingDataset
from pipeline import config

from pathlib import Path


conf = config.parse_args()

if "random_seed" in conf:
    random.seed(conf["random_seed"])


if not ("embeddings" in conf and "out" in conf["embeddings"]):
    print("[train] embeddings path not specified")
else:
    path = Path(conf["embeddings"]["out"])

    for child in path.iterdir():
        if not child.is_file():
            continue

        model_name = child.name
        model_data = torch.load(path / child.name)

        print(model_name, "mos shape:", model_data["mos"].shape, "dtype:", model_data["mos"].dtype)
        print(model_name, "has NaN:", torch.isnan(model_data["mos"]).any().item())