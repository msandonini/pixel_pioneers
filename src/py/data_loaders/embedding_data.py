from pathlib import Path

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, path: str | Path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {
            "clip_ref": self.data["clip_ref"][idx],
            "clip_dist": self.data["clip_dist"][idx],

            "siglip_ref": self.data["siglip_ref"][idx],
            "siglip_dist": self.data["siglip_dist"][idx],

            "dino_ref": self.data["dino_ref"][idx],
            "dino_dist": self.data["dino_dist"][idx],

            "mos": self.data["mos"][idx]
        }