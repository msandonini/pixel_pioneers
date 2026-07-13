from pathlib import Path

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, path: str | Path):
        path = Path(path)

        self.data = {}
        mos = None

        for child in path.iterdir():
            if not child.is_file():
                continue

            model_name = child.name.lower()
            model_data = torch.load(path / child.name)

            self.data[f"{model_name}_ref"] = model_data["ref"]
            self.data[f"{model_name}_dist"] = model_data["dist"]

            if mos is None:
                mos = model_data["mos"]
            else:
                assert torch.equal(mos, model_data["mos"]), f"mos mismatch between model {model_name} and previous one"

            self.data["mos"] = mos

    def __len__(self):
        return len(self.data)

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