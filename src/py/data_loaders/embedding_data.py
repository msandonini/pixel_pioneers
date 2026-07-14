from pathlib import Path

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, path: str | Path):
        path = Path(path)

        self.data = {}
        mos = None

        files = sorted(child for child in path.iterdir() if child.is_file())

        if not files:
            raise FileNotFoundError(f"No files found in path {path}")

        raw = {}
        for f in files:
            model_name = f.stem.lower()
            raw[model_name] = torch.load(f, weights_only=False)

        self.model_names = list(raw.keys())

        # Use first model as canonical sample order
        canon_name = next(iter(raw))
        canon_keys = list(zip(raw[canon_name]["ref_path"], raw[canon_name]["dist_path"]))

        self.ref_paths = list(raw[canon_name]["ref_path"])
        self.dist_paths = list(raw[canon_name]["dist_path"])

        self.data = {
            "mos": raw[canon_name]["mos"],
        }

        for model_name, model_data in raw.items():
            if model_name == canon_name:
                self.data[f"{model_name}_ref"] = model_data["ref"]
                self.data[f"{model_name}_dist"] = model_data["dist"]
                continue

            keys = list(zip(model_data["ref_path"], model_data["dist_path"]))
            key_to_idx = {key: i for i, key in enumerate(keys)}

            missing = [k for k in canon_keys if k not in key_to_idx]
            if missing:
                raise ValueError(f"{model_name} is missing {len(missing)} samples present in {canon_name}")

            order = torch.tensor([key_to_idx[k] for k in canon_keys], dtype=torch.long)

            self.data[f"{model_name}_ref"] = model_data["ref"][order]
            self.data[f"{model_name}_dist"] = model_data["dist"][order]

            aligned_mos = model_data["mos"][order]
            assert torch.allclose(self.data["mos"], aligned_mos)


    def __len__(self):
        return len(self.data["mos"])

    def __getitem__(self, idx):
        return {key: values[idx] for key, values in self.data.items()}