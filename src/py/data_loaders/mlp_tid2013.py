import os.path

from PIL import Image
from torch.utils.data import Dataset

from pathlib import Path


def resolve_case_insensitive(directory, filename):
    exact = os.path.join(directory, filename)
    if os.path.exists(exact):
        return filename
    target = filename.lower()

    for f in os.listdir(directory):
        if f.lower() == target:
            return f

    raise FileNotFoundError(f"No file matching {filename} in {directory}")


class TID2013Dataset(Dataset):
    def __init__(self,
        distorted_path: str | Path,
        reference_path: str | Path,
        index_path: str | Path,
        transform = None
    ):
        self.distorted = Path(distorted_path)
        self.reference = Path(reference_path)
        self.index = Path(index_path)

        self.transform = transform

        self.samples = []

        self.min_mos = None
        self.max_mos = None

        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                mos, filename = line.strip().split(' ')

                ref = resolve_case_insensitive(reference_path, filename[:3].upper() + ".BMP")

                mos = float(mos)

                self.samples.append(
                    (
                        self.reference / ref,
                        self.distorted / filename,
                        mos
                    )
                )

                if self.min_mos is None or self.min_mos > mos:
                    self.min_mos = mos

                if self.max_mos is None or self.max_mos < mos:
                    self.max_mos = mos

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ref_path, dist_path, mos = self.samples[idx]

        ref = Image.open(ref_path).convert('RGB')
        dist = Image.open(dist_path).convert('RGB')

        if self.transform is not None:
            ref = self.transform(ref)
            dist = self.transform(dist)

        return ref, dist, (mos - self.min_mos) / (self.max_mos - self.min_mos)