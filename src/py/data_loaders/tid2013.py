import os
from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TID2013Sample():
    img_name: str
    img_path: str | Path
    mos: float
    distortion_level: int

class TID2013Dataset(Dataset):
    def __init__(self, txt_index_path, images_dir, transform=None, distortion_levels=None):
        """
        Args:
            txt_index_path (str): Percorso al file TID2013.txt (l'indice con i MOS)
            images_dir (str): La cartella radice dove hai estratto il dataset (es. "TID2013")
            transform (callable, optional): Trasformazioni PyTorch per le immagini
        """
        self.images_dir = images_dir
        self.transform = transform
        self.distortion_levels = set(distortion_levels) if distortion_levels is not None else None
        self.data_samples: list[TID2013Sample] = []

        with open(txt_index_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                splitted = line.split(' ')

                img_name = splitted[1]
                value = float(splitted[0])

                level = int(Path(img_name).stem.split('_')[-1])

                if self.distortion_levels is not None and level not in self.distortion_levels:
                    continue
                
                self.data_samples.append(TID2013Sample(
                    img_name=img_name,
                    img_path=Path(images_dir) / "distorted_images" / img_name,
                    mos=value,
                    distortion_level=level
                ))


    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        image = Image.open(sample.img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, sample.mos, sample.img_name
