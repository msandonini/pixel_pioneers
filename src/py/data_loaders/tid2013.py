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
    ref_img_path: str | Path | None = None


class TID2013Dataset(Dataset):
    def __init__(self, txt_index_path, images_dir, transform=None, distortion_levels=None, include_reference=False):
        """
        Args:
            txt_index_path (str): Percorso al file TID2013.txt (l'indice con i MOS)
            images_dir (str): La cartella radice dove hai estratto il dataset (es. "TID2013")
            transform (callable, optional): Trasformazioni PyTorch per le immagini
            distortion_levels (set[int] | list[int] | None, optional): Se specificato,
                include solo le immagini con il livello di distorsione indicato (1-5).
                Es. {1, 2} per distorsioni lievi, {4, 5} per distorsioni forti.
                Se None (default), include tutti i livelli.
            include_reference (bool, optional): Se True, carica anche l'immagine di
                reference corrispondente e __getitem__ restituisce 4 valori invece
                di 3 (image, reference, mos, img_name). Default False: comportamento
                identico alla versione no-reference originale.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.distortion_levels = set(distortion_levels) if distortion_levels is not None else None
        self.include_reference = include_reference
        self.data_samples: list[TID2013Sample] = []

        # Costruiamo una mappa case-insensitive delle reference una sola volta,
        # cosi' non serve rileggere la cartella per ogni riga del file indice.
        # Necessario perche' nel dataset originale tutte le reference sono in
        # maiuscolo (I01.BMP ... I24.BMP) tranne l'ultima, salvata come "i25.bmp"
        # minuscolo: la mappa rende il lookup robusto anche su filesystem
        # case-sensitive (Linux/Mac), non solo su Windows.
        ref_lookup: dict[str, Path] = {}
        if include_reference:
            ref_dir = Path(images_dir) / "reference_images"
            for candidate in ref_dir.iterdir():
                ref_lookup[candidate.stem.lower()] = candidate

        with open(txt_index_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                splitted = line.split(' ')

                img_name = splitted[1]
                value = float(splitted[0])

                level = int(Path(img_name).stem.split('_')[-1])

                if self.distortion_levels is not None and level not in self.distortion_levels:
                    continue

                ref_img_path = None
                if include_reference:
                    ref_id = img_name.split('_')[0].lower()  # es. "i01"
                    ref_img_path = ref_lookup.get(ref_id)
                    if ref_img_path is None:
                        raise FileNotFoundError(
                            f"Nessuna reference trovata per '{ref_id}' dentro {ref_dir}"
                        )

                self.data_samples.append(TID2013Sample(
                    img_name=img_name,
                    img_path=Path(images_dir) / "distorted_images" / img_name,
                    mos=value,
                    distortion_level=level,
                    ref_img_path=ref_img_path
                ))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        image = Image.open(sample.img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.include_reference:
            reference = Image.open(sample.ref_img_path).convert('RGB')
            if self.transform:
                reference = self.transform(reference)
            return image, reference, sample.mos, sample.img_name

        return image, sample.mos, sample.img_name