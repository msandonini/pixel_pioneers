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

    clip_emb = torch.load(path / "clip.pt")
    siglip_emb = torch.load(path / "siglip.pt")
    dino_emb = torch.load(path / "dino.pt")

    print(f"CLIP - mos shape: {clip_emb['mos'].shape}, dtype: {clip_emb['mos'].dtype}")
    print(f"SigLIP2 - mos shape: {siglip_emb['mos'].shape}, dtype: {siglip_emb['mos'].dtype}")
    print(f"DINOv2 - mos shape: {dino_emb['mos'].shape}, dtype: {dino_emb['mos'].dtype}")

    print(f"CLIP & SigLIP2 close: {torch.allclose(clip_emb['mos'], siglip_emb['mos'])}")
    print(f"CLIP & DINOv2 close: {torch.allclose(clip_emb['mos'], dino_emb['mos'])}")
    print(f"SigLIP2 & DINOv2 close: {torch.allclose(siglip_emb['mos'], dino_emb['mos'])}")

    diff = (clip_emb['mos'] - siglip_emb['mos']).abs()
    print(f"CLIP - SigLIP2 max diff: {diff.max().item()} at index: {diff.argmax().item()}")
    diff = (clip_emb['mos'] - dino_emb['mos']).abs()
    print(f"CLIP - DINOv2 max diff: {diff.max().item()} at index: {diff.argmax().item()}")
    diff = (siglip_emb['mos'] - dino_emb['mos']).abs()
    print(f"SigLIP2 - DINOv2 max diff: {diff.max().item()} at index: {diff.argmax().item()}")

    clip_sorted, _ = torch.sort(clip_emb['mos'])
    siglip_sorted, _ = torch.sort(siglip_emb['mos'])
    dino_sorted, _ = torch.sort(dino_emb['mos'])

    print("CLIP & SigLIP2 sorted close:", torch.allclose(clip_sorted, siglip_sorted))
    print("CLIP & DINOv2 sorted close:", torch.allclose(clip_sorted, dino_sorted))
