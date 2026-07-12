import os
import random
from pathlib import Path

import torch.cuda
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel

from data_loaders.mlp_tid2013 import TID2013Dataset
from embeddings.extraction import get_model_embeddings, pil_collate
from pipeline import config, data_cache


def extract_dinov2_embeddings(conf):
    if not ("embeddings" in conf and "out" in conf["embeddings"]):
        print("[extract_embeddings] embeddings path not specified")
        return

    emb_out = Path(conf["embeddings"]["out"])
    if os.path.exists(emb_out / "DINOv2"):
        print("[extract_embeddings] CLIP embeddings already exist - skip")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[DINOv2] download datasets")

    data_cache.download_datasets(conf=conf)

    print("[DINOv2] download model")

    dino_processor = AutoProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()

    print("[DINOv2] freeze encoder")
    for p in dino_model.parameters():
        p.requires_grad = False

    print("[DINOv2] load TID2013")

    dataset = TID2013Dataset(
        distorted_path="data/tid2013/extracted/distorted_images",
        reference_path="data/tid2013/extracted/reference_images",
        index_path="data/tid2013/extracted/mos_with_names.txt",
    )

    if "models" in conf and "dinov2" in conf["models"] and "batch_size" in conf["models"]["dinov2"]:
        batch_size = conf["models"]["dinov2"]["batch_size"]
    else:
        batch_size = 16

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pil_collate
    )

    print("[DINOv2] extract embeddings")

    embeddings = {
        "ref": [],
        "dist": [],
        "mos": []
    }

    n = 0
    for ref, dist, mos in loader:
        print(f"[DINOv2] loop n={n}")

        print(f" -> ref")
        embeddings["ref"].append(get_model_embeddings(dino_processor, dino_model, ref, device).cpu())

        print(f" -> dist")
        embeddings["dist"].append(get_model_embeddings(dino_processor, dino_model, dist, device).cpu())

        embeddings["mos"].append(mos)

        n += 1

    print("[DINOv2] save embeddings")
    for key in embeddings:
        embeddings[key] = torch.cat(
            embeddings[key],
            dim = 0
        )



    emb_out.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, emb_out / "DINOv2")
    print("[DINOv2] extraction done")


def main():
    conf = config.parse_args()

    if "random_seed" in conf:
        random.seed(conf["random_seed"])

    extract_dinov2_embeddings(conf)


if __name__ == '__main__':
    main()
