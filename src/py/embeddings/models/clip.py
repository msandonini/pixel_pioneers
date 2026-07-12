import random

import torch.cuda
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel

from data_loaders.mlp_tid2013 import TID2013Dataset
from embeddings.extraction import get_model_embeddings, pil_collate
from pipeline import config, data_cache


def extract_clip_embeddings(conf):
    if not ("embeddings" in conf and "out" in conf["embeddings"]):
        print("[extract_embeddings] embeddings path not specified")
        return

    emb_out = conf["embeddings"]["out"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[CLIP] download datasets")

    data_cache.download_datasets(conf=conf)

    print("[CLIP] download model")

    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()

    print("[CLIP] freeze encoder")
    for p in clip_model.parameters():
        p.requires_grad = False

    print("[CLIP] load TID2013")

    dataset = TID2013Dataset(
        distorted_path="data/tid2013/extracted/distorted_images",
        reference_path="data/tid2013/extracted/reference_images",
        index_path="data/tid2013/extracted/mos_with_names.txt",
    )

    if "models" in conf and "clip" in conf["models"] and "batch_size" in conf["models"]["clip"]:
        batch_size = conf["models"]["clip"]["batch_size"]
    else:
        batch_size = 16

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=pil_collate
    )

    print("[CLIP] extract embeddings")

    embeddings = {
        "ref": [],
        "dist": [],
        "mos": []
    }

    n = 0
    for ref, dist, mos in loader:
        print(f"[CLIP] loop n={n}")

        print(f" -> ref")
        embeddings["ref"].append(get_model_embeddings(clip_processor, clip_model, ref, device).cpu())

        print(f" -> dist")
        embeddings["dist"].append(get_model_embeddings(clip_processor, clip_model, dist, device).cpu())

        embeddings["mos"].append(mos)

        n += 1

    print("[CLIP] save embeddings")
    for key in embeddings:
        embeddings[key] = torch.cat(
            embeddings[key],
            dim = 0
        )

    torch.save(embeddings, emb_out)
    print("[CLIP] extraction done")


def main():
    conf = config.parse_args()

    if "random_seed" in conf:
        random.seed(conf["random_seed"])

    extract_clip_embeddings(conf)


if __name__ == '__main__':
    main()
