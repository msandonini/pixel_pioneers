import torch
import torch.nn.functional as F

from pathlib import Path


@torch.no_grad()
def get_model_embeddings(
    processor,
    model,
    images,
    device
):
    inputs = processor(
        images=list(images),
        return_tensors="pt"
    ).to(device)

    if hasattr(model, "get_image_features"):
        embeddings = model.get_image_features(**inputs)
        embeddings = embeddings.pooler_output if hasattr(embeddings, "pooler_output") else embeddings
    else:
        embeddings = model(**inputs).pooler_output

    embeddings = F.normalize(embeddings, dim=1)

    return embeddings


def extract_embeddings(emb_out: str | Path):
    if Path(emb_out).exists():
        print("[extract_embeddings] embeddings already existing, no need for extraction")
        return


def pil_collate(batch):
    refs, dists, mos, ref_paths, dist_paths = zip(*batch)
    mos = torch.tensor(mos, dtype=torch.float32)
    return list(refs), list(dists), mos, list(ref_paths), list(dist_paths)