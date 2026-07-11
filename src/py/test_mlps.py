from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor
)
import torch
import torch.nn.functional as F

from data_loaders.embedding_data import EmbeddingDataset
from data_loaders.mlp_tid2013 import TID2013Dataset
from fusion_mlp.mlp import FusionMLP
from metric_mlp.mlp import MetricMLP
from pipeline import config
from pipeline import data_cache

import random
from pathlib import Path
from datetime import datetime


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4


@torch.no_grad()
def get_model_embeddings(processor, model, images):
    inputs = processor(
        images=list(images),
        return_tensors="pt"
    ).to(DEVICE)

    if hasattr(model, "get_image_features"):
        embeddings = model.get_image_features(**inputs)
    else:
        embeddings = model(**inputs).pooler_output

    embeddings = F.normalize(embeddings, dim=1)

    return embeddings


def pil_collate(batch):
    refs, dists, mos = zip(*batch)
    mos = torch.tensor(mos, dtype=torch.float32)
    return list(refs), list(dists), mos


def extract_embeddings(conf):
    if not ("embeddings" in conf and "out" in conf["embeddings"]):
        print("[extract_embeddings] embeddings path not specified")
        return

    emb_out = conf["embeddings"]["out"]

    if Path(emb_out).exists():
        print("[extract_embeddings] embeddings already existing, no need for extraction")
        return

    data_cache.download_datasets(conf=conf)

    # Load encoders
    print("[extract_embeddings] load models")

    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE).eval()

    siglip_processor = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')
    siglip_model = AutoModel.from_pretrained('google/siglip2-base-patch16-224').to(DEVICE).eval()

    dino_processor = AutoProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(DEVICE).eval()

    # Freeze encoders

    print("[extract_embeddings] freeze models")

    for model in [clip_model, siglip_model, dino_model]:
        for p in model.parameters():
            p.requires_grad = False

    # Load dataset

    print("[extract_embeddings] load TID2013")

    dataset = TID2013Dataset(
        distorted_path="data/tid2013/extracted/distorted_images",
        reference_path="data/tid2013/extracted/reference_images",
        index_path="data/tid2013/extracted/mos_with_names.txt",
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=pil_collate
    )

    # Extraction loop

    embeddings = {
        "clip_ref": [],
        "clip_dist": [],

        "siglip_ref": [],
        "siglip_dist": [],

        "dino_ref": [],
        "dino_dist": [],

        "mos": []
    }

    print("[extract_embeddings] extract embeddings")

    for ref, dist, mos in loader:
        embeddings["clip_ref"].append(get_model_embeddings(clip_processor, clip_model, ref).cpu())
        embeddings["siglip_ref"].append(get_model_embeddings(siglip_processor, siglip_model, ref).cpu())
        embeddings["dino_ref"].append(get_model_embeddings(dino_processor, dino_model, ref).cpu())

        embeddings["clip_dist"].append(get_model_embeddings(clip_model, clip_processor, dist).cpu())
        embeddings["siglip_dist"].append(get_model_embeddings(siglip_model, siglip_processor, dist).cpu())
        embeddings["dino_dist"].append(get_model_embeddings(dino_model, dino_processor, dist).cpu())

        embeddings["mos"].append(mos)

    print("[extract_embeddings] save embeddings")

    for key in embeddings:
        embeddings[key] = torch.cat(
            embeddings[key],
            dim = 0
        )

    torch.save(
        embeddings,
        emb_out
    )


def train(conf):
    if not ("embeddings" in conf and "out" in conf["embeddings"]):
        print("[train] embeddings path not specified")
        return

    print("[train] load embeddings")

    dataset = EmbeddingDataset(conf["embeddings"]["out"])
    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    print("[train] init MLPs and optimizator")

    fusion = FusionMLP(
        input_dims = [768, 768, 768],
        fused_dim = 256,
        hidden_dim = 512,
        out_dim = 256
    ).to(DEVICE)

    metric = MetricMLP(
        embed_dim = 256,
        hidden_dim = 512
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(fusion.parameters()) + list(metric.parameters()),
        lr = LR
    )

    dt = f"datetime.now()".replace(":", ".")

    print("[train] train loop")

    for epoch in range(EPOCHS):
        fusion.train()
        metric.train()

        epoch_loss = 0

        for sample in loader:
            ref = fusion([
                sample["clip_ref"],
                sample["siglip_ref"],
                sample["dino_ref"]
            ])

            dist = fusion([
                sample["clip_dist"],
                sample["siglip_dist"],
                sample["dino_dist"]
            ])

            pred = metric(
                dist,
                ref
            ).squeze()

            loss = F.mse_loss(
                pred,
                sample["mos"]
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        loss_fmt = f"{epoch_loss/len(loader):.4f}"
        print(
            f"Epoch {epoch+1}"
            f"Loss = {loss_fmt}"
        )

        checkpoint = {
            "epoch": epoch,
            "fusion_state_dict": fusion.state_dict(),
            "metric_state_dict": metric.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }

        torch.save(checkpoint, f"checkpoints/{dt}/model_{epoch}_loss_{loss_fmt}.pt")





def main():
    conf = config.parse_args()

    if "random_seed" in conf:
        random.seed(conf["random_seed"])

    extract_embeddings(conf)
    train(conf)


if __name__ == '__main__':
    main()