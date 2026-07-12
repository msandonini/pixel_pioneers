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
from embeddings.models.clip import extract_clip_embeddings
from embeddings.models.dinov2 import extract_dinov2_embeddings
from embeddings.models.siglip2 import extract_siglip2_embeddings
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
        embeddings = embeddings.pooler_output if hasattr(embeddings, "pooler_output") else embeddings
    else:
        embeddings = model(**inputs).pooler_output

    embeddings = F.normalize(embeddings, dim=1)

    return embeddings


def pil_collate(batch):
    refs, dists, mos = zip(*batch)
    mos = torch.tensor(mos, dtype=torch.float32)
    return list(refs), list(dists), mos


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

    extract_clip_embeddings(conf)
    extract_siglip2_embeddings(conf)
    extract_dinov2_embeddings(conf)

    train(conf)


if __name__ == '__main__':
    main()