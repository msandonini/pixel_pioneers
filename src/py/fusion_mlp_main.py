import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, kendalltau
from torch.utils.data import DataLoader, random_split

from data_loaders.embedding_data import EmbeddingDataset
from fusion.mlp import FusionMLP
from metric_mlp.mlp import MetricMLP
from pipeline import config
from pathlib import Path

from utils.utils import iqa_loss

MOS_RANGE = (0.0, 9.0)  # TID2013 scale
VAL_FRAC = 0.16
TEST_FRAC = 0.16
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
FUSION_OUT_DIM = 128


from collections import defaultdict
import random

from torch.utils.data import Subset


def group_split_by_ref(
    dataset,
    val_frac=0.16,
    test_frac=0.16,
    seed=42,
):

    # Group sample indices by reference image
    ref_to_indices = defaultdict(list)
    for idx, ref in enumerate(dataset.ref_paths):
        ref_to_indices[ref].append(idx)

    # Shuffle unique references
    refs = list(ref_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(refs)

    n_refs = len(refs)

    n_val = int(val_frac * n_refs)
    n_test = int(test_frac * n_refs)
    n_train = n_refs - n_val - n_test

    train_refs = refs[:n_train]
    val_refs = refs[n_train:n_train + n_val]
    test_refs = refs[n_train + n_val:]

    def refs_to_indices(ref_list):
        indices = []
        for ref in ref_list:
            indices.extend(ref_to_indices[ref])
        return sorted(indices)

    train_indices = refs_to_indices(train_refs)
    val_indices = refs_to_indices(val_refs)
    test_indices = refs_to_indices(test_refs)

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def get_model_names_and_dims(dataset) -> tuple[list[str], list[int]]:
    model_names = sorted(
        {k.rsplit("_", 1)[0] for k in dataset.data.keys() if k.endswith("_ref")}
    )
    sample = dataset[0]
    input_dims = [sample[f"{m}_ref"].shape[-1] for m in model_names]
    return model_names, input_dims


@torch.inference_mode()
def evaluate(
    fusion_mlp: FusionMLP,
    metric_mlp: MetricMLP,
    loader: DataLoader,
    model_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    fusion_mlp.eval()
    metric_mlp.eval()

    preds = []
    targets = []
    losses = []

    for batch in loader:
        mos = batch["mos"].float().to(device)

        ref_emb = [
            batch[f"{m}_ref"].float().to(device) for m in model_names
        ]
        dist_emb = [
            batch[f"{m}_dist"].float().to(device) for m in model_names
        ]

        fused_ref = fusion_mlp(ref_emb)
        fused_dist = fusion_mlp(dist_emb)

        pred = metric_mlp(fused_ref, fused_dist)

        loss = F.mse_loss(pred, mos)
        losses.append(loss.item())

        preds.append(pred.cpu().numpy())
        targets.append(mos.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return {
        "loss": float(np.mean(losses)),
        "plcc": pearsonr(targets, preds).statistic,
        "srocc": spearmanr(targets, preds).statistic,
        "krocc": kendalltau(targets, preds).statistic,
    }


def train_one_epoch(
    fusion_mlp: FusionMLP,
    metric_mlp: MetricMLP,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    model_names: list[str],
    device: torch.device,
) -> float:
    fusion_mlp.train()
    metric_mlp.train()   # frozen

    running_loss = 0.0

    for batch in loader:
        mos = batch["mos"].float().to(device)

        ref_emb = [
            batch[f"{m}_ref"].float().to(device) for m in model_names
        ]
        dist_emb = [
            batch[f"{m}_dist"].float().to(device) for m in model_names
        ]

        optimizer.zero_grad()

        fused_ref = fusion_mlp(ref_emb)
        fused_dist = fusion_mlp(dist_emb)

        pred_mos = metric_mlp(fused_ref, fused_dist)

        loss = iqa_loss(pred_mos, mos)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def main():
    conf = config.parse_args()

    if "random_seed" in conf:
        random.seed(conf["random_seed"])

    if not ("embeddings" in conf and "out" in conf["embeddings"]):
        print("[train] embeddings path not specified")
        return

    print("[train] load embeddings")

    dataset = EmbeddingDataset(conf["embeddings"]["out"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_names, input_dims = get_model_names_and_dims(dataset)
    # n_total = len(dataset)
    # n_test = int(n_total * TEST_FRAC)
    # n_val = int(n_total * VAL_FRAC)
    # n_train = n_total - n_val - n_test

    print("[train] create dataloaders")


    # train_set, val_set, test_set = random_split(
    #     dataset,
    #     [n_train, n_val, n_test],
    #     generator=torch.Generator().manual_seed(conf["random_seed"]),
    # )

    train_set, val_set, test_set = group_split_by_ref(
        dataset,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=conf["random_seed"]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    print("[train] create MLPs")

    fusion_mlp = FusionMLP(
        input_dims=input_dims,
        fused_dim=256,
        hidden_dim=512,
        out_dim=FUSION_OUT_DIM,
        dropout=0.2
    ).to(device)

    metric_mlp = MetricMLP(
        embed_dim=FUSION_OUT_DIM,
        hidden_dim=512,
        dropout=0.2,
        normalize_embeddings=True,
        output_range=MOS_RANGE
    ).to(device)


    optimizer = torch.optim.AdamW(
        list(fusion_mlp.parameters()) +
        list(metric_mlp.parameters()),
        lr = LR,
        weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    if not "weights_out" in conf:
        print("[train] weights position not specified")
        return

    dt = f"{datetime.now()}".replace(":", ".")

    model_save_path = Path(conf["weights_out"]) / "fusion" / dt / "model.pt"
    model_save_path_latest = Path(conf["weights_out"]) / "fusion" / "model_latest.pt"

    best_srocc = -float("inf")
    patience_count = 0
    history = []
    best_state_fusion = {}
    best_state_metric = {}
    best_epoch = 0

    print("[train] train loop")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            fusion_mlp=fusion_mlp,
            metric_mlp=metric_mlp,
            optimizer=optimizer,
            device=device,
            model_names=model_names,
            loader=train_loader
        )

        val_metrics = evaluate(
            fusion_mlp=fusion_mlp,
            metric_mlp=metric_mlp,
            device=device,
            model_names=model_names,
            loader=val_loader
        )

        sched.step(val_metrics["srocc"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_plcc": val_metrics["plcc"],
            "val_srocc": val_metrics["srocc"],
            "val_krocc": val_metrics["krocc"],
        }
        history.append(row)

        print(
            f"[train] epoch {epoch} "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"PLCC={val_metrics['plcc']:.4f} | "
            f"SROCC={val_metrics['srocc']:.4f} | "
            f"KROCC={val_metrics['krocc']:.4f}"
        )

        if val_metrics["srocc"] > best_srocc:
            best_srocc = val_metrics["srocc"]
            best_state_fusion = fusion_mlp.state_dict()
            best_state_metric = metric_mlp.state_dict()
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= sched.patience * 2:
            print("[train] early stopping")
            break

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model_save_dict = {
        "fusion_state_dict": best_state_fusion,
        "metric_state_dict": best_state_metric,
        "model_names": model_names,
        "input_dims": input_dims,
        "fusion_out_dim": FUSION_OUT_DIM,
        "epoch": best_epoch,
        "best_srocc": best_srocc,
    }
    torch.save(model_save_dict, model_save_path)
    torch.save(model_save_dict, model_save_path_latest)

    fusion_mlp.load_state_dict(best_state_fusion)
    metric_mlp.load_state_dict(best_state_metric)

    test_metrics = evaluate(
        fusion_mlp=fusion_mlp,
        metric_mlp=metric_mlp,
        model_names=model_names,
        loader=test_loader,
        device=device
    )

    print(
        f"[train] test results: "
        f"loss={test_metrics['loss']:.5f} | "
        f"PLCC={test_metrics['plcc']:.4f} | "
        f"SROCC={test_metrics['srocc']:.4f} | "
        f"KROCC={test_metrics['krocc']:.4f}"
    )


if __name__ == '__main__':
    main()