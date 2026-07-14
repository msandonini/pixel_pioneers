import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr, kendalltau
from torch.utils.data import DataLoader, Dataset

from data_loaders.embedding_data import EmbeddingDataset
from fusion_mlp.mlp import FusionMLP
from metric_mlp.mlp import MetricMLP
from pipeline import config

MOS_RANGE = (0.0, 9.0)  # TID2013 scale
VAL_FRAC = 0.16
TEST_FRAC = 0.16
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3


class SingleEncoderView(Dataset):
    def __init__(self, dataset: EmbeddingDataset, model_name: str, indices: list[int]):
        self.ref = dataset.data[f"{model_name}_ref"]
        self.dist = dataset.data[f"{model_name}_dist"]
        self.mos = dataset.data["mos"]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        return self.dist[idx].float(), self.ref[idx].float(), self.mos[idx].float()


def split_by_ref(ref_paths, val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC, seed: int = 42):
    unique_refs = sorted(set(ref_paths), key=str)
    random.shuffle(unique_refs)

    n = len(unique_refs)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))

    test_refs = set(unique_refs[:n_test])
    val_refs = set(unique_refs[n_test:n_test + n_val])

    train_idx, val_idx, test_idx = [], [], []
    for i, ref in enumerate(ref_paths):
        if ref in test_refs:
            test_idx.append(i)
        elif ref in val_refs:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return train_idx, val_idx, test_idx


def iqa_loss(pred: torch.Tensor, mos: torch.Tensor, rank_weight: float = 0.3) -> torch.Tensor:
    # MSE + pairwise ranking

    mse = F.mse_loss(pred, mos)

    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_mos = mos.unsqueeze(1) - mos.unsqueeze(0)
    rank_loss = torch.relu(-torch.sign(diff_mos) * diff_pred).mean()

    return mse + rank_weight * rank_loss


@torch.no_grad
def evaluate(model: MetricMLP, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds, targets = [], []
    for pred_emb, target_emb, mos in loader:
        pred_emb, target_emb = pred_emb.to(device), target_emb.to(device)
        score = model(pred_emb, target_emb)
        preds.append(score.cpu())
        targets.append(mos)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    return {
        "srocc": spearmanr(preds, targets)[0],
        "plcc": pearsonr(preds, targets)[0],
        "krocc": kendalltau(preds, targets)[0],
    }


def train(
    model_name: str,
    dataset: EmbeddingDataset,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    device: torch.device,
    model_out: str | Path | None = None
) -> dict:
    train_loader = DataLoader(SingleEncoderView(dataset, model_name, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SingleEncoderView(dataset, model_name, val_idx), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SingleEncoderView(dataset, model_name, test_idx), batch_size=BATCH_SIZE, shuffle=True)

    embed_dim = dataset.data[f"{model_name}_ref"].shape[-1]
    model = MetricMLP(embed_dim=embed_dim, output_range=MOS_RANGE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_srocc = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for pred_emb, target_emb, mos in train_loader:
            pred_emb, target_emb, mos = pred_emb.to(device), target_emb.to(device), mos.to(device)

            optimizer.zero_grad()

            score = model(pred_emb, target_emb)

            loss = iqa_loss(score, mos)
            loss.backward()

            optimizer.step()

            total_loss += loss.item() * pred_emb.size(0)

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["srocc"] > best_val_srocc:
            best_val_srocc = val_metrics["srocc"]
            best_state = {k : v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[{model_name}] epoch {epoch + 1} - "
                f"train_loss = {total_loss / len(train_idx):.4f} - "
                f"val_srocc = {val_metrics['srocc']:.4f}"
            )

    model.load_state_dict(best_state)
    if model_out is not None:
        model_out = (Path(model_out) / "metric" / f"{datetime.now()}".replace(":", "."))
        model_out.mkdir(parents=True)
        torch.save(best_state, model_out / "model.pt")
    return evaluate(model, test_loader, device)


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

    train_idx, val_idx, test_idx = split_by_ref(dataset.ref_paths)

    results = {}
    for model_name in dataset.model_names:
        print(f"[train] training on {model_name} embeddings")
        results[model_name] = train(model_name, dataset, train_idx, val_idx, test_idx, device, model_out=conf["weights_out"] if "weights_out" in conf else None)

    print(f"[train] comparison")
    for name, m in results.items():
        print(f"{name}: SROCC = {m['srocc']:.4f}, PLCC = {m['plcc']:.4f}, KROCC = {m['krocc']:.4f}")


if __name__ == '__main__':
    main()