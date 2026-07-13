import random
import timm
import torch
import datetime
import sys

from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

from frozen_layer_model.frozen_last_layer import get_frozen_siglip2

from pipeline.model import PipelineVisionEncoder, PipelineVisionEncoderConfiguration
from pipeline.metrics import compute_metrics
from pipeline.plots import scatter_plot
from pipeline.export import export_predictions
from pipeline import config
from pipeline import data_cache

from data_loaders.tid2013 import TID2013Dataset


class MockIQAModel(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        # One random score per image
        return torch.rand(batch_size)


def main():
    print("[pipeline] start")

    conf = config.parse_args()

    if "random_seed" in conf:
        random.seed(conf["random_seed"])

    print("[pipeline] downloading datasets...")

    # data = data_cache.download_datasets(conf=conf)
    data = {
        "tid2013": Path(conf["data_cache"]) / "tid2013" / "extracted"
    }

    print("[pipeline] downloading models...")

    models: dict[str, nn.Module] = {
        "mock": MockIQAModel(),
        "siglip2_standard": PipelineVisionEncoder(
            model_config=PipelineVisionEncoderConfiguration(
                model=timm.create_model("vit_base_patch16_siglip_224", pretrained=True),
                feature_extractor=lambda m, x: m.forward_features(x)  # Salta la testa
            )
        ),
        "siglip2_frozen": PipelineVisionEncoder(
            model_config=get_frozen_siglip2()  # Usa la configurazione con la testa bloccata
        )
    }

    print("[pipeline] loading TID2013 dataset...")

    dataset = TID2013Dataset(
        txt_index_path=data["tid2013"] / "mos_with_names.txt",
        images_dir=data["tid2013"],
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    print(f"[pipeline] n={len(dataset)} dataset elements successfully mapped")

    models_out: dict[str, dict[str, list]] = {}
    # "<name>":
    #   "img_names": []
    #   "distances": []
    #   "real_mos": []

    print("[pipeline] running model...")

    img_limit = 40

    for model_name, model in models.items():
        img_names = []
        distances = []
        real_mos = []
        count = 0

        # Mettiamo il modello in modalità valutazione sulla device corretta
        if hasattr(model, "to") and hasattr(model, "device"):
            model = model.to(model.device).eval()
        elif hasattr(model, "eval"):
            model.eval()

        for img, mos, name in dataloader:
            pred = model(img)

            # Se il modello restituisce feature multi-dimensionali (es. frozen),
            # le schiacciamo in uno score singolo per immagine
            if isinstance(pred, torch.Tensor) and pred.ndim > 1:
                pred_scores = pred.reshape(pred.shape[0], -1).mean(dim=-1)
            else:
                pred_scores = pred

            distances.extend(pred_scores.cpu().tolist())
            real_mos.extend(mos.tolist())
            img_names.extend(name)

            count += img.size(0)

            if img_limit > 0 and count >= img_limit:
                distances = distances[:img_limit]
                real_mos = real_mos[:img_limit]
                img_names = img_names[:img_limit]
                break

        models_out[model_name] = {
            "img_names": img_names,
            "distances": distances,
            "real_mos": real_mos
        }

        print(f"[pipeline] {model_name} completato con {len(distances)} predizioni.")

    print("[pipeline] computing correlation...")

    out_path = Path("out") / str(datetime.datetime.now())

    for name, output in models_out.items():
        csv_name = f"{name}.csv"
        plot_corr_name = f"{name}_corr.png"
        results = compute_metrics(output["distances"], output["real_mos"])

        print(f"[pipeline] {name} - PLCC: {results['PLCC']}, SROCC: {results['SROCC']}")

        export_predictions(
            filepath=f"{out_path}/{csv_name}",
            model_name=name,
            image_names=output["img_names"],
            predictions=output["distances"],
            mos_scores=output["real_mos"]
        )

    print(f"[pipeline] output saved to {out_path}")
    print(f"[pipeline] process finished")


if __name__ == "__main__":
    main()