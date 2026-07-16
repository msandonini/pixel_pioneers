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


# Scenari di distorsione da confrontare: nome leggibile -> set di livelli (None = tutti)
DISTORTION_SCENARIOS: dict[str, set[int] | None] = {
    "all_levels": None,
    "low_distortion": {1, 2},
    "high_distortion": {4, 5},
}


def build_models() -> dict[str, nn.Module]:
    return {
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


def run_models(models: dict[str, nn.Module], dataloader: DataLoader, img_limit: int) -> dict[str, dict[str, list]]:
    models_out: dict[str, dict[str, list]] = {}
    # "<name>":
    #   "img_names": []
    #   "distances": []
    #   "real_mos": []

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

    return models_out


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

    img_limit = 40
    out_path = Path("out") / str(datetime.datetime.now()).replace(":", ".")

    # Qui accumuliamo i risultati di ogni scenario per il confronto finale
    # "<scenario_name>": {"<model_name>": {"PLCC":..., "SROCC":..., "KROCC":...}}
    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for scenario_name, distortion_levels in DISTORTION_SCENARIOS.items():
        print(f"\n[pipeline] ===== Scenario: {scenario_name} (distortion_levels={distortion_levels}) =====")

        print("[pipeline] downloading models...")
        models = build_models()

        print("[pipeline] loading TID2013 dataset...")

        dataset = TID2013Dataset(
            txt_index_path=data["tid2013"] / "mos_with_names.txt",
            images_dir=data["tid2013"],
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            distortion_levels=distortion_levels
        )

        # Verifica che il filtro sui livelli di distorsione funzioni davvero,
        # controllando quali livelli sono effettivamente presenti nei campioni caricati
        levels_found = sorted({s.distortion_level for s in dataset.data_samples})
        print(f"[pipeline] filtro distortion_levels richiesto: {distortion_levels}")
        print(f"[pipeline] livelli di distorsione effettivamente presenti nel dataset: {levels_found}")
        print(f"[pipeline] n={len(dataset)} dataset elements successfully mapped")

        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4
        )

        print("[pipeline] running model...")
        models_out = run_models(models, dataloader, img_limit)

        print("[pipeline] computing correlation...")

        scenario_results: dict[str, dict[str, float]] = {}

        for name, output in models_out.items():
            csv_name = f"{name}.csv"
            results = compute_metrics(output["distances"], output["real_mos"])
            scenario_results[name] = results

            print(f"[pipeline] {name} - SROCC: {results['SROCC']:.4f}, PLCC: {results['PLCC']:.4f}, KROCC: {results['KROCC']:.4f}")

            export_predictions(
                filepath=f"{out_path}/{scenario_name}/{csv_name}",
                model_name=name,
                image_names=output["img_names"],
                predictions=output["distances"],
                mos_scores=output["real_mos"]
            )

        all_results[scenario_name] = scenario_results

    print(f"\n[pipeline] output saved to {out_path}")

    # Tabella di confronto finale tra tutti gli scenari e tutti i modelli
    print("\n[pipeline] ===== Confronto finale tra scenari di distorsione =====")
    for scenario_name, scenario_results in all_results.items():
        print(f"\n--- {scenario_name} ---")
        for model_name, results in scenario_results.items():
            print(f"  {model_name:20s} SROCC: {results['SROCC']:.4f}  PLCC: {results['PLCC']:.4f}  KROCC: {results['KROCC']:.4f}")

    print("\n[pipeline] process finished")


if __name__ == "__main__":
    main()