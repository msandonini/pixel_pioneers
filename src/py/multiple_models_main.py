import random
import timm
import torch
import torch.nn.functional as F
import datetime

from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

from pipeline.model import PipelineVisionEncoder, PipelineVisionEncoderConfiguration
from pipeline.metrics import compute_metrics
from pipeline.export import export_predictions
from pipeline import config

from data_loaders.tid2013 import TID2013Dataset  # stesso file usato per NR, ora con include_reference=True


DISTORTION_SCENARIOS: dict[str, set[int] | None] = {
    "all_levels": None,
    "low_distortion": {1, 2},
    "high_distortion": {4, 5},
}

# Cambia qui se vuoi provare l'altra misura di distanza (coerente con ZS-IQA)
DISTANCE_METRIC = "cosine"  # oppure "l2"


def build_models() -> dict[str, nn.Module]:
    return {
        "siglip2_fr": PipelineVisionEncoder(
            model_config=PipelineVisionEncoderConfiguration(
                model=timm.create_model("vit_base_patch16_siglip_224", pretrained=True),
                feature_extractor=lambda m, x: m.forward_features(x)
            )
        ),
        "clip_fr": PipelineVisionEncoder(
            model_config=PipelineVisionEncoderConfiguration(
                model=timm.create_model("vit_base_patch32_clip_224.openai", pretrained=True),
                feature_extractor=lambda m, x: m.forward_features(x)
            )
        ),
        "dinov2_fr": PipelineVisionEncoder(
            model_config=PipelineVisionEncoderConfiguration(
                model=torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14"),
                feature_extractor=lambda m, x: m.get_intermediate_layers(x, n=1)[0]
            )
        )
    }


def compute_distance(emb_a: torch.Tensor, emb_b: torch.Tensor, metric: str) -> torch.Tensor:
    # Appiattisce tutto tranne il batch, cosi' funziona sia con vettori (B, D)
    # sia con feature multi-dimensionali (B, T, D)
    emb_a = emb_a.reshape(emb_a.shape[0], -1)
    emb_b = emb_b.reshape(emb_b.shape[0], -1)

    if metric == "cosine":
        similarity = F.cosine_similarity(emb_a, emb_b, dim=-1)
        return 1 - similarity  # distanza = 1 - similarita'
    elif metric == "l2":
        return torch.norm(emb_a - emb_b, p=2, dim=-1)
    else:
        raise ValueError(f"Metrica di distanza non supportata: {metric}")


def run_models(models: dict[str, nn.Module], dataloader: DataLoader, img_limit: int, distance_metric: str) -> dict[str, dict[str, list]]:
    models_out: dict[str, dict[str, list]] = {}

    for model_name, model in models.items():
        img_names = []
        distances = []
        real_mos = []
        count = 0

        if hasattr(model, "to") and hasattr(model, "device"):
            model = model.to(model.device).eval()
        elif hasattr(model, "eval"):
            model.eval()

        for distorted, reference, mos, name in dataloader:
            with torch.no_grad():
                emb_distorted = model(distorted)
                emb_reference = model(reference)

            pred_scores = compute_distance(emb_distorted, emb_reference, distance_metric)

            distances.extend(pred_scores.cpu().tolist())
            real_mos.extend(mos.tolist())
            img_names.extend(name)

            count += distorted.size(0)

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
    print(f"[pipeline] start (Full-Reference, distance={DISTANCE_METRIC})")

    conf = config.parse_args()

    if "random_seed" in conf:
        random.seed(conf["random_seed"])

    data = {
        "tid2013": Path(conf["data_cache"]) / "tid2013" / "extracted"
    }

    img_limit = 0  # 0 = usa tutte le immagini disponibili nello scenario
    out_path = Path("out") / f"fr_{str(datetime.datetime.now()).replace(':', '.')}"

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for scenario_name, distortion_levels in DISTORTION_SCENARIOS.items():
        print(f"\n[pipeline] ===== Scenario: {scenario_name} (distortion_levels={distortion_levels}) =====")

        models = build_models()

        dataset = TID2013Dataset(
            txt_index_path=data["tid2013"] / "mos_with_names.txt",
            images_dir=data["tid2013"],
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            distortion_levels=distortion_levels,
            include_reference=True  # NUOVO: attiva il caricamento della reference per FR-IQA
        )

        levels_found = sorted({s.distortion_level for s in dataset.data_samples})
        print(f"[pipeline] filtro distortion_levels richiesto: {distortion_levels}")
        print(f"[pipeline] livelli di distorsione effettivamente presenti nel dataset: {levels_found}")
        print(f"[pipeline] n={len(dataset)} dataset elements successfully mapped")

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

        models_out = run_models(models, dataloader, img_limit, DISTANCE_METRIC)

        scenario_results: dict[str, dict[str, float]] = {}

        for name, output in models_out.items():
            results = compute_metrics(output["distances"], output["real_mos"])
            scenario_results[name] = results

            print(f"[pipeline] {name} - SROCC: {results['SROCC']:.4f}, PLCC: {results['PLCC']:.4f}, KROCC: {results['KROCC']:.4f}")
            print(f"           (nota: score = distanza {DISTANCE_METRIC}; correlazione negativa attesa con MOS)")

            export_predictions(
                filepath=f"{out_path}/{scenario_name}/{name}.csv",
                model_name=name,
                image_names=output["img_names"],
                predictions=output["distances"],
                mos_scores=output["real_mos"]
            )

        all_results[scenario_name] = scenario_results

    print(f"\n[pipeline] output saved to {out_path}")

    print("\n[pipeline] ===== Confronto finale (Full-Reference) tra scenari di distorsione =====")
    for scenario_name, scenario_results in all_results.items():
        print(f"\n--- {scenario_name} ---")
        for model_name, results in scenario_results.items():
            print(f"  {model_name:20s} SROCC: {results['SROCC']:.4f}  PLCC: {results['PLCC']:.4f}  KROCC: {results['KROCC']:.4f}")

    print("\n[pipeline] process finished")


if __name__ == "__main__":
    main()