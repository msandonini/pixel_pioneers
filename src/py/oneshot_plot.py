"""
Genera un bar chart raggruppato che confronta i modelli zero-shot
(siglip2_standard, clip_standard, dinov2_standard) sulle 3 metriche
(SROCC, PLCC, KROCC), con un subplot per ciascuno scenario di distorsione.

Uso:
    python plot_zeroshot_comparison.py --out-path "out/2026-07-20 10.30.00"

Si aspetta la struttura di cartelle prodotta da main.py:
    <out-path>/all_levels/siglip2_standard.csv
    <out-path>/all_levels/clip_standard.csv
    <out-path>/all_levels/dinov2_standard.csv
    <out-path>/low_distortion/...
    <out-path>/high_distortion/...

Ogni CSV deve avere colonne: Model, Image, Prediction, MOS
(esattamente il formato prodotto da pipeline.export.export_predictions)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


# Deve combaciare con i nomi usati in build_models() dentro main.py
MODEL_NAMES = ["siglip2_oneshot", "clip_oneshot", "dinov2_oneshot"]

# Deve combaciare con DISTORTION_SCENARIOS dentro main.py
SCENARIOS = ["all_levels", "low_distortion", "high_distortion"]

# Etichette più leggibili per il grafico
SCENARIO_LABELS = {
    "all_levels": "All levels",
    "low_distortion": "Low distortion (1,2)",
    "high_distortion": "High distortion (4,5)",
}
MODEL_LABELS = {
    "siglip2_oneshot": "SigLIP2",
    "clip_oneshot": "CLIP",
    "dinov2_oneshot": "DINOv2",
}


def compute_metrics_from_csv(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    preds = df["Prediction"].to_numpy()
    mos = df["MOS"].to_numpy()

    plcc, _ = pearsonr(preds, mos)
    srocc, _ = spearmanr(preds, mos)
    krocc, _ = kendalltau(preds, mos)

    return {"PLCC": plcc, "SROCC": srocc, "KROCC": krocc}


def collect_results(out_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """
    Ritorna: { scenario: { model_name: {"PLCC":.., "SROCC":.., "KROCC":..} } }
    """
    results: dict[str, dict[str, dict[str, float]]] = {}

    for scenario in SCENARIOS:
        results[scenario] = {}
        for model_name in MODEL_NAMES:
            csv_path = out_path / scenario / f"{model_name}.csv"
            if not csv_path.exists():
                print(f"[warn] file non trovato, salto: {csv_path}")
                continue
            results[scenario][model_name] = compute_metrics_from_csv(csv_path)

    return results


def plot_comparison(results: dict[str, dict[str, dict[str, float]]], save_path: Path):
    metrics = ["SROCC", "PLCC", "KROCC"]
    n_scenarios = len(SCENARIOS)

    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4.5), sharey=True)
    if n_scenarios == 1:
        axes = [axes]

    bar_width = 0.25
    x = np.arange(len(metrics))

    for ax, scenario in zip(axes, SCENARIOS):
        scenario_results = results.get(scenario, {})

        for i, model_name in enumerate(MODEL_NAMES):
            if model_name not in scenario_results:
                continue
            values = [scenario_results[model_name][m] for m in metrics]
            offset = (i - (len(MODEL_NAMES) - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, values, bar_width,
                label=MODEL_LABELS.get(model_name, model_name)
            )
            # Etichetta numerica sopra ogni barra
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7
                )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_ylim(-0.3, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Correlation")
    axes[-1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Zero-Shot Model Comparison on TID2013", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[plot] salvato in: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-path", type=str, required=True,
        help="Cartella prodotta da main.py, es. 'out/2026-07-20 10.30.00'"
    )
    parser.add_argument(
        "--save-path", type=str, default="images/zeroshot_comparison.png",
        help="Percorso di output per il grafico"
    )
    args = parser.parse_args()

    out_path = Path(args.out_path)
    results = collect_results(out_path)

    # Stampa anche a schermo, comodo per doppio controllo
    for scenario, scenario_results in results.items():
        print(f"\n--- {scenario} ---")
        for model_name, metrics in scenario_results.items():
            print(f"  {model_name:20s} SROCC: {metrics['SROCC']:.4f}  PLCC: {metrics['PLCC']:.4f}  KROCC: {metrics['KROCC']:.4f}")

    plot_comparison(results, Path(args.save_path))


if __name__ == "__main__":
    main()