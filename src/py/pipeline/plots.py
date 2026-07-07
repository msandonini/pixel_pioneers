import matplotlib.pyplot as plt
import seaborn as sns


def scatter_plot(
    predictions,
    mos_scores,
    metrics,
    title,
    output_path=None,
):
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=predictions,
        y=mos_scores,
        s=70,
    )

    text = "\n".join(
        f"{metric}: {value:.4f}"
        for metric, value in metrics.items()
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.5),
    )

    plt.title(f"{title} - Prediction vs MOS")
    plt.xlabel("Model Prediction")
    plt.ylabel("MOS")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_metric(
    df,
    metric_name,
    output_path=None,
):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Model",
        y=metric_name,
    )

    sns.stripplot(
        data=df,
        x="Model",
        y=metric_name,
        color="black",
        alpha=0.5,
    )

    plt.title(f"Comparison of {metric_name}")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def boxplot_iqa_metrics(
    df,
    output_path=None,
):
    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="Metric",
        y="Score",
        hue="Model",
    )

    plt.title("IQA Metric Comparison Across Models")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()
