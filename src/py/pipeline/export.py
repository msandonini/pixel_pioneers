import pandas as pd
from pathlib import Path


def export_predictions(
    filepath,
    model_name,
    image_names,
    predictions,
    mos_scores,
):
    df = pd.DataFrame({
        "Model": model_name,
        "Image": image_names,
        "Prediction": predictions,
        "MOS": mos_scores,
    })

    filepath = Path(filepath)
    filepath = Path(str(filepath).replace(":", "-"))
    filepath.parent.mkdir(parents=True, exist_ok=True)

    write_header = not filepath.exists()

    df.to_csv(
        filepath,
        mode="a",
        index=False,
        header=write_header,
    )
