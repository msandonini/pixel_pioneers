# pixel_pioneers
Repo containing the final project for our Computer Vision and Cognitive Systems university course.

## How to run

The instructions below expect the `python <main.py>` command to be run from inside the project root.
Each dependency needed to run the project can be found in the `requirements.txt` file.
In order to have the same file structure and download the datasets to the correct directories on all PCs, we used the `config.yaml` file.
The configuration file is loaded from its position by calling the `parse_args()` function located inside of `pipeline.config`
It's possible to change the configuration file path by adding the argument `--config <path/to/file.yaml>`

In order to run the files on the cluster a `job_run.sh` SBATCH script was created, which can be changed in order to correctly point to the desired venv and project directory

### Embeddings extraction

The metric MLP and fusion MLP implementations need all the embeddings to be extracted beforehand.
In order to do so, inside of `embeddings.models`, we created a main for each foundation model we used which downloads the model from HuggingFace and loads the desired dataset.
The datasets can be downloaded, extracted and cached via the functions inside of `pipeline.data_cache`.
The provided mains inside of `embeddings.models` should automatically take care of it by calling the functions when run.

### Zero-Shot models
The evaluation pipeline for this setting can be found in the `multiple_models_main.py` file.
The PipelineVisionEncoder and its configuration can be found inside  `pipeline.model`.

The dataset is loaded via TID2013Dataset, modeled inside  `data_loaders.tid2013`, with
include_reference=True so that each sample provides both the distorted image and its
corresponding reference image.

Unlike the Metric MLP and Fusion MLP, this setting requires no training: no parameters are
learned, and each encoder is evaluated purely on its frozen, pre-trained embeddings.

Three pre-trained vision encoders are evaluated:

SigLIP2 (vit_base_patch16_siglip_224)
CLIP (vit_base_patch32_clip_224.openai)
DINOv2 (dinov2_vitb14)

Since a higher distance indicates lower similarity between the distorted and reference
embeddings (i.e. a more severe distortion), a negative correlation with the MOS is expected.

The pipeline evaluates each encoder across three distortion-level scenarios (defined in
DISTORTION_SCENARIOS), in order to assess whether correlation with human MOS depends on
distortion severity:

all_levels: the full dataset, including all 5 distortion levels.
low_distortion: only levels 1 and 2 (mild, subtle distortions).
high_distortion: only levels 4 and 5 (severe, clearly visible distortions).

For each scenario, the following correlation metrics are computed via
 `pipeline.metrics.compute_metrics`:

PLCC (Pearson Linear Correlation Coefficient)
SROCC (Spearman Rank Correlation Coefficient)
KROCC (Kendall Rank Correlation Coefficient)

Per-image predictions, image names, and ground-truth MOS values are exported to CSV via
 `pipeline.export.export_predictions`, saved under out/fr_<timestamp>/<scenario_name>/<model_name>.csv.

After evaluating every encoder and scenario, the pipeline prints a summary table comparing
zero-shot performance across models and distortion-level scenarios.


### Metric MLP

The training pipeline for this model can be found in the `metrics_main.py` file
The `MetricMLP` module implementation can be found inside `metric_mlp.mlp`.

The data is loaded from the `EmbeddingDataset` modeled inside `data_loaders.embedding_data`
The dataset is expected to contain pre-computed embeddings for:

- reference images
- corresponding distorted images
- ground truth MOS values

Each embedding model found in the dataset is processed independently, resulting in one trained `MetricMLP` per encoder.

The dataset is split using a reference-image split rather than a random image split.

All distorted versions of a given reference image are assigned exclusively to one of:

- Training set (68%)
- Validation set (16%)
- Test set (16%)

This prevents information leakage between dataset partitions and provides a more realistic evaluation of generalization.

The pipeline iterates over every embedding model in the dataset.

For each model:

1. Reference and distorted embeddings are loaded.
2. A new `MetricMLP` is initialized.
3. The network predicts the MOS from each pair of embeddings.
4. Predictions are optimized using the IQA loss function.

Training uses:

- `AdamW` optimizer
- learning rate: 1e-3
- weight decay: 1e-4
- batch size: 64
- maximum epochs: 100

The learning rate is automatically reduced when the validation SROCC plateaus.

After each epoch, the current model is evaluated on the validation set.

The following correlation metrics are computed:

- PLCC (Pearson Linear Correlation Coefficient)
- SROCC (Spearman Rank Correlation Coefficient)
- KROCC (Kendall Rank Correlation Coefficient)

The checkpoint achieving the highest validation SROCC is retained.

Training stops automatically when the validation SROCC has not improved for a number of epochs equal to twice the scheduler patience.

After training finishes the best-performing model (highest validation SROCC) is restored and gets evaluated on the test set.

The same evaluation metrics are reported:
- PLCC
- SROCC
- KROCC
If an output directory is specified, the pipeline saves the best checkpoint for each encoder separately.

Each checkpoint contains the trained weights of the corresponding `MetricMLP`, allowing it to be reloaded later for inference or evaluation.

After all encoders have been trained, the pipeline prints a summary table comparing their performance on the test set.

This enables direct comparison of the predictive power of the embeddings produced by different vision foundation models

### Fusion MLP

The training pipeline for this model can be found in the `fusion_mlp_main.py` file
The `FusionMLP` module implementation can be found inside `fusion.mlp.FusionMLP`.

The data is loaded from the `EmbeddingDataset` modeled inside `data_loaders.embedding_data`
The dataset is expected to contain pre-computed embeddings for:

- reference images
- corresponding distorted images
- ground truth MOS values

Embeddings from different models are automatically detected and used as inputs to the fusion network.

The dataset is not split randomly by image, instead it performs a reference-image split, so that all distorted versions of the same reference image belong to exactly one of:

- training set (68%)
- validation set (16%)
- test set (16%)

This prevents information leakage since distortions originating from the same reference image never appear in multiple splits.

For each mini-batch:

1. Reference embeddings are loaded for every model.
2. Distorted embeddings are loaded.
3. The `FusionMLP` produces a common embedding for the reference image.
4. The same `FusionMLP` produces a common embedding for the distorted image.
5. The `MetricMLP` predicts the MOS from the two fused embeddings.
6. The prediction is optimized using the IQA loss function.

Training uses:

- `AdamW` optimizer
- learning rate: 1e-3
- weight decay: 1e-4
- batch size: 64
- maximum epochs: 100

The learning rate is automatically reduced when the validation SROCC stops improving.

After every epoch the model is evaluated on the validation set.

The following metrics are computed:

- MSE Loss
- PLCC (Pearson Linear Correlation Coefficient)
- SROCC (Spearman Rank Correlation Coefficient)
- KROCC (Kendall Rank Correlation Coefficient)

The model with the highest validation SROCC is kept as the best checkpoint.

Training stops automatically when the validation SROCC has not improved for a number of epochs equal to twice the scheduler patience.

After training finishes the best-performing model (highest validation SROCC) is restored and gets evaluated on the test set.

The same evaluation metrics are reported:
- PLCC
- SROCC
- KROCC
- MSE Loss

The saved checkpoint contains:

- `FusionMLP` weights
- `MetricMLP` weights
- list of models used
- input embedding dimensions
- fused embedding dimension
- best validation SROCC
- epoch corresponding to the best model
