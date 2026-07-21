# pixel_pioneers
Repo containing the final project for our Computer Vision and Cognitive Systems university course.

## How to run

Each dependency needed to run the project is found in the `requirements.txt` in the project root.

### Fusion MLP

The fusion MLP can be found in the `FusionMLP` class inside `fusion.mlp`.
An example on how to train it can be found in `fusion_mlp_main.py`

In order to run the fusion MLP, embeddings from visual encoders must be previously extracted.
Once the embeddings are obtained they must be stored in a torch `Dataset`.

Once the dataset is ready, the MLP must be trained on the embeddings so that it learns how to fuse them.

In order to train `FusionMLP`, we provide a `MetricMLP` class, located in `metric_mlp.mlp`, which needs to be trained alongside `FusionMLP`, in order to drive its training.

In our training, we used an `AdamW` optimizer in combination with a `ReduceLROnPlateau` scheduler, with a patience initialized as following:

```python
optimizer = torch.optim.AdamW(
    list(fusion_mlp.parameters()) +
    list(metric_mlp.parameters()),
    lr = 1e-3,
    weight_decay=1e-4
)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5
)
```

In our case, training was performed with a limit of 100 epochs, with an early stopping set to 2 times the patience of the scheduler.
Both early stopping and the scheduler were watching the growth of the SROCC metric.

Once the model is trained, we suggest to re-load the best model obtained.
