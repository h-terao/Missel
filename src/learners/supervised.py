from __future__ import annotations
from typing import Any

import jax
import jax.random as jr
from flax import core

from .learner import Learner, TrainState, functional as F, transforms as T

Batch = Any


class Supervised(Learner):
    """Supervised learner.

    Args:
        data_meta: Meta information of the dataset.
        train_steps (int): Total number of steps for training.
        base_model (Module): Base model.
        tx (GradientTransformation): Optax optimizer.
        label_smoothing (float): Label smoothing parameter.
        momentum_ema (float): Momentum value to update EMA model.
        precision (str): Precision. fp16, bf16 and fp32 are supported.
    """

    default_entries: list[str] = ["warmup", "loss", "sup_loss", "acc1"]

    def loss_fn(self, params: core.FrozenDict, train_state: TrainState, batch: Batch):
        rng, new_rng = jr.split(train_state.rng)
        rng = jr.fold_in(rng, jax.lax.axis_index("batch"))
        transform = T.random_crop_flip(self.data_meta["image_size"], self.data_meta["no_flip"])

        x = transform(rng, batch["labeled"]["inputs"] / 255.0)
        lx = F.one_hot(
            batch["labeled"]["labels"], self.data_meta["num_classes"], self.label_smoothing
        )

        variables = {"params": params, **train_state.model_state}
        output, new_model_state = train_state.apply_fn(
            variables, x, train=True, mutable=["batch_stats"]
        )
        logits_x = output["logits"]
        del output

        loss = F.cross_entropy(logits_x, lx).mean()
        updates = {"model_state": new_model_state, "rng": new_rng}
        scalars = {
            "loss": loss,
            "acc1": F.accuracy(logits_x, lx, k=1).mean(),
            "acc5": F.accuracy(logits_x, lx, k=5).mean(),
        }
        return loss, (updates, scalars)
