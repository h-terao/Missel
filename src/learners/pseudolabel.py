from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from flax import core

from .learner import Learner, TrainState, functional as F, transforms as T

Batch = Any


class PseudoLabel(Learner):
    """A Learner for Pseudo Labeling.

    Args:
        data_meta: Meta information of the dataset.
        train_steps (int): Total number of steps for training.
        base_model (Module): Base model.
        tx (GradientTransformation): Optax optimizer.
        label_smoothing (float): Label smoothing parameter.
        momentum_ema (float): Momentum value to update EMA model.
        precision (str): Precision. fp16, bf16 or fp32.

        lambda_y (float): Coefficient of the unsupervised loss.
        threshold (float): Threshold to cutoff.
        unsup_warmup_pos (float): Warmup position of epochs.
    """

    lambda_y: float = 1.0
    threshold: float = 0.9
    unsup_warmup_pos: float = 0.4

    default_entries: list[str] = [
        "warmup",
        "loss",
        "ce_loss",
        "pl_loss",
        "confidence",
        "mask_prob",
        "acc1",
    ]

    def loss_fn(self, params: core.FrozenDict, train_state: TrainState, batch: Batch):
        rng, new_rng = jr.split(train_state.rng)
        rng = jr.fold_in(rng, jax.lax.axis_index("batch"))
        warmup = jnp.clip(train_state.step / self.train_steps / self.unsup_warmup_pos, 0, 1)
        transform = jax.vmap(T.random_crop_flip(self.data_meta["image_size"]))

        def apply_fn(x):
            variables = {"params": params, **train_state.model_state}
            output, new_model_state = train_state.apply_fn(
                variables, x, train=True, mutable=["batch_stats"]
            )
            logits = output["logits"]
            return logits, new_model_state

        x = batch["labeled"]["inputs"] / 255.0
        y = batch["unlabeled"]["inputs"] / 255.0
        lx = F.one_hot(
            batch["labeled"]["labels"], self.data_meta["num_classes"], self.label_smoothing
        )

        rng, x_rng, y_rng = jr.split(rng, 3)
        x = transform(jr.split(x_rng, len(x)), x)
        y = transform(jr.split(y_rng, len(y)), y)

        logits_x, new_model_state = apply_fn(x)
        ce_loss = F.cross_entropy(logits_x, lx).mean()

        logits_y, _ = apply_fn(y)
        pseudo_labels = nn.softmax(logits_y)

        ly = jnp.argmax(pseudo_labels, axis=-1)
        masks = jnp.sum(pseudo_labels > self.threshold, axis=-1)
        pl_loss = (F.cross_entropy(logits_y, ly) * masks).mean()

        loss = ce_loss + self.lambda_y * warmup * pl_loss
        updates = {
            "model_state": new_model_state,
            "rng": new_rng,
        }
        scalars = {
            "loss": loss,
            "ce_loss": ce_loss,
            "pl_loss": pl_loss,
            "confidence": jnp.max(pseudo_labels, axis=-1).mean(),
            "mask_prob": masks.mean(),
            "warmup": warmup,
            "acc1": F.accuracy(logits_x, lx, k=1),
            "acc5": F.accuracy(logits_x, lx, k=5),
        }

        return loss, (updates, scalars)
