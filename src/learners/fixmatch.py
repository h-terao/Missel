from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen, core
import optax

from .learner import Learner, TrainState, functional as F, transforms as T

Batch = Any


class FixMatch(Learner):
    """FixMatch learner.

    Args:
        data_meta: Meta information of the dataset.
        train_steps: Total number of steps for training.
        base_model: Base model.
        tx: Optax optimizer.
        label_smoothing: Label smoothing parameter.
        momentum_ema: Momentum value to update EMA model.
        precision: Precision. fp16, bf16 and fp32 are supported.
        lambda_y: Coefficient of the unsupervised loss.
        T: Temperature parameter. If zero, use one-hot labels for unsupervised loss.
        threshold: Threshold parameter.
    """

    default_entries: list[str] = [
        "loss",
        "sup_loss",
        "unsup_loss",
        "conf",
        "mask",
        "acc1",
    ]

    def __init__(
        self,
        data_meta: dict,
        train_steps: int,
        base_model: linen.Module,
        tx: optax.GradientTransformation,
        lambda_y: float = 1.0,
        T: float = 0,
        threshold: float = 0.95,
        label_smoothing: float = 0,
        momentum_ema: float = 0.999,
        precision: str = "fp32",
    ) -> None:
        super().__init__(
            data_meta, train_steps, base_model, tx, label_smoothing, momentum_ema, precision
        )
        self.lambda_y = lambda_y
        self.T = T
        self.threshold = threshold

    def loss_fn(self, params: core.FrozenDict, train_state: TrainState, batch: Batch):
        rng, new_rng = jr.split(train_state.rng)
        rng = jr.fold_in(rng, jax.lax.axis_index("batch"))
        transform_weak = T.random_crop_flip(self.data_meta["image_size"], self.data_meta["no_flip"])
        transform_strong = T.randaugment(
            self.data_meta["image_size"], self.data_meta["no_flip"], cutout=True
        )

        def apply_fn(x, params):
            variables = {"params": params, **train_state.model_state}
            output, new_model_state = train_state.apply_fn(
                variables, x, train=True, mutable=["batch_stats"]
            )
            logits = output["logits"]
            return logits, new_model_state

        x_rng, y_w_rng, y_s_rng = jr.split(rng, 3)
        x = transform_weak(x_rng, batch["labeled"]["inputs"] / 255.0)
        y_w = transform_weak(y_w_rng, batch["unlabeled"]["inputs"] / 255.0)
        y_s = transform_strong(y_s_rng, batch["unlabeled"]["inputs"] / 255.0)
        lx = F.one_hot(
            batch["labeled"]["labels"], self.data_meta["num_classes"], self.label_smoothing
        )

        inputs = jnp.concatenate([x, y_w, y_s], axis=0)
        logits, new_model_state = apply_fn(inputs, params)
        logits_x = logits[: len(x)]
        logits_y_w, logits_y_s = jnp.split(logits[len(x) :], 2)

        sup_loss = F.cross_entropy(logits_x, lx).mean()

        logits_y_w = jax.lax.stop_gradient(logits_y_w)
        probs_y = linen.softmax(logits_y_w)
        max_probs = jnp.max(probs_y, axis=-1)
        mask = (max_probs > self.threshold).astype(jnp.float32)
        if self.T > 0:
            ly = linen.softmax(logits_y_w / self.T)
        else:
            ly = F.one_hot(jnp.argmax(probs_y, axis=-1), self.data_meta["num_classes"])
        unsup_loss = (mask * F.cross_entropy(logits_y_s, ly)).mean()

        loss = sup_loss + self.lambda_y * unsup_loss
        updates = {"model_state": new_model_state, "rng": new_rng}
        scalars = {
            "loss": loss,
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "conf": max_probs.mean(),
            "mask": mask.mean(),
            "acc1": F.accuracy(logits_x, lx),
            "acc5": F.accuracy(logits_x, lx, k=5),
        }
        return loss, (updates, scalars)
