from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen, core
import optax

from .learner import Learner, TrainState, functional as F, transforms as T

Batch = Any


class PseudoLabel(Learner):
    """PseudoLabel Learner.

    Args:
        data_meta: Meta information of the dataset.
        train_steps (int): Total number of steps for training.
        base_model (Module): Base model.
        tx (GradientTransformation): Optax optimizer.
        label_smoothing (float): Label smoothing parameter.
        momentum_ema (float): Momentum value to update EMA model.
        precision (str): Precision. fp16, bf16 and fp32 are supported.
        lambda_y (float): Coefficient of the unsupervised loss.
        threshold (float): Threshold to filter low-confidence predictions.
        unsup_warmup_pos (float): Warmup position of epochs.
    """

    default_entries: list[str] = [
        "warmup",
        "loss",
        "sup_loss",
        "unsup_loss",
        "conf",
        "mask_prob",
        "acc1",
    ]

    def __init__(
        self,
        data_meta: dict,
        train_steps: int,
        base_model: linen.Module,
        tx: optax.GradientTransformation,
        lambda_y: float = 100,
        threshold: float = 0.95,
        unsup_warmup_pos: float = 1 / 64,
        label_smoothing: float = 0,
        momentum_ema: float = 0.999,
        precision: str = "fp32",
    ) -> None:
        super().__init__(
            data_meta, train_steps, base_model, tx, label_smoothing, momentum_ema, precision
        )
        self.lambda_y = lambda_y
        self.unsup_warmup_pos = unsup_warmup_pos
        self.threshold = threshold

    def loss_fn(self, params: core.FrozenDict, train_state: TrainState, batch: Batch):
        rng, new_rng = jr.split(train_state.rng)
        rng = jr.fold_in(rng, jax.lax.axis_index("batch"))
        warmup = jnp.clip(train_state.step / self.train_steps / self.unsup_warmup_pos, 0, 1)
        transform = T.random_crop_flip(self.data_meta["image_size"], self.data_meta["no_flip"])

        def apply_fn(x, params):
            variables = {"params": params, **train_state.model_state}
            output, new_model_state = train_state.apply_fn(
                variables, x, train=True, mutable=["batch_stats"]
            )
            logits = output["logits"]
            return logits, new_model_state

        x_rng, y_rng = jr.split(rng)
        x = transform(x_rng, batch["labeled"]["inputs"] / 255.0)
        y = transform(y_rng, batch["unlabeled"]["inputs"] / 255.0)
        lx = F.one_hot(
            batch["labeled"]["labels"], self.data_meta["num_classes"], self.label_smoothing
        )

        logits_x, new_model_state = apply_fn(x, params)
        sup_loss = F.cross_entropy(logits_x, lx).mean()

        logits_y, _ = apply_fn(y, params)
        probs_y = linen.softmax(logits_y)
        ly = F.one_hot(jnp.argmax(probs_y, axis=-1), self.data_meta["num_classes"])
        max_probs = jnp.max(probs_y, axis=-1)
        mask = (max_probs > self.threshold).astype(jnp.float32)
        unsup_loss = (mask * F.cross_entropy(logits_y, ly)).mean()

        loss = sup_loss + self.lambda_y * warmup * unsup_loss
        updates = {"model_state": new_model_state, "rng": new_rng}
        scalars = {
            "loss": loss,
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "warmup": warmup,
            "conf": max_probs.mean(),
            "mask_prob": mask.mean(),
            "acc1": F.accuracy(logits_x, lx),
            "acc5": F.accuracy(logits_x, lx, k=5),
        }

        return loss, (updates, scalars)
