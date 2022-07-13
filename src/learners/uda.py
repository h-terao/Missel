from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen, core
import optax

from .learner import Learner, TrainState, functional as F, transforms as T

Batch = Any


class UDA(Learner):
    """Unsupervised Data Augmentation (UDA) learner.

    Args:
        data_meta: Meta information of the dataset.
        train_steps (int): Total number of steps for training.
        base_model (Module): Base model.
        tx (GradientTransformation): Optax optimizer.
        label_smoothing (float): Label smoothing parameter.
        momentum_ema (float): Momentum value to update EMA model.
        precision (str): Precision. fp16, bf16 and fp32 are supported.
        tsa (str): Type of TSA schedule. none, linear, log or exp.
        lambda_y (float): Coefficient of the unsupervised loss.
        T (float): Temperature parameter.
        threshold (float): Threshold parameter.
        unsup_warmup_pos (float): Warmup position of epochs.
    """

    default_entries: list[str] = [
        "warmup",
        "tsa",
        "loss",
        "sup_loss",
        "unsup_loss",
        "conf",
        "sup_mask",
        "unsup_mask",
        "acc1",
    ]

    def __init__(
        self,
        data_meta: dict,
        train_steps: int,
        base_model: linen.Module,
        tx: optax.GradientTransformation,
        tsa: str = "linear",
        lambda_y: float = 1.0,
        T: float = 0.4,
        threshold: float = 0.8,
        label_smoothing: float = 0,
        momentum_ema: float = 0.999,
        precision: str = "fp32",
    ) -> None:
        assert tsa in ["linear", "exp", "log", "none"]
        super().__init__(
            data_meta, train_steps, base_model, tx, label_smoothing, momentum_ema, precision
        )
        self.tsa_schedule = tsa
        self.lambda_y = lambda_y
        self.T = T
        self.threshold = threshold

    def TSA(self, step: int):
        """Training signal annealing."""
        nC = self.data_meta["num_classes"]
        threshold = step / self.train_steps
        if self.tsa_schedule == "linear":
            pass
        elif self.tsa_schedule == "exp":
            scale = 5.0
            threshold = jnp.exp((threshold - 1) * scale)
        elif self.tsa_schedule == "log":
            scale = 5.0
            threshold = 1 - jnp.exp(-threshold * scale)
        elif self.tsa_schedule == "none":
            return 1.0
        else:
            raise ValueError
        tsa = threshold * (1 - 1 / nC) + 1 / nC
        return tsa

    def loss_fn(self, params: core.FrozenDict, train_state: TrainState, batch: Batch):
        raise NotImplementedError

        rng, new_rng = jr.split(train_state.rng)
        rng = jr.fold_in(rng, jax.lax.axis_index("batch"))
        transform_weak = T.random_crop_flip(self.data_meta["image_size"], self.data_meta["no_flip"])
        transform_strong = T.random_crop_flip(
            self.data_meta["image_size"], self.data_meta["no_flip"]
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

        tsa = self.TSA(train_state.step)
        sup_mask = (jnp.max(linen.softmax(logits_x), axis=-1) < tsa).astype(jnp.float32)
        sup_mask = jax.lax.stop_gradient(sup_mask)
        sup_loss = (F.cross_entropy(logits_x, lx) * sup_mask).mean()

        logits_y_w = jax.lax.stop_gradient(logits_y_w)
        probs_y = linen.softmax(logits_y_w / self.T)
        ly = F.one_hot(jnp.argmax(probs_y, axis=-1), self.data_meta["num_classes"])
        max_probs = jnp.max(probs_y, axis=-1)
        unsup_mask = (max_probs > self.threshold).astype(jnp.float32)
        unsup_loss = (unsup_mask * F.cross_entropy(logits_y_s, ly)).mean()

        loss = sup_loss + self.lambda_y * unsup_loss
        updates = {"model_state": new_model_state, "rng": new_rng}
        scalars = {
            "loss": loss,
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "tsa": tsa,
            "conf": max_probs.mean(),
            "sup_mask": sup_mask.mean(),
            "unsup_mask": unsup_mask.mean(),
            "acc1": F.accuracy(logits_x, lx),
            "acc5": F.accuracy(logits_x, lx, k=5),
        }
        return loss, (updates, scalars)
