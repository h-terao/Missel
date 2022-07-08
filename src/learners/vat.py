from __future__ import annotations
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from flax import core
import chex

from .learner import Learner, TrainState, functional as F, transforms as T

Batch = Any


class VAT(Learner):
    """A Learner for Virtual Adversarial Training.

    Args:
        data_meta: Meta information of the dataset.
        train_steps (int): Total number of steps for training.
        base_model (Module): Base model.
        tx (GradientTransformation): Optax optimizer.
        label_smoothing (float): Label smoothing parameter.
        momentum_ema (float): Momentum value to update EMA model.
        precision (str): Precision. fp16, bf16 and fp32 are supported.
        lambda_y (float): Coefficient of the unsupervised loss.
        lambda_entmin (float): Coefficient of the entropy loss.
        unsup_warmup_pos (float): Warmup position of epochs.
        vat_eps (float): Norm of adversarial noises.
        xi (float): Norm of adversarial noises to sample.
        num_iters (int): Number of steps to update the adversarial noises.
    """

    lambda_y: float = 1.0
    lambda_entmin: float = 0.0
    unsup_warmup_pos: float = 0.4
    vat_eps: float = 6
    xi: float = 1e-6
    num_iters: int = 1
    default_entries: list[str] = ["warmup", "loss", "ce_loss", "vat_loss", "entmin_loss", "acc1"]

    def vat_loss(
        self, rng: chex.PRNGKey, y: chex.Array, logits_y: chex.Array, apply_fn: Callable
    ) -> chex.Array:
        """Compute VAT loss.

        Args:
            rng (Array): A PRNG key.
            y (Array): Unlabeled images.
            logits_y (Array): Logits computed from y.
            apply_fn (Callable): Forward function of the model.

        Returns:
            VAT loss.
        """

        def grad_fn(z: chex.Array):
            logits_yhat = apply_fn(y + z)
            loss = F.kl_div(logits_y, nn.log_softmax(logits_yhat)).mean()
            return loss

        def normalize(x: chex.Array):
            x /= jnp.linalg.norm(x, ord=2, axis=(-1, -2, -3), keepdims=True) + 1e-6
            return x

        def scan_fn(z: chex.Array, _):
            z = self.xi * normalize(z)
            z = grad_fn(z)
            return z, _

        z = jr.normal(rng, y.shape, dtype=y.dtype)
        z = jax.lax.scan(scan_fn, z, jnp.arange(self.num_iters))
        yhat = y + self.vat_eps * normalize(z)
        logits_yhat = apply_fn(yhat)

        vat_loss = F.kl_div(logits_yhat, nn.log_softmax(logits_y), log_targets=True).mean()
        return vat_loss

    def loss_fn(self, params: core.FrozenDict, train_state: TrainState, batch: Batch):
        """Compute loss of the VAT training.

        Args:
            params (FrozenDict): Model parameters.
            train_state (TrainState): Training state.
            batch (Batch): Batch.

        Returns:
            Total loss, updates and metrics.
        """
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
        vat_loss = self.vat_loss(rng, y, logits_y, apply_fn)
        entmin_loss = F.entropy(logits_y).mean()

        loss = ce_loss + self.lambda_y * warmup * vat_loss + self.lambda_entmin * entmin_loss
        updates = {"model_state": new_model_state, "rng": new_rng}
        scalars = {
            "loss": loss,
            "ce_loss": ce_loss,
            "vat_loss": vat_loss,
            "entmin_loss": entmin_loss,
            "warmup": warmup,
            "acc1": F.accuracy(logits_x, lx, k=1),
            "acc5": F.accuracy(logits_x, lx, k=5),
        }
        return loss, (updates, scalars)
