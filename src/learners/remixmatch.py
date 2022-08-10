from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen, core
import optax
from squidink.functional import rot90
import chex

from .learner import (
    Learner,
    TrainState as TrainStateBase,
    classifiers,
    functional as F,
    transforms as T,
)

Batch = Any


class TrainState(TrainStateBase):
    p_model: chex.Array | None = None


class RemixMatch(Learner):
    """RemixMatch learner.

    Args:
        data_meta: Meta information of the dataset.
        train_steps: Total number of steps for training.
        base_model: Base model.
        tx: Optax optimizer.
        label_smoothing: Label smoothing parameter.
        momentum_ema: Momentum value to update EMA model.
        precision: Precision. fp16, bf16 and fp32 are supported.
        lambda_y: Coefficient of the unsupervised loss.
        lambda_match
        lambda_rot: Coefficient of the rotation loss.
        T: Temperature parameter. If zero, use one-hot labels for unsupervised loss.
    """

    train_state_cls = TrainState
    classifier_cls = classifiers.RotClassifier
    default_entries: list[str] = [
        "loss",
        "sup_loss",
        "rot_loss",
        "unsup_loss",
        "unsup_mix_loss",
        "acc1",
        "rot_acc1",
    ]

    def __init__(
        self,
        data_meta: dict,
        train_steps: int,
        base_model: linen.Module,
        tx: optax.GradientTransformation,
        lambda_y: float = 1.0,
        lambda_match: float = 1.0,
        lambda_rot: float = 1.0,
        unsup_warmup_pos: float = 0.4,
        label_smoothing: float = 0,
        momentum_ema: float = 0.999,
        precision: str = "fp32",
    ) -> None:
        super().__init__(
            data_meta, train_steps, base_model, tx, label_smoothing, momentum_ema, precision
        )
        self.lambda_y = lambda_y
        self.lambda_match = lambda_match
        self.lambda_rot = lambda_rot
        self.unsup_warmup_pos = unsup_warmup_pos
        self.T = T

    def init_fn(
        self, rng: chex.PRNGKey, batch: Batch, **kwargs
    ) -> tuple[TrainState, dict[str, int | float]]:
        # Initialize p_model as the uniform distribution.
        num_classes = self.data_meta["num_classes"]
        p_model = jnp.full((num_classes,), 1.0 / num_classes)
        return super().init_fn(rng, batch, p_model=p_model, **kwargs)

    def loss_fn(self, params: core.FrozenDict, train_state: TrainState, batch: Batch):
        rng, new_rng = jr.split(train_state.rng)
        rng = jr.fold_in(rng, jax.lax.axis_index("batch"))
        warmup = jnp.clip(train_state.step / self.train_steps / self.unsup_warmup_pos, 0, 1)
        transform_weak = T.random_crop_flip(self.data_meta["image_size"], self.data_meta["no_flip"])
        transform_strong = T.randaugment(
            self.data_meta["image_size"], self.data_meta["no_flip"], cutout=True
        )

        def apply_fn(x, params, return_rot: bool = False):
            variables = {"params": params, **train_state.model_state}
            output, new_model_state = train_state.apply_fn(
                variables, x, train=True, mutable=["batch_stats"]
            )
            if return_rot:
                return output["logits"], new_model_state
            else:
                return output["logits_rot"], new_model_state

        if self.data_meta["dist"] is not None:
            p_target = self.data_meta["dist"]
        else:
            nC = self.data_meta["num_classes"]
            p_target = jnp.full((nC,), 1 / nC)

        rng, y_w_rng = jr.split(rng)
        y_w = transform_weak(y_w_rng, batch["unlabeled"]["inputs"] / 255.0)
        probs_y = linen.softmax(apply_fn(y_w, params)[0], axis=-1)
        p_model = train_state.p_model * 0.999 + jnp.mean(probs_y, axis=0) * 0.001

        ly = probs_y * p_target / p_model
        ly /= jnp.sum(ly, axis=-1, keepdims=True)

        # sharpen.
        ly = ly ** (1 / self.T)
        ly /= jnp.sum(ly, axis=-1, keepdims=True)
        ly = jax.lax.stop_gradient(ly)

        rng, x_rng, y_s1_rng, y_s2_rng, y_rot_rng = jr.split(rng, 5)
        x = transform_weak(x_rng, batch["labeled"]["inputs"] / 255.0)
        y_s1 = transform_strong(y_s1_rng, batch["unlabeled"]["inputs"] / 255.0)
        y_s2 = transform_strong(y_s2_rng, batch["unlabeled"]["inputs"] / 255.0)
        y_rot, l_rot = rotate(y_rot_rng, batch["unlabeled"]["inputs"] / 255.0)
        lx = F.one_hot(
            batch["labeled"]["labels"], self.data_meta["num_classes"], self.label_smoothing
        )

        # mixup
        inputs = jnp.concatenate([x, y_s1, y_s2, y_w], axis=0)
        labels = jnp.concatenate([lx, ly, ly, ly], axis=0)
        index_rng, mixup_ratio_rng = jr.split(rng)
        index = jr.permutation(index_rng, len(inputs))
        lam = jr.beta(mixup_ratio_rng, self.alpha, self.alpha)
        lam = jnp.maximum(lam, 1 - lam)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        # interleave.
        mixed_inputs = jnp.array_split(mixed_inputs, len(inputs) // len(x))
        mixed_inputs = interleave(mixed_inputs)

        logits, new_model_state = apply_fn(mixed_inputs[0], params)
        logits = [logits] + [apply_fn(v, params)[0] for v in mixed_inputs[1:]]
        logits = interleave(logits)

        logits_x, logits_y = logits[0], jnp.concatenate(logits[1:])
        labels_x, labels_y = mixed_labels[: len(x)], mixed_labels[len(x) :]
        assert len(logits_x) == len(labels_x)
        assert len(logits_y) == len(labels_y)

        logits_y_u1, _ = apply_fn(y_s1, params)
        logits_rot, _ = apply_fn(y_rot, params, return_rot=True)

        rot_loss = F.cross_entropy(logits_rot, l_rot).mean()
        sup_loss = F.cross_entropy(logits_x, labels_x).mean()
        unsup_mix_loss = F.cross_entropy(logits_y, labels_y).mean()
        unsup_loss = F.cross_entropy(logits_y_u1, ly)

        lambda_y = warmup * self.lambda_y
        lambda_match = warmup * self.lambda_match
        loss = (
            sup_loss
            + self.lambda_rot * rot_loss
            + lambda_y * unsup_loss
            + lambda_match * unsup_mix_loss
        )

        updates = {"model_state": new_model_state, "rng": new_rng, "p_model": p_model}
        scalars = {
            "loss": loss,
            "sup_loss": sup_loss,
            "rot_loss": rot_loss,
            "unsup_loss": unsup_loss,
            "unsup_mix_loss": unsup_mix_loss,
            "acc1": F.accuracy(logits_x, lx),
            "acc5": F.accuracy(logits_x, lx, k=5),
            "rot_acc1": F.accuracy(logits_rot, l_rot),
        }
        return loss, (updates, scalars)


def interleave(xy):
    nu = len(xy) - 1
    xy = [[x[::-1] for x in reversed(jnp.array_split(v[::-1], nu + 1))] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [jnp.concatenate(v) for v in xy]


def rotate(rng, x):
    def f(rng, xi):
        n = jr.randint(rng, (), 0, 4)
        x = rot90(xi, n)
        return x, n

    N = len(x)
    x_rot, l_rot = jax.vmap(f)(jr.split(rng, N), x)
    l_rot = F.one_hot(l_rot, 4)
    return x_rot, l_rot
