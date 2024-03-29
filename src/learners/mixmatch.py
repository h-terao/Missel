from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen, core
import optax

from .learner import Learner, TrainState, functional as F, transforms as T

Batch = Any


class MixMatch(Learner):
    """MixMatch learner.

    Args:
        data_meta: Meta information of the dataset.
        train_steps (int): Total number of steps for training.
        base_model (Module): Base model.
        tx (GradientTransformation): Optax optimizer.
        label_smoothing (float): Label smoothing parameter.
        momentum_ema (float): Momentum value to update EMA model.
        precision (str): Precision. fp16, bf16 and fp32 are supported.
        lambda_y (float): Coefficient of the unsupervised loss.
        unsup_warmup_pos (float): Warmup position of epochs.
    """

    default_entries: list[str] = ["warmup", "loss", "sup_loss", "unsup_loss", "acc1"]

    def __init__(
        self,
        data_meta: dict,
        train_steps: int,
        base_model: linen.Module,
        tx: optax.GradientTransformation,
        lambda_y: float = 100,
        T: float = 0.5,
        alpha: float = 0.5,
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
        self.T = T
        self.alpha = alpha

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

        rng, x_rng, y1_rng, y2_rng = jr.split(rng, 4)
        x = transform(x_rng, batch["labeled"]["inputs"] / 255.0)
        y1 = transform(y1_rng, batch["unlabeled"]["inputs"] / 255.0)
        y2 = transform(y2_rng, batch["unlabeled"]["inputs"] / 255.0)
        lx = F.one_hot(
            batch["labeled"]["labels"], self.data_meta["num_classes"], self.label_smoothing
        )

        # guess labels.
        logits_y1 = apply_fn(y1, train_state.params)[0]
        logits_y2 = apply_fn(y2, train_state.params)[0]

        # average
        ly = (linen.softmax(logits_y1) + linen.softmax(logits_y2)) / 2
        ly /= jnp.sum(ly, axis=-1, keepdims=True)

        # sharpening
        ly = ly ** (1 / self.T)
        ly /= jnp.sum(ly, axis=-1, keepdims=True)

        # mixup
        inputs = jnp.concatenate([x, y1, y2])
        labels = jnp.concatenate([lx, ly, ly])

        index_rng, mixup_ratio_rng = jr.split(rng)
        index = jr.permutation(index_rng, len(inputs))
        lam = jr.beta(mixup_ratio_rng, self.alpha, self.alpha)
        lam = jnp.maximum(lam, 1 - lam)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        mixed_inputs = jnp.array_split(mixed_inputs, len(inputs) // len(x))
        mixed_inputs = interleave(mixed_inputs)

        logits, new_model_state = apply_fn(mixed_inputs[0], params)
        logits = [logits] + [apply_fn(v, params)[0] for v in mixed_inputs[1:]]
        logits = interleave(logits)

        logits_x, logits_y = logits[0], jnp.concatenate(logits[1:])
        labels_x, labels_y = mixed_labels[: len(x)], mixed_labels[len(x) :]
        assert len(logits_x) == len(labels_x)
        assert len(logits_y) == len(labels_y)

        sup_loss = F.cross_entropy(logits_x, labels_x).mean()
        unsup_loss = F.squared_error(linen.softmax(logits_y), labels_y).mean()

        loss = sup_loss + self.lambda_y * warmup * unsup_loss
        updates = {"model_state": new_model_state, "rng": new_rng}

        logits, _ = apply_fn(x, train_state.params)
        scalars = {
            "loss": loss,
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "warmup": warmup,
            "acc1": F.accuracy(logits, lx),
            "acc5": F.accuracy(logits, lx, k=5),
        }

        return loss, (updates, scalars)


def interleave(xy):
    nu = len(xy) - 1
    xy = [[x[::-1] for x in reversed(jnp.array_split(v[::-1], nu + 1))] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [jnp.concatenate(v) for v in xy]


# def interleave_offsets(batch, nu):
#     groups = [batch // (nu + 1)] * (nu + 1)
#     for x in range(batch - sum(groups)):
#         groups[-x - 1] += 1
#     offsets = [0]
#     for g in groups:
#         offsets.append(offsets[-1] + g)
#     assert offsets[-1] == batch
#     return offsets


# def interleave(xy, batch):
#     nu = len(xy) - 1
#     offsets = interleave_offsets(batch, nu)
#     xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
#     for i in range(1, nu + 1):
#         xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
#     return [tf.concat(v, axis=0) for v in xy]
