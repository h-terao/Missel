"""SGD w/ warmup and cosine decay schedule."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import chex


def sgd(
    train_steps: int,
    lr: float,
    weight_decay: float = 1e-5,
    momentum: float = 0.9,
    nesterov: bool = True,
    warmup_steps: int = 0,
    num_cycles: float = 7.0 / 16.0,
) -> optax.GradientTransformation:
    def scheduled_lr(step: int) -> float:
        warmup_factor = step / warmup_steps
        cosine_steps = step - warmup_steps
        cosine_steps /= train_steps - warmup_steps
        decay_factor = jnp.cos(num_cycles * cosine_steps * jnp.pi)
        return lr * jnp.where(step < warmup_steps, warmup_factor, decay_factor)

    def mask_fn(tree: chex.PyTreeDef) -> chex.PyTreeDef:
        return jax.tree_map(lambda x: x.ndim > 1, tree)

    return optax.chain(
        optax.additive_weight_decay(weight_decay, mask=mask_fn),
        optax.sgd(scheduled_lr, momentum=momentum, nesterov=nesterov),
    )
