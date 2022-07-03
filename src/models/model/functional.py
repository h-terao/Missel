from __future__ import annotations

import jax.numpy as jnp
from jax import nn
import chex
from hazuchi.functional import one_hot, kl_div, cross_entropy, accuracy

__all__ = [
    "one_hot",
    "kl_div",
    "cross_entropy",
    "accuracy",
    "entropy",
    "absolute_error",
    "squared_error",
]


def entropy(logits: chex.Array) -> chex.Array:
    """Entropy of the specified probability.

    Args:
        logits (Array): Pre-softmax score.

    Returns:
        Entropy of the given logits.
    """
    return -(nn.softmax(logits) * nn.log_softmax(logits)).sum(axis=-1)


def absolute_error(inputs: chex.Array, targets: chex.Array) -> chex.Array:
    """Absolute error."""
    return jnp.absolute(inputs - targets)


def squared_error(inputs: chex.Array, targets: chex.Array) -> chex.Array:
    """Squared error."""
    return jnp.square(inputs - targets)
