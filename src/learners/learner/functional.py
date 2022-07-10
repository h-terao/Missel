from __future__ import annotations

import jax.numpy as jnp
from jax import nn
import chex
from hazuchi.functional import one_hot, cross_entropy, accuracy, permutate

__all__ = [
    "one_hot",
    "cross_entropy",
    "accuracy",
    "entropy",
    "absolute_error",
    "squared_error",
    "permutate",
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


def kl_div(q_logit, p_logit):
    q = nn.softmax(q_logit)
    logq = nn.log_softmax(q_logit)
    logp = nn.log_softmax(p_logit)

    qlogq = (q * logq).sum(axis=-1)
    qlogp = (q * logp).sum(axis=-1)
    return qlogq - qlogp
