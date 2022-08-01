from __future__ import annotations
import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import chex
from squidink import functional as T


def randaugment(
    crop_size: int,
    no_flip: bool,
    num_layers: int = 2,
    num_bins: int = 10,
    cutout: bool = False,
    order: int = 0,
    mode: str = "constant",
    cval: float = 0.5,
):
    """TorchSSL ver RandAugment."""

    def sample_v(rng: chex.PRNGKey, min_v: float, max_v: float, negate: bool) -> chex.Array:
        v_rng, neg_rng = jr.split(rng)
        v = jr.uniform(key=v_rng, minval=min_v, maxval=max_v)
        if negate:
            v = jnp.where(jr.uniform(neg_rng) < 0.5, -v, v)
        return v

    def shear_x(
        rng: chex.PRNGKey, x: chex.Array, min_v: float = 0, max_v: float = 0.3, negate: bool = True
    ) -> chex.Array:
        v = sample_v(rng, min_v, max_v, negate)
