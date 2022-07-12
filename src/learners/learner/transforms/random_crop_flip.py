from __future__ import annotations
from typing import Callable

import jax
import jax.random as jr
from squidink.functional import random_crop, random_hflip
import chex


def random_crop_flip(
    crop_size: int, no_flip: bool = False
) -> Callable[[chex.PRNGKey, chex.Array], chex.Array]:
    """Returns transform function.

    Args:
        crop_size (int): Desired size of outputs.
        no_flip (bool): If True, apply horizontal flipping to inputs
            with 50% probabilities.
    """
    pad_size = crop_size // 8

    @jax.jit
    def f(rng: chex.PRNGKey, x: chex.Array) -> chex.Array:
        crop_rng, flip_rng = jr.split(rng, 2)
        x = random_crop(crop_rng, x, crop_size, pad_size, mode="reflect")
        if not no_flip:
            x = random_hflip(flip_rng, x)
        return x

    def wrapped(rng: chex.PRNGKey, x: chex.Array) -> chex.Array:
        return jax.vmap(f)(jr.split(rng, len(x)), x)

    return wrapped
