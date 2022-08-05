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
    del num_layers, num_bins  # deprecated.

    def shear_x(rng: chex.PRNGKey, x: chex.Array):
        v = jr.uniform(rng, minval=-0.3, maxval=0.3)
        v = jnp.degrees(jnp.arctan(v))
        return T.shear_x(x, v, order=order, mode=mode, cval=cval)

    def shear_y(rng: chex.PRNGKey, x: chex.Array):
        v = jr.uniform(rng, minval=-0.3, maxval=0.3)
        v = jnp.degrees(jnp.arctan(v))
        return T.shear_y(x, v, order=order, mode=mode, cval=cval)

    def translate_x(rng: chex.PRNGKey, x: chex.Array):
        v = crop_size * jr.uniform(rng, minval=-0.3, maxval=0.3)
        return T.translate_x(x, v, order=order, mode=mode, cval=cval)

    def translate_y(rng: chex.PRNGKey, x: chex.Array):
        v = crop_size * jr.uniform(rng, minval=-0.3, maxval=0.3)
        return T.translate_y(x, v, order=order, mode=mode, cval=cval)

    def rotate(rng: chex.PRNGKey, x: chex.Array):
        v = crop_size * jr.uniform(rng, minval=-30, maxval=30)
        return T.translate_y(x, v, order=order, mode=mode, cval=cval)

    def brightness(rng: chex.PRNGKey, x: chex.Array):
        v = jr.uniform(rng, minval=0.05, maxval=0.95)
        return T.brightness(x, v)

    def color(rng: chex.PRNGKey, x: chex.Array):
        v = jr.uniform(rng, minval=0.05, maxval=0.95)
        return T.color(x, v)

    def contrast(rng: chex.PRNGKey, x: chex.Array):
        v = jr.uniform(rng, minval=0.05, maxval=0.95)
        return T.contrast(x, v)

    def sharpness(rng: chex.PRNGKey, x: chex.Array):
        v = jr.uniform(rng, minval=0.05, maxval=0.95)
        return T.sharpness(x, v)

    def posterize(rng: chex.PRNGKey, x: chex.Array):
        v = jr.uniform(rng)
        bits = 4 + (4 * v).astype(jnp.int32)
        return T.posterize(x, bits)

    def solarize(rng: chex.PRNGKey, x: chex.Array):
        v = jr.uniform(rng)
        return T.solarize(x, v)

    def autocontrast(rng: chex.PRNGKey, x: chex.Array):
        return T.autocontrast(x)

    def equalize(rng: chex.PRNGKey, x: chex.Array):
        return T.equalize(x)

    def invert(rng: chex.PRNGKey, x: chex.Array):
        return T.invert(x)

    def identity(rng: chex.PRNGKey, x: chex.Array):
        return x
