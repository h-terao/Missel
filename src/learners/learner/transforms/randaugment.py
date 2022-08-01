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
    def shear_x(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = jnp.degrees(jnp.arctan(magnitudes[idx]))
        return T.shear_x(x, v, order=order, mode=mode, cval=cval)

    def shear_y(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = jnp.degrees(jnp.arctan(magnitudes[idx]))
        return T.shear_y(x, v, order=order, mode=mode, cval=cval)

    def translate_x(x: chex.Array, idx: int, magnitudes: chex.Array):
        width = x.shape[1]
        v = width * magnitudes[idx]
        return T.translate_x(x, v, order=order, mode=mode, cval=cval)

    def translate_y(x: chex.Array, idx: int, magnitudes: chex.Array):
        height = x.shape[0]
        v = height * magnitudes[idx]
        return T.translate_y(x, v, order=order, mode=mode, cval=cval)

    def rotate(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return T.rotate(x, v, order=order, mode=mode, cval=cval)

    def brightness(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return T.brightness(x, v)

    def color(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return T.color(x, v)

    def contrast(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return T.contrast(x, v)

    def sharpness(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return T.sharpness(x, v)

    def posterize(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = jnp.int32(magnitudes[idx])
        return T.posterize(x, v)

    def solarize(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return T.solarize(x, v)

    def solarize_add(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return T.solarize_add(x, addition=v)

    def autocontrast(x: chex.Array, idx: int, magnitudes: chex.Array):
        return T.autocontrast(x)

    def equalize(x: chex.Array, idx: int, magnitudes: chex.Array):
        return T.equalize(x)

    def invert(x: chex.Array, idx: int, magnitudes: chex.Array):
        return T.invert(x)

    def identity(x: chex.Array, idx: int, magnitudes: chex.Array):
        return x

    operations = {
        "ShearX": shear_x,
        "ShearY": shear_y,
        "TranslateX": translate_x,
        "TranslateY": translate_y,
        "Rotate": rotate,
        "Brightness": brightness,
        "Color": color,
        "Contrast": contrast,
        "Sharpness": sharpness,
        "Posterize": posterize,
        "Solarize": solarize,
        "SolarizeAdd": solarize_add,
        "AutoContrast": autocontrast,
        "Equalize": equalize,
        "Invert": invert,
        "Identity": identity,
    }

    augment_space = {
        "ShearX": (jnp.linspace(0, 0.3, num_bins), True),
        "ShearY": (jnp.linspace(0, 0.3, num_bins), True),
        "TranslateX": (jnp.linspace(0, 0.3, num_bins), True),
        "TranslateY": (jnp.linspace(0, 0.3, num_bins), True),
        "Rotate": (jnp.linspace(0, 30, num_bins), True),
        "Brightness": (jnp.linspace(0.05, 0.95, num_bins), True),
        "Color": (jnp.linspace(0.05, 0.95, num_bins), True),
        "Contrast": (jnp.linspace(0.05, 0.95, num_bins), True),
        "Sharpness": (jnp.linspace(0.05, 0.95, num_bins), True),
        "Posterize": (8 - jnp.round(jnp.arange(num_bins) / (num_bins - 1) / 4), False),
        "Solarize": (jnp.linspace(1.0, 0.0, num_bins), False),
        "AutoContrast": (jnp.zeros(num_bins), False),
        "Equalize": (jnp.zeros(num_bins), False),
        "Invert": (jnp.zeros(num_bins), False),
        "Identity": (jnp.zeros(num_bins), False),
    }

    branches = []
    for key, (magnitudes, signed) in augment_space.items():
        op = operations[key]
        if signed:
            magnitudes = jnp.concatenate([magnitudes, -magnitudes])
        else:
            magnitudes = jnp.concatenate([magnitudes, magnitudes])
        branches.append(functools.partial(op, magnitudes=magnitudes))

    @jax.jit
    def f(rng: chex.PRNGKey, x: chex.Array) -> chex.Array:
        # Apply crop and flip.
        rng, crop_rng, flip_rng = jr.split(rng, 3)
        x = T.random_crop(crop_rng, x, crop_size, crop_size // 8, mode="reflect")
        if not no_flip:
            x = T.random_hflip(flip_rng, x)

        # Apply RandAugment.
        rng, op_rng, mag_rng = jr.split(rng, 3)
        op_idxs = jr.randint(op_rng, [num_layers], 0, len(branches))
        mag_idxs = jr.randint(mag_rng, [num_layers], 0, 2 * num_bins)
        x, _ = jax.lax.scan(
            lambda carry, idxs: (jax.lax.switch(idxs["op"], branches, carry, idxs["mag"]), idxs),
            x,
            xs={"op": op_idxs, "mag": mag_idxs},
        )

        # Apply cutout.
        if cutout:
            x = T.cutout(rng, x, crop_size // 2)

        return x

    def wrapped(rng: chex.PRNGKey, x: chex.Array) -> chex.Array:
        return jax.vmap(f)(jr.split(rng, len(x)), x)

    return wrapped
