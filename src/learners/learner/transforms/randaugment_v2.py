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
    augment_space = {
        "ShearX": (jnp.linspace(0, 0.3, num_bins), True),
        "ShearY": (jnp.linspace(0, 0.3, num_bins), True),
        "TranslateX": (jnp.linspace(0, 150.0 / 331.0, num_bins), True),
        "TranslateY": (jnp.linspace(0, 150.0 / 331.0, num_bins), True),
        "Rotate": (jnp.linspace(0, 30, num_bins), True),
        "Brightness": (jnp.linspace(0, 0.9, num_bins), True),
        "Color": (jnp.linspace(0, 0.9, num_bins), True),
        "Contrast": (jnp.linspace(0, 0.9, num_bins), True),
        "Sharpness": (jnp.linspace(0, 0.9, num_bins), True),
        "Posterize": (8 - jnp.round(jnp.arange(num_bins) / (num_bins - 1) / 4), False),
        "Solarize": (jnp.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (jnp.zeros(num_bins), False),
        "Equalize": (jnp.zeros(num_bins), False),
        "Invert": (jnp.zeros(num_bins), False),
        "Identity": (jnp.zeros(num_bins), False),
    }

    augment_space = {
        "AutoContrast": (0, 1),
        "Brightness": ()
        "ShearX": (-0.3, 0.3),
        "ShearY": (-0.3, 0.3),
        "TranslateX": (-0.3, 0.3),
        "TranslateY": (-0.3, 0.3),
        "Solarize": (0, 1.0),
    }