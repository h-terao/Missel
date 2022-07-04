from __future__ import annotations
from typing import Any
import functools

from jax import nn
import jax.numpy as jnp
from flax import linen
import chex
from einops import reduce


ModuleDef = Any


class ResBlock(linen.Module):

    conv: ModuleDef
    norm: ModuleDef

    num_filters: int
    stride: int = 1

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        h = self.conv(self.num_filters, (3, 3), self.stride)(x)
        h = self.norm()(h)
        h = nn.relu(h)

        h = self.conv(self.num_filters, (3, 3))(h)
        h = self.norm()(h)

        if x.shape != h.shape:
            x = self.conv(self.num_filters, (1, 1), self.stride)(x)
            x = self.norm()(x)

        return nn.relu(h + x)


class ResBottleneck(linen.Module):

    conv: ModuleDef
    norm: ModuleDef

    num_filters: int
    stride: int = 1

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        h = self.conv(self.num_filters, (1, 1))(x)
        h = self.norm()(h)
        h = nn.relu(h)

        h = self.conv(self.num_filters, (3, 3), self.stride)(h)
        h = self.norm()(h)
        h = nn.relu(h)

        h = self.conv(self.num_filters * 4, (1, 1))(h)
        h = self.norm()(h)

        if x.shape != h.shape:
            x = self.conv(self.num_filters * 4, (1, 1), self.stride)(x)
            x = self.norm()(x)

        return nn.relu(h + x)


class ResNet(linen.Module):
    block_type: str
    stage_sizes: list[int]
    precision: str

    @linen.compact
    def __call__(self, x: chex.Array, train: bool = False) -> chex.Array:
        assert self.block_type in ["block", "bottleneck"]
        if self.block_type == "block":
            block_cls = ResBlock
        else:
            block_cls = ResBottleneck

        conv = functools.partial(linen.Conv, dtype=self.dtype)
        norm = functools.partial(
            linen.BatchNorm,
            use_running_average=not train,
            momentum=0.999,
            epsilon=0.001,
            dtype=self.bn_dtype,
            axis_name="batch",
        )

        x = conv(64, (7, 7), 2)(x)
        x = norm()(x)
        x = nn.relu(x)
        x = linen.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            num_filters = 64 * (2**i)
            for j in range(block_size):
                stride = 2 if j == 0 and i > 0 else 1
                x = block_cls(
                    conv,
                    norm,
                    num_filters=num_filters,
                    stride=stride,
                )(x)

        x = reduce(x, "B H W C -> B C", "mean")
        return x

    @property
    def dtype(self) -> chex.ArrayDType:
        assert self.precision in ["fp16", "fp32", "bf16"]
        if self.precision == "bf16":
            return jnp.bfloat16
        elif self.precision == "fp16":
            return jnp.float16
        else:
            return jnp.float32

    @property
    def bn_dtype(self) -> chex.ArrayDType:
        assert self.precision in ["fp16", "fp32", "bf16"]
        if self.precision == "bf16":
            return jnp.bfloat16
        else:
            return jnp.float32
