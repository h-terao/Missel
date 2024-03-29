from __future__ import annotations
from typing import Any, Callable
import functools
import math

from jax import nn
import jax.numpy as jnp
from flax import linen
import chex
from einops import reduce

ModuleDef = Any


class BasicBlock(linen.Module):

    conv: ModuleDef
    norm: ModuleDef
    act: Callable

    num_filters: int
    stride: int = 1
    activate_before_residual: bool = False

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        if x.shape[-1] != self.num_filters and self.activate_before_residual:
            x = self.act(self.norm()(x))
        else:
            out = self.act(self.norm()(x))
        if x.shape[-1] != self.num_filters:
            out = x
        out = self.conv(self.num_filters, (3, 3), self.stride)(out)
        out = self.act(self.norm()(out))
        out = self.conv(self.num_filters, (3, 3))(out)
        if x.shape != out.shape:
            x = self.conv(self.num_filters, (1, 1), self.stride)(x)
        return out + x


class WideResNet(linen.Module):
    stage_sizes: list[int]
    widen_factor: int
    precision: str

    @linen.compact
    def __call__(self, x: chex.Array, train: bool = False) -> chex.Array:
        first_stride = x.shape[-2] > 128

        conv = functools.partial(
            linen.Conv,
            kernel_init=nn.initializers.variance_scaling(
                math.sqrt(2 / (1 + 0.01)),
                "fan_out",
                "normal",
            ),
            dtype=self.dtype,
        )
        norm = functools.partial(
            linen.BatchNorm,
            use_running_average=not train,
            momentum=0.999,
            epsilon=0.001,
            dtype=self.bn_dtype,
            axis_name="batch",
        )
        act = functools.partial(nn.leaky_relu, negative_slope=0.1)

        x = conv(16, (3, 3))(x)
        for i, block_size in enumerate(self.stage_sizes):
            num_filters = self.widen_factor * 16 * (2**i)
            for j in range(block_size):
                stride = 2 if j == 0 and (i > 0 or first_stride) else 1
                activate_before_residual = i == 0 and j == 0
                x = BasicBlock(
                    conv,
                    norm,
                    act,
                    num_filters=num_filters,
                    stride=stride,
                    activate_before_residual=activate_before_residual,
                )(x)
        x = act(norm()(x))
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
