from __future__ import annotations
from typing import Any, Callable

import jax
from flax import struct, core
from flax.optim.dynamic_scale import DynamicScale

import optax
import chex


class TrainState(struct.PyTreeNode):
    step: int
    rng: chex.PRNGKey
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    params_ema: core.FrozenDict[str, Any]
    model_state: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    dynamic_scale: DynamicScale | None = None
    momentum_ema: float = 0.999

    def apply_gradients(self, *, grads, **kwargs) -> TrainState:
        """
        Args:
            grads (FrozenDict): Gradients.
            is_fin (bool): If False, gradients contain Inf/NaNs.
                Only required if you use float16 precision.
            **kwargs: Other new variables to overwrite.

        Returns:
            Updated train state.
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_params_ema = jax.tree_map(
            lambda p_ema, p: self.momentum_ema * (p_ema - p) + p,
            self.params_ema,
            new_params,
        )
        return self.replace(
            step=self.step + 1,
            params=new_params,
            params_ema=new_params_ema,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(
        cls, *, rng, apply_fn, params, model_state, tx, momentum_ema: float = 0.999, **kwargs
    ) -> TrainState:
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            rng=rng,
            apply_fn=apply_fn,
            params=params,
            params_ema=params,
            model_state=model_state,
            tx=tx,
            opt_state=opt_state,
            momentum_ema=momentum_ema,
            **kwargs,
        )
