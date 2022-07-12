from __future__ import annotations
from typing import Any, Dict, Type
import functools

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen, struct, core
from flax.optim.dynamic_scale import DynamicScale
from squidink.functional import center_crop
import optax
import chex

from .train_state import TrainState
from .classifiers import Classifier
from . import functional as F

Scalars = Dict[str, chex.Array]
Updates = Dict[str, Any]
Batch = Dict[str, chex.Array]


class Learner:
    """Abstract class to implement SSL methods.

    Attribute:
        train_state_cls: TrainState class.
        classifier_cls: Classifier class.
        default_entries: Entries to log on console.
        always_updates: Parameters to update everytime
            even when infinite grads are found. (Only used when precision=fp16.)
        model: Flax model.
    """

    train_state_cls: Type[struct.PyTreeNode] = TrainState
    classifier_cls: Type[linen.Module] = Classifier
    default_entries: list[str] = []
    always_updates: list[str] = []

    def __init__(
        self,
        data_meta: dict,
        train_steps: int,
        base_model: linen.Module,
        tx: optax.GradientTransformation,
        label_smoothing: float = 0.0,
        momentum_ema: float = 0.999,
        precision: str = "fp32",
    ) -> None:
        assert precision in ["fp16", "fp32", "bf16"]
        self.data_meta = data_meta
        self.train_steps = train_steps
        self.base_model = base_model
        self.tx = tx
        self.label_smoothing = label_smoothing
        self.momentum_ema = momentum_ema
        self.precision = precision

    def loss_fn(
        self, params: core.FrozenDict, train_state: TrainState, batch: Batch
    ) -> tuple[chex.Array, tuple[Updates, Scalars]]:
        raise NotImplementedError("Implement loss_fn!")

    @property
    def model(self) -> linen.Module:
        """Returns flax model."""
        return self.classifier_cls(
            self.base_model,
            num_classes=self.data_meta["num_classes"],
            mean=self.data_meta["mean"],
            std=self.data_meta["std"],
        )

    def init_fn(self, rng: chex.PRNGKey, batch: Batch, **kwargs) -> TrainState:
        """Initialize train_state."""
        param_rng, state_rng = jr.split(rng)
        # model = self.classifier_cls(
        #     self.base_model,
        #     num_classes=self.data_meta["num_classes"],
        #     mean=self.data_meta["mean"],
        #     std=self.data_meta["std"],
        # )

        @functools.partial(jax.jit, backend="cpu")
        def initialize(rng, batch):
            x = batch["labeled"]["inputs"] / 255.0
            variables = self.model.init({"params": rng}, x, train=True)
            model_state, params = variables.pop("params")
            return params, model_state

        if self.precision == "fp16":
            dynamic_scale = DynamicScale()
        else:
            dynamic_scale = None

        params, model_state = initialize(param_rng, batch)
        train_state = self.train_state_cls.create(
            rng=state_rng,
            apply_fn=self.model.apply,
            params=params,
            model_state=model_state,
            tx=self.tx,
            momentum_ema=self.momentum_ema,
            dynamic_scale=dynamic_scale,
            **kwargs,
        )

        # TODO: Add model information such as GFLOPs, #params, ...
        model_info = dict()

        return train_state, model_info

    def train_fn(self, train_state: TrainState, batch: Batch) -> tuple[TrainState, Scalars]:
        """Update training state by the given mini-batch data.

        Args:
            train_state (TrainState): Training state.
            batch (Batch): Batched data to evaluate the training model.

        Returns:
            Updated training state and metrics for logging.
        """
        dynamic_scale = train_state.dynamic_scale

        is_fin = True
        if dynamic_scale:
            grad_fn = dynamic_scale.value_and_grad(self.loss_fn, has_aux=True, axis_name="batch")
            dynamic_scale, is_fin, (_, (updates, scalars)), grads = grad_fn(
                train_state.params, train_state, batch
            )
            scalars["scale"] = dynamic_scale.scale
        else:
            grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
            (_, (updates, scalars)), grads = grad_fn(train_state.params, train_state, batch)
            grads = jax.lax.pmean(grads, axis_name="batch")

        new_train_state = train_state.apply_gradients(
            grads=grads,
            dynamic_scale=dynamic_scale,
            **updates,
        )

        if dynamic_scale:
            # Skip to update most variables in train_state if some gradients are infinite.
            # step, rng and **always_updates are always updated.
            f = functools.partial(jnp.where, is_fin)
            overrides = dict(
                params=jax.tree_map(f, new_train_state.params, train_state.params),
                params_ema=jax.tree_map(f, new_train_state.params_ema, train_state.params_ema),
                opt_state=jax.tree_map(f, new_train_state.opt_state, train_state.opt_state),
                **{
                    k: jax.tree_map(f, getattr(new_train_state, k), getattr(train_state, k))
                    for k in updates
                    if k not in self.always_updates
                },
            )
            new_train_state = new_train_state.replace(**overrides)

        return new_train_state, scalars

    def test_fn(self, train_state: TrainState, batch: Batch) -> Scalars:
        """Test function to evaluate the training model.

        Args:
            train_state (TrainState): Training state.
            batch (Batch): Batched data to evaluate the training model.

        Returns:
            The standard classification metrics (i.e., cross-entropy, top-1 and top-5 accuracy).
        """
        x = center_crop(
            batch["inputs"] / 255.0,
            crop_size=self.data_meta["image_size"],
            pad_size=self.data_meta["image_size"] // 8,
            mode="reflect",
        )
        lx = F.one_hot(batch["labels"], self.data_meta["num_classes"], self.label_smoothing)

        def compute_scalars(params: core.FrozenDict, prefix: str = None) -> Scalars:
            prefix = prefix or ""
            variables = {"params": params, **train_state.model_state}
            logits = train_state.apply_fn(variables, x, train=False)["logits"]
            scalars = {
                f"{prefix}loss": F.cross_entropy(logits, lx),
                f"{prefix}acc1": F.accuracy(logits, lx, k=1),
                f"{prefix}acc5": F.accuracy(logits, lx, k=5),
            }
            return scalars

        scalars = compute_scalars(train_state.params)
        scalars.update(compute_scalars(train_state.params_ema, "EMA/"))
        return scalars
