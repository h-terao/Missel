from __future__ import annotations

from flax import linen
import chex

from .classifier import Classifier


class RotClassifier(Classifier):
    """A classifer that predicts labels and rotation degrees."""

    @linen.compact
    def __call__(self, x: chex.Array, train: bool = False) -> dict[str, chex.Array]:
        x = super().__call__(x, train)
        x["logits_rot"] = linen.Dense(4, dtype=self.model.dtype)(x["features"])
        return x
