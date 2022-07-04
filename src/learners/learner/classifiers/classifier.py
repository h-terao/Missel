from __future__ import annotations
from typing import Any

from flax import linen
from squidink.functional import normalize
import chex


class Classifier(linen.Module):
    """A simple classifier."""

    model: linen.Module
    num_classes: int
    mean: Any
    std: Any

    @linen.compact
    def __call__(self, x: chex.Array, train: bool = False) -> dict[str, chex.Array]:
        x = normalize(x, self.mean, self.std)
        features = self.model(x, train)
        logits = linen.Dense(self.num_classes, dtype=self.model.dtype)(features)
        return {"logits": logits, "features": features}
