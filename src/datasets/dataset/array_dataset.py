from __future__ import annotations
from typing import Any
import tensorflow as tf

import jax.numpy as jnp
from .base import Base
from . import data_utils


class ArrayDataset(Base):

    data_name: str
    num_classes: int
    image_size: int
    mean: Any
    std: Any

    no_flip: bool = False

    train_split: str = "train"
    ulb_split: str | None = None
    test_split: str = "test"

    def __init__(
        self,
        num_labels: int,
        batch_size: int,
        uratio: int,
        test_batch_size: int | None = None,
        include_lb_to_ulb: bool = False,
        shuffle_buffer_size: int = -1,
        seed: int = 12,
    ):
        super().__init__(
            batch_size, uratio, test_batch_size, include_lb_to_ulb, shuffle_buffer_size
        )

        if self.ulb_split is None:
            train_data = data_utils.load_arrays(self.data_name, self.train_split)
            lb_data, ulb_data = data_utils.split_data(train_data, num_labels, seed)
        else:
            lb_data = data_utils.load_arrays(self.data_name, self.train_split)
            lb_data, _ = data_utils.split_data(lb_data, num_labels, seed)
            ulb_data = data_utils.load_arrays(self.data_name, self.ulb_split)
        test_data = data_utils.load_arrays(self.data_name, self.test_split)

        self.lb_data = tf.data.Dataset.from_tensor_slices(lb_data.to_dict())
        self.ulb_data = tf.data.Dataset.from_tensor_slices(ulb_data.to_dict())
        self.test_data = tf.data.Dataset.from_tensor_slices(test_data.to_dict())

        self._data_meta = {
            "num_lb_examples": len(lb_data),
            "num_ulb_examples": len(ulb_data),
            "num_test_examples": len(test_data),
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "mean": jnp.array(self.mean),
            "std": jnp.array(self.std),
            "no_flip": self.no_flip,
            "dist": None,
        }
