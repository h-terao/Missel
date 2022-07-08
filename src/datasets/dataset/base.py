from __future__ import annotations
from copy import deepcopy
import math
import tensorflow as tf


class Base:

    data_meta: dict
    lb_data: tf.data.Dataset
    ulb_data: tf.data.Dataset
    test_data: tf.data.Dataset
    _data_meta: dict

    def __init__(
        self,
        batch_size: int,
        uratio: int,
        test_batch_size: int | None = None,
        include_lb_to_ulb: bool = False,
        shuffle_buffer_size: int = -1,
    ):
        self.lb_batch_size = batch_size
        self.ulb_batch_size = batch_size * uratio
        self.test_batch_size = test_batch_size or batch_size
        self.include_lb_to_ulb = include_lb_to_ulb
        self.shuffle_buffer_size = shuffle_buffer_size if shuffle_buffer_size > 0 else math.inf

    def train_loader(self):
        lb_data, ulb_data = self.lb_data, self.ulb_data
        num_lb_examples = self.data_meta["num_lb_examples"]
        num_ulb_examples = self.data_meta["num_ulb_examples"]
        if self.include_lb_to_ulb:
            ulb_data = lb_data.concatenate(ulb_data)
            num_ulb_examples += num_lb_examples
        lb_data = (
            lb_data.repeat()
            .shuffle(min(self.shuffle_buffer_size, num_lb_examples))
            .batch(self.lb_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        )
        ulb_data = (
            ulb_data.repeat()
            .shuffle(min(self.shuffle_buffer_size, num_ulb_examples))
            .batch(self.ulb_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        )
        train_data = tf.data.Dataset.zip({"labeled": lb_data, "unlabeled": ulb_data})
        return train_data.as_numpy_iterator()

    def test_loader(self):
        return (
            self.test_data.batch(self.test_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .repeat()
            .as_numpy_iterator()
        )

    @property
    def data_meta(self):
        data_meta = deepcopy(self._data_meta)
        if self.include_lb_to_ulb:
            data_meta["num_ulb_examples"] += data_meta["num_lb_examples"]
        data_meta["test_steps_per_epoch"] = math.ceil(
            data_meta["num_test_examples"] / self.test_batch_size
        )
        return data_meta
