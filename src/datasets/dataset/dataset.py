from __future__ import annotations
import os
from typing import Callable
import math

import jax.random as jr
import tensorflow as tf

from . import data_utils


class Dataset:
    """A base class of datasets.

    Args:
        rng: PRNG key.
        num_lables: Number of labels per class.
        batch_size: Batch size of labeled data.
        uratio: Unlabeled ratio.
    """

    name: str
    lb_path: str
    test_path: str
    data_meta: dict

    ulb_path: str | None = None
    preprocess: Callable | None = None

    def __init__(
        self,
        data_dir,
        num_labels: int,
        batch_size: int,
        uratio: int,
        test_batch_size: int | None = None,
        include_lb_to_ulb: bool = True,
        shuffle_batches: int = -1,
        seed: int = 0,
        cache: bool = False,
        download: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.lb_batch_size = batch_size
        self.ulb_batch_size = batch_size * uratio
        self.test_batch_size = test_batch_size or self.ulb_batch_size
        self.include_lb_to_ulb = include_lb_to_ulb
        self.shuffle_batches = shuffle_batches if shuffle_batches > 0 else math.inf
        self.cache = cache

        self.cache_dir = os.path.join(data_dir, "caches", f"{self.name}.{num_labels}.{seed}")
        os.makedirs(self.cache_dir, exist_ok=True)

        if self.ulb_path is None:
            train_data = data_utils.get_data(self.lb_path, download)
            lb_data, ulb_data = data_utils.split_data(
                jr.PRNGKey(seed), train_data, num_labels, include_lb_to_ulb
            )
        else:
            lb_data = data_utils.get_data(self.lb_path, download)
            lb_data, _ = data_utils.split_data(jr.PRNGKey(seed), lb_data, num_labels)
            ulb_data = data_utils.get_data(self.ulb_path, download)

        test_data = data_utils.get_data(self.test_path, download)

        self.lb_data = lb_data
        self.ulb_data = ulb_data
        self.test_data = test_data

        # Add data info.
        self.data_meta["lb_examples"] = len(lb_data)
        self.data_meta["lb_batch_size"] = self.lb_batch_size
        self.data_meta["ulb_examples"] = len(ulb_data)
        if include_lb_to_ulb:
            self.data_meta["ulb_examples"] += len(lb_data)
        self.data_meta["ulb_batch_size"] = self.ulb_batch_size
        self.data_meta["test_examples"] = len(test_data)
        self.data_meta["test_batch_size"] = self.test_batch_size
        self.data_meta["dist"] = data_utils.estimate_label_dist(lb_data)
        self.data_meta["num_classes"] = data_utils.get_num_classes(test_data)

    def map_fn(self, index, item):
        images = item["images"]
        labels = item["labels"]

        images = self.preprocess(images)
        labels = tf.reshape(labels, shape=[-1])
        return {
            "indices": index,
            "images": images,
            "labels": labels,
        }

    def train_loader(self):
        num_lb_examples = len(self.lb_data)
        num_ulb_examples = len(self.ulb_data)

        lb_data: tf.data.Dataset = self.lb_data.tensorflow()
        ulb_data: tf.data.Dataset = self.ulb_data.tensorflow()
        if self.include_lb_to_ulb:
            ulb_data = lb_data.concatenate(ulb_data)
            num_ulb_examples += num_lb_examples

        lb_data = lb_data.enumerate()
        lb_data = lb_data.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if self.cache:
            lb_data = lb_data.cache(os.path.join(self.cache_dir, "lb.cache"))
        lb_data = lb_data.repeat().shuffle(
            min(self.shuffle_batches * self.lb_batch_size, num_lb_examples)
        )
        lb_data = lb_data.batch(
            self.lb_batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        ulb_data = ulb_data.enumerate()
        ulb_data = ulb_data.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if self.cache:
            ulb_data = ulb_data.cache(os.path.join(self.cache_dir, "ulb.cache"))
        ulb_data = ulb_data.repeat().shuffle(
            min(self.shuffle_batches * self.ulb_batch_size, num_ulb_examples)
        )
        ulb_data = ulb_data.batch(
            self.ulb_batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        train_data = tf.data.Dataset.zip({"labeled": lb_data, "unlabeled": ulb_data})
        return train_data.as_numpy_iterator()

    def test_loader(self):
        test_data: tf.data.Dataset = self.test_data.tensorflow()
        test_data = test_data.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        test_data = test_data.batch(self.test_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        if self.cache:
            test_data = test_data.cache(os.path.join(self.cache_dir, "test.cache"))
        return test_data.as_numpy_iterator()
