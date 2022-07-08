from __future__ import annotations
from typing import NamedTuple

import numpy as np
import tensorflow_datasets as tfds


class Arrays(NamedTuple):
    images: np.ndarray
    labels: np.ndarray
    indices: np.ndarray

    def __getitem__(self, index):
        return Arrays(self.images[index], self.labels[index], self.indices[index])

    def __len__(self) -> int:
        return len(self.images)

    def to_dict(self):
        return {
            "images": self.images,
            "labels": self.labels,
            "indices": self.indices,
        }


def load_arrays(data_name, split: str):
    arrays = tfds.as_numpy(tfds.load(data_name, split=split, batch_size=1, as_supervised=True))
    if len(arrays) == 1:
        images = arrays[0]
        labels = np.zeros(len(images))
    else:
        print("LEN ARRAYS:", len(arrays))
        images, labels = arrays
    indices = np.arange(len(labels))
    return images, labels, indices


def split_data(arrays: Arrays, num_labels: int | float, seed: int = 0):
    rng = np.random.RandomState(seed)
    labels = arrays.labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)
    lb_inds, ulb_inds = [], []
    for label, count in zip(unique_labels, counts):
        index = np.where(labels == label)[0]
        index = rng.permutation(index)
        if num_labels >= 1.0:
            thr = int(num_labels // num_classes)
        elif num_labels >= 0:
            thr = int(num_labels * count)
        else:
            thr = -1
        lb_inds.append(index[:thr])
        ulb_inds.append(index[thr:])
    lb_inds = rng.permutation(np.concatenate(lb_inds))
    ulb_inds = rng.permutation(np.concatenate(ulb_inds))
    return arrays[lb_inds], arrays[ulb_inds]
