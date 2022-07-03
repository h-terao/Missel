import jax.numpy as jnp
import jax.random as jr
import chex
import hub


def get_data(path: str, download: bool) -> hub.Dataset:
    """Returns a hub dataset.

    Args:
        path (str): Dataset path.
        download (bool):
            If True, download the specified dataset in `HUB_DOWNLOAD_PATH`.
    """
    if not download:
        return hub.dataset(path, access_method="stream")
    else:
        try:
            return hub.dataset(path, access_method="local")
        except:  # noqa: E722
            return hub.dataset(path, access_method="download")


def split_data(
    rng: chex.Array, data: hub.Dataset, num_labels: int | float
) -> tuple[hub.Dataset, hub.Dataset]:
    """Split dataset into two sub datasets.

    Args:
        rng: PRNG key.
        data: Dataset to split.
        num_labels (int | float): Number or ratio of labeled samples per label.
        include_lb_to_ulb (bool): If True, unlabeled data also holds samples
            stored in labeled data.
    """
    labels = data["labels"].numpy().flatten()
    unique_labels, counts = jnp.unique(labels, return_counts=True)

    lb_inds, ulb_inds = [], []
    for label, count in zip(unique_labels, counts):
        rng, shuffle_rng = jr.split(rng)
        (index,) = jnp.where(labels == label)
        index = jr.permutation(shuffle_rng, index)

        if num_labels >= 1.0:
            thr = int(num_labels)
        elif num_labels >= 0:
            thr = int(num_labels * count)
        else:
            thr = count

        lb_inds.append(num_labels[:thr])
        ulb_inds.append(num_labels[thr:])

    lb_rng, ulb_rng = jr.split(rng)
    lb_inds = jr.permutation(lb_rng, jnp.concatenate(lb_inds))
    ulb_inds = jr.permutation(ulb_rng, jnp.concatenate(ulb_inds))
    return data[lb_inds.tolist()], data[ulb_inds.tolist()]


def estimate_label_dist(lb_data: hub.Dataset) -> chex.Array:
    """Estimate the label distribution from labeled data.

    Args:
        lb_data (hub.Dataset): Labeled dataset to estimate the label distribution.

    Returns:
        [N] array that an i-th element represents i-th label's ratio.
    """
    labels = lb_data["labels"].numpy().flatten()
    labels = jnp.sort(labels)

    _, counts = jnp.unique(labels, return_counts=True)
    return counts / len(labels)


def get_num_classes(data):
    labels = data["labels"].numpy().flatten()
    unique_labels = jnp.unique(labels)
    return len(unique_labels)
