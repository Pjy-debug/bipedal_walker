# The following code is from openai/evolution-strategies-starter
# (https://github.com/openai/evolution-strategies-starter)
# under the MIT License.


import numpy as np


def compute_ranks(x: np.ndarray):
    """
    Returns ranks in [0, len(x)-1]
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

# modified to avoid divided by zero
def compute_centered_ranks(x: np.ndarray):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    size = x.size
    if size == 1:
        y = np.array([0]).astype(np.float32)
    else:
        y /= size-1
        y -= .5
    return y

def compute_centered_ranks_origin(x: np.ndarray):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= x.size
    # y /= (x.size - 1)
    y -= .5
    return y


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32),
                        np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed
