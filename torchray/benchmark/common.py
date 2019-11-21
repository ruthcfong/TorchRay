import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn as nn


import torch


def compute_similarity(x, y, similarity_metric="spearman_no_abs", eps=1e-6):
    """Compute similarity score between two tensors batch-wise.

    Args:
        x (:class:`torch.Tensor`): tensor to compare batch-wise with :attr:`y`.
        y (:class:`torch.Tensor`): tensor to compare batch-wise with :attr:`x`.
        similarity_metric (str): name of similarity metric to use.
            Default: ``"spearman_no_abs"``.

    Returns:
        (corrs, ps): tuple containing the following:
            * numpy array of correlation scores
            * numpy array of p-values
    """
    x = x + eps
    y = y + eps
    if isinstance(x, torch.Tensor):
        x = x.cpu().data.numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().data.numpy()

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.shape == y.shape
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    assert len(x.shape) >= 2

    bs = x.shape[0]
    corrs = np.zeros(bs)
    ps = np.zeros(bs)

    for i in range(bs):
        if "spearman" in similarity_metric:
            if "no_abs" in similarity_metric:
                corr, p = spearmanr(x[i].reshape(-1),
                                    y[i].reshape(-1))
            else:
                corr, p = spearmanr(np.abs(x[i]).reshape(-1),
                                    np.abs(y[i]).reshape(-1))
            corrs[i] = corr
            ps[i] = p
        else:
            assert False

    if bs == 1:
        return corrs[0], ps[0]
    else:
        return corrs, ps
