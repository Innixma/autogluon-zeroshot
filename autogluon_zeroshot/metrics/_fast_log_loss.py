from typing import Tuple, Optional

import numpy as np

from autogluon.core.metrics import make_scorer


def convert_multi_to_binary_fast_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[None, np.ndarray]:
    """
    Converts standard multiclass classification prediction probabilities into a proxy binary format
    to accelerate log_loss computation.

    The output can be passed into `_fast_log_loss`.

    Note: Data must be pre-normalized. The data is not normalized within this function for speed purposes.
    """
    ndim = y_pred.ndim
    if ndim == 1:
        raise AssertionError(f'Binary classification not yet implemented for fast_log_loss (it is possible to add)')
    elif ndim == 2:
        y_pred = y_pred[range(y_pred.shape[0]), y_true]
    elif ndim == 3:
        """Convert multiple model's prediction probabilities at once when stacked in a 3-dimensional numpy array"""
        y_pred = y_pred[:, range(y_pred.shape[1]), y_true]
    else:
        raise AssertionError(f'ndim={ndim} not supported.')

    # This is what y_true is treated as, but no need to perform operation
    # y_true = np.ones(y_true.shape, dtype=np.uint8)

    return None, y_pred


def _fast_log_loss(y_true: Optional[np.ndarray], y_pred: np.ndarray) -> float:
    """
    Heavily optimized log_loss implementation that is valid under a specific context and avoids all sanity checks.
    This is >100x faster than sklearn.

    NOTE: You must first preprocess the input y_pred by calling `convert_multi_to_binary_fast_log_loss`.

    1. There is no epsilon / value clipping, ensure y_pred ranges do not include `0` or `1` to avoid infinite loss.
    2. `y_true` is ignored. It is assumed to be in the form `np.ones(len(y_pred), dtype=np.uint8)`
      By assuming this form, we can avoid unnecessary computation.
    3. `y_pred` must be formatted as a 1-dimensional ndarray.
      The values are the prediction probability assigned to the ground truth class.
      All other classes that are not the ground truth are ignored, as they are not necessary to calculate log_loss.
    4. There is no support for sample weights.

    Parameters
    ----------
    y_true : Unused
        Ignored. y_true is assumed to be `1` for every row.

    y_pred : array-like of float
        The prediction probabilities of the ground truth class. shape = (n_samples,)

    Returns
    -------
    loss
        The negative log-likelihood
    """
    return - np.log(y_pred).mean()


def _fast_log_loss_end_to_end(y_true, y_pred):
    y_true, y_pred = convert_multi_to_binary_fast_log_loss(y_true=y_true, y_pred=y_pred)
    return _fast_log_loss(y_true=y_true, y_pred=y_pred)


# Score function for probabilistic classification
fast_log_loss = make_scorer('log_loss',
                            _fast_log_loss,
                            optimum=0,
                            greater_is_better=False,
                            needs_proba=True)


# Score function for probabilistic classification
fast_log_loss_end_to_end = make_scorer('fast_log_loss_end_to_end',
                                       _fast_log_loss_end_to_end,
                                       optimum=0,
                                       greater_is_better=False,
                                       needs_proba=True)
