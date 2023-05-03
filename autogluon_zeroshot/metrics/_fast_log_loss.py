import numpy as np

from autogluon.core.metrics import make_scorer


def extract_true_class_prob(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Note: Data must be pre-normalized. The data is not normalized within this function for speed purposes.
    The predictions can then be passed to _fast_log_loss.
    :param y_true: an array with shape (n_samples,) that contains all the classes, values must be in [0, n_class[
    :param y_pred: an array with shape (n_samples, n_class) whose values are the prediction probability assigned to the
    ground truth class
    :return: an array with shape (n_samples) that contains the predicted probability of the true class for each sample
    """
    assert len(y_true) == len(y_pred)
    assert y_pred.ndim == 2
    return y_pred[range(len(y_pred)), y_true]


def mean_log_loss(true_class_prob: np.ndarray) -> float:
    return - np.log(true_class_prob).mean()


def _fast_log_loss_end_to_end(y_true, y_pred):
    true_class_prob = extract_true_class_prob(y_true=y_true, y_pred=y_pred)
    return mean_log_loss(true_class_prob=true_class_prob)


"""
Heavily optimized log_loss implementation that is valid under a specific context and avoids all sanity checks.
This can be >100x faster than sklearn.

NOTE: 
1. You must first preprocess the input y_pred by calling `extract_true_class_prob` which converts to a 1-dimensional
  array whose values are the prediction probability assigned to the ground truth class. All other classes that are not 
  the ground truth are ignored, as they are not necessary to calculate log_loss.
2. There is no epsilon / value clipping, ensure y_pred ranges do not include `0` or `1` to avoid infinite loss.
3. There is no support for sample weights.

Parameters
----------
y_true : ignored
true_class_prob : array-like of float that contains the prediction probabilities of the ground truth class. shape = (n_samples,)

Returns
-------
loss
    The negative log-likelihood
"""
# Score function for probabilistic classification
fast_log_loss = make_scorer('log_loss',
                            lambda _, true_class_prob: mean_log_loss(true_class_prob),
                            optimum=0,
                            greater_is_better=False,
                            needs_proba=True)


# Score function for probabilistic classification
fast_log_loss_end_to_end = make_scorer('fast_log_loss_end_to_end',
                                       _fast_log_loss_end_to_end,
                                       optimum=0,
                                       greater_is_better=False,
                                       needs_proba=True)
