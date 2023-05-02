import numpy as np
import pytest
import sklearn
from sklearn.preprocessing import normalize
from autogluon.core.metrics import log_loss

from autogluon_zeroshot.metrics import _fast_log_loss


def generate_y_true_and_y_pred_proba(num_samples, num_classes, random_seed=0):
    np.random.seed(seed=random_seed)
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = np.random.rand(num_samples, num_classes)
    y_pred = normalize(y_pred, axis=1, norm='l1')
    return y_true, y_pred


@pytest.mark.parametrize('y_true,y_pred',
                         [([0, 2, 1, 1],
                           [[0.1, 0.2, 0.7],
                            [0.2, 0.1, 0.7],
                            [0.3, 0.4, 0.3],
                            [0.01, 0.9, 0.09]])])
def test_fast_log_loss(y_true, y_pred):
    """Ensure fast_log_loss produces equivalent scores to AutoGluon and Scikit-Learn log_loss implementations"""
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.float32)
    ag_loss = log_loss(y_true, y_pred)
    sk_loss = -sklearn.metrics.log_loss(y_true, y_pred)
    np.testing.assert_allclose(ag_loss, sk_loss)

    y_true_opt, y_pred_opt = _fast_log_loss.convert_multi_to_binary_fast_log_loss(y_true, y_pred)
    fast_loss = _fast_log_loss.fast_log_loss(y_true_opt, y_pred_opt)
    fast_loss_end_to_end = _fast_log_loss.fast_log_loss_end_to_end(y_true, y_pred)

    np.testing.assert_allclose(ag_loss, fast_loss)
    np.testing.assert_allclose(ag_loss, fast_loss_end_to_end)


@pytest.mark.parametrize('num_samples,num_classes',
                         [
                             (1, 2),
                             (1, 10),
                             (1000, 2),
                             (1000, 10),
                             (10000, 2),
                             (10000, 100),
                         ])
def test_fast_log_loss_large(num_samples, num_classes):
    """
    Ensure fast_log_loss produces equivalent scores to AutoGluon and Scikit-Learn log_loss implementations
    across various data dimensions.
    """
    y_true, y_pred = generate_y_true_and_y_pred_proba(num_samples=num_samples, num_classes=num_classes)

    ag_loss = log_loss(y_true, y_pred)
    sk_loss = -sklearn.metrics.log_loss(y_true, y_pred, labels=list(range(num_classes)))
    np.testing.assert_allclose(ag_loss, sk_loss)

    y_true_opt, y_pred_opt = _fast_log_loss.convert_multi_to_binary_fast_log_loss(y_true, y_pred)
    fast_loss = _fast_log_loss.fast_log_loss(y_true_opt, y_pred_opt)
    fast_loss_end_to_end = _fast_log_loss.fast_log_loss_end_to_end(y_true, y_pred)

    np.testing.assert_allclose(ag_loss, fast_loss)
    np.testing.assert_allclose(ag_loss, fast_loss_end_to_end)
