from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    idx_range = np.arange(X.shape[0])
    fold_idxs = np.array_split(idx_range, cv)
    test_errors = []
    train_errors = []
    for idxs in fold_idxs:
        train_x = X.copy()
        train_y = y.copy()
        test_x = train_x[idxs]
        test_y = train_y[idxs]
        train_x = np.delete(train_x, idxs, axis=0)
        train_y = np.delete(train_y, idxs, axis=0)
        estimator.fit(train_x, train_y)
        predict_train_y = estimator.predict(train_x)
        train_errors.append(scoring(predict_train_y, train_y))
        predict_test_y = estimator.predict(test_x)
        test_errors.append(scoring(predict_test_y, test_y))

    return (np.mean(train_errors), np.mean(test_errors))
