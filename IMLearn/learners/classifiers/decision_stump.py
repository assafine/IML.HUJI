from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _find_thr_of_feature(self, values: np.ndarray, labels):
        my_values = values.copy()
        thr_sp, thr_err_sp = self._find_threshold(my_values, labels[0], 1)
        thr_sn, thr_err_sn = self._find_threshold(my_values, labels[0], -1)
        if thr_err_sn < thr_err_sp:
            return [thr_sn, thr_err_sn, -1]
        return [thr_sp, thr_err_sp, 1]

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        X_work = X.copy()
        results = np.apply_along_axis(self._find_thr_of_feature,1,X_work.T,[y])
        best_thr = np.argmin(results[:, 1])
        self.threshold_, self.j_, self.sign_ = results[best_thr][0], best_thr, \
                                               results[best_thr][2]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        X_work = X.copy()
        return np.where(X_work[:, self.j_] < self.threshold_, -self.sign_,
                        self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        prediction_mat = np.column_stack((values, labels))
        prediction_mat = prediction_mat[prediction_mat[:, 0].argsort()]
        thr_err = np.abs(
            prediction_mat[np.sign(prediction_mat[:, 1]) != sign][:, 1].sum())
        mat_length = prediction_mat.shape[0]
        prediction_mat = np.vstack(
            [prediction_mat, [prediction_mat[-1, 0] + 1, 0]])
        thr = prediction_mat[0, 0] - 1
        temp_thr_err = thr_err
        for i in range(mat_length):
            temp_thr_err = temp_thr_err + sign * prediction_mat[i, 1]
            if temp_thr_err < thr_err:
                thr_err = temp_thr_err
                thr = 0.5 * (prediction_mat[i, 0] + prediction_mat[i + 1, 0])
        return (thr, thr_err)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        y_predict = self.predict(X)
        new_y = y.copy()
        new_y = np.sign(np.where(new_y == 0, 1))
        return misclassification_error(y_predict, new_y)
