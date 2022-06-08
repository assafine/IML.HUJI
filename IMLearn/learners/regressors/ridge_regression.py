from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

# Added line
from ...metrics import loss_functions as lf


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float,
                 include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """

        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        # print(X.shape)
        X_rows, X_cols = X.shape[0], X.shape[1]
        if self.include_intercept_:
            intercept_vec = np.ones(X_rows)
            X = np.insert(X, 0, intercept_vec, axis=1)
            X_cols += 1
        concat_mat = np.sqrt(self.lam_) * np.identity(X_cols)
        if self.include_intercept_:
            concat_mat[0, 0] = 0
        concat_mat_response = np.zeros(X_cols)
        lam_X = np.vstack((X, concat_mat))
        lam_y = np.concatenate([y,concat_mat_response])
        self.coefs_ = np.linalg.pinv(lam_X) @ lam_y

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        X_rows = X.shape[0]
        if self.include_intercept_:
            intercept_vec = np.ones(X_rows)
            X = np.insert(X, 0, intercept_vec, axis=1)
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        predicted_values = self.predict(X)
        return lf.mean_square_error(y, predicted_values)
