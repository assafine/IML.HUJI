from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        self.mu_ = self._estimate_mean(self, X)
        self.var_ = self._estimate_var(self, X, self.biased_)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            X to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        vector_func = np.vectorize(self._gu_pdf_calc)
        return vector_func(X, self.mu_, self.var_)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            X to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        n = X.size
        updated_X = (X - mu) ** 2
        return -np.log(2 * np.pi) * n / 2 - np.log(sigma) * n / 2 - np.sum(
            updated_X) / (2 * sigma)

    @staticmethod
    def _gu_pdf_calc(x, mu: float, sigma: float) -> float:
        if sigma == 0:
            raise ValueError("Not a legal variance")
        coefficient = 1 / (np.sqrt(2 * np.pi * sigma))
        exp = np.exp((-(x - mu) ** 2) / (2 * sigma))
        return exp * coefficient

    @staticmethod
    def _estimate_mean(self, X: np.ndarray) -> float:
        return np.mean(X)

    @staticmethod
    def _estimate_var(self, X: np.ndarray, biased_: bool) -> float:
        ddof = 0 if biased_ else 1
        return np.var(X, ddof=ddof)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = self._estimate_mean_multi(self, X)
        self.cov_ = self._estimate_cov_multi(self, X)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            X to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        inv_cov = inv(self.cov_)
        det_cov = det(self.cov_)
        return np.apply_along_axis(self._multi_gu_feature_pdf_calc, 1, X,
                                   self.mu_,
                                   inv_cov, det_cov)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            X to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """

        d = mu.shape[0]
        n = X.shape[1]
        inv_cov = inv(cov)
        dist_vec = X - mu
        # Calculate equation using element wise multiplication
        sub_mat = (inv_cov @ dist_vec.T).T
        sub_mat_2 = np.sum(dist_vec * sub_mat, axis=1)
        first_item = -0.5 * np.sum(sub_mat_2)
        log_det = slogdet(inv_cov)
        second_item = 0.5 * n * log_det[1]
        third_item = -0.5 * d * n * np.log(np.pi)
        return first_item + second_item + third_item

    @staticmethod
    def _estimate_mean_multi(self, X: np.ndarray):
        return np.mean(X, axis=0)

    @staticmethod
    def _estimate_cov_multi(self, X: np.ndarray):
        return np.cov(X.T)

    @staticmethod
    def _multi_gu_feature_pdf_calc(x: np.ndarray, mu: np.ndarray,
                                   inv_cov: np.ndarray, det_cov) -> float:
        d = x.shape[0]
        coefficient = 1 / (np.sqrt(((2 * np.pi) ** d) * det_cov))
        exp = np.exp(- sample_vector_multiply(x, mu, inv_cov) / 2)
        return exp * coefficient


def sample_vector_multiply(x: np.ndarray, mu: np.ndarray,
                           inv_cov: np.ndarray):
    return (x - mu).T @ inv_cov @ (x - mu)
