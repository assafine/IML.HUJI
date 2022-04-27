from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def extract_label(self, y, k):
        idxs = np.where(y == k)
        return idxs, len(y[idxs])

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        labels = np.unique(y)
        self.classes_ = np.array(labels)
        n_classes = len(labels)
        n_features = X.shape[1]
        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.pi_ = np.zeros(n_classes)
        m = len(y)
        for l in labels:
            k = np.where(self.classes_ == l)[0]
            idx_k, n_k = self.extract_label(y, l)
            X_k = X[idx_k].copy()
            self.pi_[k] = n_k / m
            self.mu_[k] = np.mean(X[idx_k], axis=0)
            for i in range(X.shape[1]):
                self.vars_[k, i] = np.var(X_k[:, i], ddof=1)

    @staticmethod
    def _predict_sample(x, vars, mu, pi, classes):
        k_num = len(classes)
        likelihoods = np.zeros(k_num)
        d = len(x)
        for k in range(k_num):
            cov = np.diag(vars[k, :])
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            my_vec = x - mu[k]
            likelihoods[k] = np.log(pi[k]) - d * np.log(
                np.pi) / 2 - 0.5 * np.log(
                det_cov) - 0.5 * my_vec.T @ inv_cov @ my_vec
        max_k = np.argmax(likelihoods)
        return classes[max_k]

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
        # sample_predict = np.vectorize(self._predict_sample)
        # return sample_predict(X, self.vars_, self.mu_, self.pi_,
        #                       self.classes_)

        return np.apply_along_axis(self._predict_sample, 1, X, self.vars_,
                                   self.mu_, self.pi_,
                                   self.classes_)

    @staticmethod
    def log_likelihood_sample(x, vars, mu, pi, classes):
        k_num = len(classes)
        likelihoods = np.zeros(k_num)
        d = len(x)
        for k in range(k_num):
            cov = np.diag(vars[k, :])
            inv_cov = np.linalg.inv(cov)
            det_cov = np.det(cov)
            my_vec = x - mu[k]
            likelihoods[k] = np.log(pi[k]) - d * np.log(
                np.pi) / 2 - 0.5 * np.log(
                det_cov) - 0.5 * my_vec.T @ inv_cov @ my_vec
        return likelihoods

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        return np.apply_along_axis(self.log_likelihood_sample, 1, X,
                                   self.vars_, self.mu_, self.pi_,
                                   self.classes_)

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
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)
