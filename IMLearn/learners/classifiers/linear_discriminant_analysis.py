from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def extract_label(self, y, k):
        idxs = np.where(y == k)
        return idxs, len(y[idxs])

    @staticmethod
    def update_cov(x, mu):
        return np.outer(x - mu, x - mu)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # np.append(an_array, new_column, axis=1)

        labels = np.unique(y)
        self.classes_ = np.array(labels)
        n_classes = len(labels)
        n_features = X.shape[1]
        self.mu_ = np.zeros((n_classes, n_features))
        self.cov_ = np.zeros((n_features, n_features))
        self._cov_inv = self.cov_.copy()
        self.pi_ = np.zeros(n_classes)
        m = len(y)
        cov_update = np.vectorize(self.update_cov)
        for l in labels:
            k = np.where(self.classes_ == l)[0]
            idx_k, n_k = self.extract_label(y, l)
            self.pi_[k] = n_k / m
            self.mu_[k] = np.mean(X[idx_k], axis=0)

        self.cov_ = np.cov(X.T)
        self._cov_inv = inv(self.cov_)

    @staticmethod
    def _predict_sample(x, inv_cov, mu, pi, classes):
        # print(x,inv_cov, mu,pi,classes,sep = "\n")
        k_num = len(classes)
        likelihoods = np.zeros(k_num)
        for k in range(k_num):
            likelihoods[k] = np.log(
                pi[k]) + x.T @ inv_cov @ mu[k] - 0.5 * mu[k].T @ inv_cov @ mu[k]
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
        # sample_predict = np.vectorize(self.predict_sample)
        # return sample_predict(X, self._cov_inv, self.mu_, self.pi_,
        #                       self.classes_)

        return np.apply_along_axis(self._predict_sample, 1, X, self._cov_inv, self.mu_, self.pi_,
                              self.classes_)

    @staticmethod
    def log_likelihood_sample(x, inv_cov, mu, pi, classes, cov):
        k_num = len(classes)
        likelihoods = np.zeros(k_num)
        det_cov = det(cov)
        d = len(x)
        for k in range(k_num):
            my_vec = x - mu[k]
            likelihoods[k] = np.log(pi[k]) - d * np.log(
                np.pi) / 2 - 0.5 * det_cov - 0.5 * my_vec.T @ inv_cov @ my_vec
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

        # sample_likelihood = np.vectorize(self.log_likelihood_sample)
        # return sample_likelihood(X, self._cov_inv, self.mu_, self.pi_,
        #                          self.classes_, self.cov_)

        return np.apply_along_axis(self.log_likelihood_sample, 1, X, self._cov_inv, self.mu_, self.pi_,
                                 self.classes_, self.cov_)

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
