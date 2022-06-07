from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    true_hypothesis = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(-1.2, 2, n_samples)
    perfect_y = true_hypothesis(X)
    y = perfect_y + np.random.normal(scale=noise, size=len(perfect_y))
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y), 2 / 3)
    train_X = train_X.to_numpy()[:,0]
    test_X = test_X.to_numpy()[:,0]
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()

    # print(train_X.shape)
    # print(test_X)
    # print(train_y)
    # print(test_y)

    my_fig = go.Figure()
    my_fig.add_trace(
        go.Scatter(x=X, y=perfect_y, mode="markers",
                   name="Real function"))

    my_fig.add_trace(
        go.Scatter(x=train_X, y=train_y, mode="markers",
                   name="Train Set"))
    my_fig.add_trace(
        go.Scatter(x=test_X, y=test_y, mode="markers",
                   name="Test Set"))
    my_fig.update_layout(
        title=f"Plot f(x) train and test, using n_samples: {n_samples}, noise: {noise}",
        xaxis_title="x",
        yaxis_title="f(x)",
        font=dict(
            size=18
        ))
    my_fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores = []
    validation_scores = []
    k_array = list(range(11))
    for k in k_array:
        k_train_score, k_val_score = cross_validate(PolynomialFitting(k),
                                                     train_X, train_y, mean_square_error)
        train_scores.append(k_train_score)
        validation_scores.append(k_val_score)
    my_fig = go.Figure()
    my_fig.add_trace(
        go.Scatter(x=k_array, y=train_scores, mode="markers",
                   name="Train Errors"))
    my_fig.add_trace(
        go.Scatter(x=k_array, y=validation_scores, mode="markers",
                   name="Validation Errors"))
    my_fig.update_layout(
        title=f"Validation and Train Errors of data with n_samples: {n_samples}, noise: {noise}",
        xaxis_title="k",
        yaxis_title="Error",
        font=dict(
            size=18
        ))
    my_fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    chosen_k = np.argmin(validation_scores)
    # chosen_k = 8
    best_k_model = PolynomialFitting(chosen_k).fit(train_X,train_y)
    print(f"Test Error for n_samples: {n_samples}, noise: {noise},  k = {chosen_k} : {round(best_k_model.loss(test_X,test_y),2)}")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Part 1
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500,10)
