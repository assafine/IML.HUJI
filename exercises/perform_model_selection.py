from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.linear_model
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
    train_X = train_X.to_numpy()[:, 0]
    test_X = test_X.to_numpy()[:, 0]
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
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
                                                    train_X, train_y,
                                                    mean_square_error)


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
    best_k_model = PolynomialFitting(chosen_k).fit(train_X, train_y)
    print(
        f"Test Error for n_samples: {n_samples}, noise: {noise},  k = {chosen_k} : {round(best_k_model.loss(test_X, test_y), 2)}")


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
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_X, train_y, test_X, test_y = split_train_test(X, y,
                                                        n_samples / X.shape[0])
    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_train_scores = []
    ridge_validation_scores = []
    lasso_train_scores = []
    lasso_validation_scores = []
    ridge_lam_range = 0.001
    lasso_lam_range = 0.01
    ridge_lam_array = np.linspace(0, ridge_lam_range, n_evaluations).tolist()
    lasso_lam_array = np.linspace(0, lasso_lam_range,
                                  n_evaluations + 1).tolist()[1:]
    for lam in ridge_lam_array:
        lam_train_score, lam_val_score = cross_validate(RidgeRegression(lam),
                                                        train_X, train_y,
                                                        mean_square_error)
        ridge_train_scores.append(lam_train_score)
        ridge_validation_scores.append(lam_val_score)

    my_fig = go.Figure()
    my_fig.add_trace(
        go.Scatter(x=ridge_lam_array, y=ridge_train_scores, mode="markers",
                   name="Train Errors"))
    my_fig.add_trace(
        go.Scatter(x=ridge_lam_array, y=ridge_validation_scores,
                   mode="markers",
                   name="Validation Errors"))
    my_fig.update_layout(
        title=f"Validation and Train Errors of Ridge Regression with n_samples: {n_samples}",
        xaxis_title="k",
        yaxis_title="Error",
        font=dict(
            size=18
        ))
    my_fig.show()

    for lam in lasso_lam_array:
        lam_train_score, lam_val_score = cross_validate(
            Lasso(lam, max_iter=10000),
            train_X, train_y,
            mean_square_error)
        lasso_train_scores.append(lam_train_score)
        lasso_validation_scores.append(lam_val_score)

    my_fig = go.Figure()
    my_fig.add_trace(
        go.Scatter(x=lasso_lam_array, y=lasso_train_scores, mode="markers",
                   name="Train Errors"))
    my_fig.add_trace(
        go.Scatter(x=lasso_lam_array, y=lasso_validation_scores,
                   mode="markers",
                   name="Validation Errors"))
    my_fig.update_layout(
        title=f"Validation and Train Errors of Lasso Regression with n_samples: {n_samples}",
        xaxis_title="k",
        yaxis_title="Error",
        font=dict(
            size=18
        ))
    my_fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    chosen_lam_ridge = ridge_lam_array[np.argmin(ridge_validation_scores)]
    chosen_lam_lasso = lasso_lam_array[np.argmin(lasso_validation_scores)]
    best_ridge_model = RidgeRegression(chosen_lam_ridge).fit(train_X, train_y)
    print(
        f"Test Error for n_samples: {n_samples}\n  "
        f"Ridge Regression with lam = {chosen_lam_ridge} : "
        f"{best_ridge_model.loss(test_X, test_y)}")
    best_lasso_model = Lasso(chosen_lam_lasso, max_iter=10000).fit(train_X,
                                                                   train_y)
    best_lasso_test_pred = best_lasso_model.predict(test_X)
    print(
        f"Test Error for n_samples: {n_samples}\n  "
        f"Lasso Regression with lam = {chosen_lam_lasso} : "
        f"{mean_square_error(best_lasso_test_pred, test_y)}")
    my_ls = LinearRegression().fit(train_X, train_y)
    print(
        f"Test Error for n_samples: {n_samples}\n  "
        f"Our LS Algorithm : "
        f"{my_ls.loss(test_X, test_y)}")


if __name__ == '__main__':
    np.random.seed(0)

    # Part 1
    print("Part 1")
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)

    # Part 2
    print("Part 2")
    select_regularization_parameter()
