import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


# import sys
# sys.path.insert(0,'C:/Users/Assaf/Documents/University/Year 2 - Semester B/IML/Git Environment/IML.HUJI/datasets')


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_calc_callback(fit: Perceptron, X, y):
            fit.fitted_ = True
            losses.append(fit.loss(X, y))

        model = Perceptron(callback=loss_calc_callback)
        model.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        iter_num_arr = [i for i in range(len(losses))]
        my_fig = go.Figure()
        my_fig.add_trace(
            go.Scatter(x=iter_num_arr, y=losses, mode="markers"))
        my_fig.update_layout(title=f"{n} data loss per iteration",
                             xaxis_title="Iteration",
                             yaxis_title="Loss",
                             font=dict(
                                 size=16
                             ))
        my_fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      showlegend=False,
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        naive_base = GaussianNaiveBayes()
        lda = LDA()
        naive_base.fit(X, y)
        naive_pred = naive_base.predict(X)
        lda.fit(X, y)
        lda_pred = lda.predict(X)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        my_fig = make_subplots(rows=1, cols=2,
                               subplot_titles=[
                                   f"Gaussian Naive Base, Accuracy: {accuracy(y, naive_pred)}",
                                   f"LDA, Accuracy: {accuracy(y, lda_pred)}"],
                               horizontal_spacing=0.1, vertical_spacing=1)

        # Add traces for data-points setting symbols and colors
        label_symbols = np.array(["circle", "cross", "triangle-up"])
        my_fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                    showlegend=False,
                                    marker=dict(color=naive_pred,
                                                symbol=label_symbols[y],
                                                colorscale=["red", "blue",
                                                            "green"])),
                         row=1, col=1)

        my_fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                    showlegend=False,
                                    marker=dict(color=lda_pred,
                                                symbol=label_symbols[y],
                                                colorscale=["red", "blue",
                                                            "green"])),
                         row=1, col=2)


        # Add `X` dots specifying fitted Gaussians' means
        my_fig.add_trace(
            go.Scatter(x=naive_base.mu_[:, 0], y=naive_base.mu_[:, 1],
                       mode="markers",
                       showlegend=False,
                       marker=dict(symbol=['x', 'x', 'x'],
                                   # color=[1,1,1],
                                   color="black",
                                   size=15,
                                   line=dict(color="black", width=0.5))),
            row=1, col=1)

        my_fig.add_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                                    mode="markers",
                                    showlegend=False,
                                    marker=dict(symbol=['x', 'x', 'x'],
                                                color="black",
                                                size=15,
                                                line=dict(color="black",
                                                          width=0.5))),
                         row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            naive_cov = np.diag(naive_base.vars_[i, :])
            my_fig.add_trace(get_ellipse(naive_base.mu_[i], naive_cov), row=1,
                             col=1)
            my_fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

        my_fig.update_layout(title = dict(text=f"Classification of {f}",
                                          x=0.5,
                                          y=0.98,
                                          xanchor= 'center',
                                          yanchor = 'top'),
                                          font=dict(size=16))
        my_fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
