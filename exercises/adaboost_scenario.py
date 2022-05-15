import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_ml = AdaBoost(DecisionStump, n_learners)
    ada_ml.fit(train_X, train_y)
    results = []
    for t in range(1, n_learners + 1):
        train_error = ada_ml.partial_loss(train_X, train_y, t)
        test_error = ada_ml.partial_loss(test_X, test_y, t)
        results.append([t, train_error, test_error])

    results = np.asarray(results)
    my_fig = go.Figure()
    my_fig.add_trace(
        go.Scatter(x=results[:, 0], y=results[:, 1], mode="markers",
                   name="Train error"))
    my_fig.add_trace(
        go.Scatter(x=results[:, 0], y=results[:, 2], mode="markers",
                   name="Test error"))
    my_fig.update_layout(
        title=f"Mean loss plot as function of number of week-learners",
        xaxis_title="Num. learners",
        yaxis_title="MSE",
        font=dict(
            size=18
        ))
    my_fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    my_fig = make_subplots(rows=2, cols=2,
                           subplot_titles=[rf"$\textbf{{Ensemble size: {t}}}$"
                                           for t in
                                           T],
                           horizontal_spacing=0.01, vertical_spacing=.03)
    symbols = np.array(["circle", "x"])
    # print(train_y)
    i = -1
    for t in T:
        i += 1

        def my_predict(X):
            return ada_ml.partial_predict(X, t)

        my_fig.add_traces([decision_surface(my_predict, lims[0], lims[1],
                                            showscale=False),
                           go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                      mode="markers",
                                      showlegend=False,
                                      marker=dict(color=[
                                          ((test_y + 1) / 2).astype(int)],
                                                  symbol=symbols[((
                                                                              test_y + 1) / 2).astype(
                                                      int)],
                                                  colorscale=[custom[0],
                                                              custom[-1]],
                                                  line=dict(color="black",
                                                            width=1)))],
                          rows=(i // 2) + 1, cols=(i % 2) + 1)

    my_fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries Of Ensemble (per size) }}$",
        title_x=0.5,
        font=dict(
            size=18
        ),
        margin=dict(t=50)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    my_fig.show()

    # Question 3: Decision surface of best performing ensemble
    losses = [ada_ml.partial_loss(test_X, test_y, t) for t in range(n_learners)]
    min_idx = np.argmin(losses)

    my_fig = go.Figure()
    t = min_idx

    def my_predict(X):
        return ada_ml.partial_predict(X, t)

    my_fig.add_traces([decision_surface(my_predict, lims[0], lims[1],
                                        showscale=False),
                       go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                  mode="markers",
                                  showlegend=False,
                                  marker=dict(color=[
                                      ((test_y + 1) / 2).astype(int)],
                                      symbol=symbols[((test_y + 1) / 2).astype(
                                          int)],
                                      colorscale=[custom[0],
                                                  custom[-1]],
                                      line=dict(color="black",
                                                width=1)))])

    my_fig.update_layout(
        title=rf"$\textbf{{Maximum Accuracy Ensemble Size: {t}, Accuracy: {1 - losses[min_idx]} }}$",
        title_x=0.5,
        font=dict(
            size=18
        ),
        margin=dict(t=50)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    my_fig.show()

    # Question 4: Decision surface with weighted samples

    t = n_learners
    def my_predict(X):
        return ada_ml.partial_predict(X, t)
    my_fig = go.Figure()
    point_sizes = (ada_ml.D_[n_learners] / np.max(ada_ml.D_[n_learners])) * 10
    my_fig.add_traces([decision_surface(my_predict, lims[0], lims[1],
                                        showscale=False),
                       go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                  mode="markers",
                                  showlegend=False,
                                  marker=dict(color=[
                                      ((train_y + 1) / 2).astype(int)],
                                      symbol=symbols[((train_y + 1) / 2).astype(
                                          int)],
                                      colorscale=[custom[0],
                                                  custom[-1]],
                                      size = point_sizes,
                                      line=dict(color="black",
                                                width=1)))])

    my_fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries of Weighted points, Ensemble Size: {t}}}$",
        title_x=0.5,
        font=dict(
            size=18
        ),
        margin=dict(t=50)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    my_fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
