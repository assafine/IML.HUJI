from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics import loss_functions as ls
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)

    # Drop problematic values
    data["date"] = pd.to_datetime(data["date"], errors='coerce')
    data.dropna(inplace=True)
    data.drop(data[(data["price"] <= 0) | (data["bedrooms"] > 20) | (
            data["sqft_lot15"] <= 0) | (data["price"] <= 0) | (
                           data["bedrooms"] <= 0)].index,
              inplace=True)
    # Create new feature of time since last build pf house
    data["time_since_built"] = data.apply(lambda row: delta_time(row), axis=1)
    data.loc[data["time_since_built"] < 0, "time_since_built"] = 0
    data["grade^6"] = data["grade"] ** 6
    data["time_since_built^0.3"] = data["time_since_built"] ** 0.3
    data["baths_by_beds"] = data["bathrooms"] / data["bedrooms"]
    data["living_room_ratio2"] = data["sqft_living"] / data["sqft_lot"]
    data["bed_times_baths_sqaured"] = data["bedrooms"] * data["bathrooms"] ** 2
    data["condition_by_grade^5"] = data["condition"] * data["grade"] ** 5
    # Catagorise zipcode
    data = pd.concat([data, pd.get_dummies(data["zipcode"], prefix="zipcode")],
                     axis=1)
    # Choose features based on Pearson Correlation:
    X = data[['condition_by_grade^5', 'grade^6', 'sqft_living', 'grade',
              'sqft_above', 'sqft_living15', 'bed_times_baths_sqaured',
              'bathrooms',
              'view', 'sqft_basement', 'bedrooms', 'lat', 'baths_by_beds',
              'zipcode_98004.0', 'waterfront', 'floors', 'zipcode_98039.0',
              'zipcode_98040.0', 'zipcode_98112.0', 'zipcode_98006.0',
              'yr_renovated',
              'living_room_ratio2', 'zipcode_98033.0', 'zipcode_98105.0',
              'sqft_lot',
              'zipcode_98075.0', 'zipcode_98199.0', 'sqft_lot15',
              'zipcode_98002.0',
              'zipcode_98168.0', 'zipcode_98001.0', 'zipcode_98042.0',
              'time_since_built', 'zipcode_98023.0', 'time_since_built^0.3']]

    y = data['price']
    return (X, y)


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X.columns:
        x = X[feature]
        pc = np.cov(x, y=y)[0][1] / (np.std(x) * np.std(y))
        my_fig = go.Figure()
        my_fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers"))
        my_fig.update_layout(
            title=f"{feature} to response, correlation: {pc}",
            xaxis_title=f"{feature}",
            yaxis_title="y",
            font=dict(
                size=12
            ))
        my_fig.write_image(f"{output_path}/{feature}_scatter.jpeg")


def delta_time(row):
    if row["yr_renovated"] == 0:
        return row["date"].year - row["yr_built"]
    return row["date"].year - row["yr_renovated"]


if __name__ == '__main__':
    # Note: Running the code takes a while, uses a lot of features
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    filename = f"C:/Users/Assaf/Documents/University/Year 2 - Semester B/IML/" \
               f"Git Environment/IML.HUJI/datasets/house_prices.csv"
    X, y = load_data(filename)

    # Question 2 - Feature evaluation with respect to response
    dir_path = r"C:\Users\Assaf\Documents\University\Year 2 - Semester B\IML\Exercises" \
               r"\Ex2\plots_houses"
    feature_evaluation(X, y, dir_path)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    model = LinearRegression()
    full_train_mat = pd.concat([train_X, train_y], axis=1)
    loss_array = []
    for p in range(10, 101):
        loss_test_p = []
        for _ in range(10):
            sample = full_train_mat.sample(frac=p / 100)
            sample_y = sample["price"].to_numpy()
            sample_X = sample.drop(columns=["price"]).to_numpy()
            model.fit(sample_X, sample_y)
            predict_y = model.predict(test_X.to_numpy())
            loss_test_p.append(ls.mean_square_error(test_y, predict_y))
        loss_array.append([p, np.mean(loss_test_p), np.std(loss_test_p)])

    loss_array = np.array(loss_array).T
    my_fig = go.Figure()
    my_fig.add_trace(
        go.Scatter(x=loss_array[0, :], y=loss_array[1, :], mode="markers"))
    my_fig.add_trace(
        go.Scatter(x=loss_array[0, :],
                   y=loss_array[1, :] - 2 * loss_array[2, :],
                   fill=None, mode="lines", line=dict(color="lightgrey"),
                   showlegend=False))

    my_fig.add_trace(
        go.Scatter(x=loss_array[0, :],
                   y=loss_array[1, :] + 2 * loss_array[2, :],
                   fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                   showlegend=False))
    my_fig.update_layout(
        title=f"Mean loss plot as function of sample size",
        xaxis_title="Sample percent",
        yaxis_title="MSE",
        font=dict(
            size=18
        ))
    my_fig.show()
