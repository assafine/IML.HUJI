import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def delta_time(row):
    return (row["Date"] - row["date_time_year"])


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=[2])
    data["date_time_year"] = pd.to_datetime(data["Year"], format="%Y")
    data["DayOfYear"] = data.apply(lambda row: delta_time(row), axis=1)
    data["DayOfYear"] = data["DayOfYear"].dt.days
    data.dropna(inplace=True)
    data.drop(data[(data["Temp"] <= -50) | (data["Temp"] >= 50)].index,
              inplace=True)
    return data


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    filename = f"C:/Users/Assaf/Documents/University/" \
               f"Year 2 - Semester B/IML/" \
               f"Git Environment/IML.HUJI/datasets/City_Temperature.csv"
    data = load_data(filename)

    # Question 2 - Exploring data for specific country
    data_israel = data[data["Country"] == "Israel"].copy()
    str_year = data_israel["Year"].copy()
    str_year = str_year.astype(str)
    data_israel.loc[:, "Year of sample"] = str_year
    my_fig = px.scatter(data_israel, x="DayOfYear", y="Temp",
                        color="Year of sample",
                        color_discrete_sequence=["red", "blue", "green",
                                                 "yellow",
                                                 "orange", "purple", "black",
                                                 "grey"])
    my_fig.update_layout(
        title=f"Temperture by day of year",
        xaxis_title="Day of year",
        yaxis_title="Temperture",
        font=dict(
            size=18
        ))
    my_fig.show()
    data_is_month = data_israel.groupby("Month").agg("std")["Temp"]
    my_fig = px.bar(data_is_month, y='Temp',
                    title="Standard deviation of temperature by month, Israel")
    my_fig.show()

    # Question 3 - Exploring differences between countries
    data_all_month = data[["Country", "Month", "Temp"]].groupby(
        by=["Country", "Month"], as_index=False).agg(
        ["mean", "std"]).reset_index()
    data_all_month.columns = [" ".join(pair) for pair in
                              data_all_month.columns]
    fig = px.line(data_all_month, x="Month ", y="Temp mean", color="Country ",
                  error_y="Temp std",
                  title="Mean temperature by month per country")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(
        data_israel["DayOfYear"], data_israel["Temp"])
    loss_array = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X, train_y)
        loss_array.append([k, model.loss(test_X, test_y)])
    for res in loss_array:
        print(f"Polynom of degree {res[0]}, loss {res[1]}")
    poly_df = pd.DataFrame(loss_array, columns=["k", "loss"])
    fig = px.bar(poly_df, x="k", y='loss',
                 title="MSE of model by degree (complexity) ")
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(4)
    model.fit(data_israel["DayOfYear"], data_israel["Temp"])
    country_error = []
    for name, group in data.groupby("Country"):
        country_error.append(
            [name, model.loss(group["DayOfYear"], group["Temp"])])
    country_error_df = pd.DataFrame(country_error,
                                    columns=["Country", "Error"])
    fig = px.bar(country_error_df, x="Country", y='Error',
                 title="MSE of Polyfit degree 4 by Country")
    fig.show()
