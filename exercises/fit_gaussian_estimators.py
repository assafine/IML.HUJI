from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"



def create_abs_dist_data(my_gau, drawn_samples):
    mean_data = np.zeros((100, 2))
    for i in range(1, 101):
        sample_size = 10 * i
        samples = drawn_samples[:sample_size]
        my_gau.fit(samples)
        mean_data[i - 1, 0] = 10 * i
        mean_data[i - 1, 1] = np.abs(my_gau.mu_ - 10)
    # print (mean_data.shape)
    # print(mean_data)
    return mean_data


def test_univariate_gaussian():

    # Question 1 - Draw samples and print fitted model
    my_gau = UnivariateGaussian()
    samples = np.random.normal(10, 1, 1000)
    my_gau.fit(samples)
    print(f"Q1. Estimated mean and variance: ({my_gau.mu_},{my_gau.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    second_gau = UnivariateGaussian()
    abs_data = create_abs_dist_data(second_gau, samples)
    my_fig = go.Figure()
    my_fig.add_trace(
        go.Scatter(x=abs_data[:, 0], y=abs_data[:, 1], mode="lines+markers"))
    my_fig.update_layout(title="Distance of real mean from estimated value",
                         xaxis_title="Sample size",
                         yaxis_title="Absolute value distance",
                         font=dict(
                             size=18
                         ))
    my_fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sample_pdfs = my_gau.pdf(samples)
    my_fig = go.Figure()
    my_fig.add_trace(
        go.Scatter(x=samples, y=sample_pdfs, mode="markers"))
    my_fig.update_layout(title="Sample PDF",
                         xaxis_title="Sample",
                         yaxis_title="PDF",
                         font=dict(
                             size=18
                         ))
    my_fig.show()


def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model
    samples_true_mean = np.array([0, 0, 4, 0])
    samples_true_cov = np.array([[1, 0.2, 0, 0.5],
                                 [0.2, 2, 0, 0],
                                 [0, 0, 1, 0],
                                 [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(samples_true_mean,
                                            samples_true_cov, 1000)
    my_gau = MultivariateGaussian()
    my_gau.fit(samples)
    print("\nQ4. Estimated mean:")
    print("\n", my_gau.mu_)
    print("\nQ4. Estimated cov matrix:")
    print("\n", my_gau.cov_)

    # Question 5 - Likelihood evaluation
    d_1_linspace = np.linspace(-10, 10, 200)
    d_2_linspace = np.transpose(
        np.array([np.repeat(d_1_linspace, 200), np.tile(d_1_linspace, 200)]))
    f1 = d_2_linspace[:, 0]
    f2 = np.zeros((40000, 1))
    f3 = d_2_linspace[:, 1]
    f4 = np.zeros((40000, 1))
    my_mu_mat = np.column_stack((f1, f2, f3, f4))
    likelihood_results = np.apply_along_axis(my_gau.log_likelihood, 1,
                                             my_mu_mat, samples_true_cov,
                                             samples)
    results = np.column_stack((f1, f3, likelihood_results))
    my_graph_data = go.Heatmap(x=f3, y=f1, z=likelihood_results)
    my_fig = go.Figure(my_graph_data)
    my_fig.update_layout(title="Log Likelihood as a function of (f3,f1)",
                         xaxis_title="f3",
                         yaxis_title="f1",
                         font=dict(
                             size=18
                         ))
    my_fig.show()

    # Question 6 - Maximum likelihood
    max_idx = np.argmax(results[:, 2])
    print(
        f"\nEstimated f3, f1 values with maximum likelihood: "
        f"{np.round(results[max_idx, 0], 3)}, "
        f"{np.round(results[max_idx, 1], 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
