import os.path
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from gmm import MixtureModel
from bayesian_gmm import BayesianMixtureModel
from utils import preprocess_data, split_data, build_trace

if __name__ == '__main__':

    original_df = pd.read_csv('../data/iris.csv')
    new_name = '../data/iris_numeric.csv'

    if not os.path.exists(new_name):
        preprocess_data(original_df, new_name)

    numeric_iris = pd.read_csv(new_name)

    K = 3
    x_train, y_train, x_test, y_test = split_data(numeric_iris)

    """
    mixture_model = MixtureModel(K, x_train, y_train)
    mixture_model.train()
    mixture_model.predict(x_test, y_test)
    """

    bayesian_gmm = BayesianMixtureModel(K,)
    # bayesian_gmm.train(20, x_train, y_train)

    miu_samples, sigma_samples = bayesian_gmm.predict(x_test, y_test)
    trace = build_trace(miu_samples, sigma_samples)

    # print(trace.posterior)
    # print(list(trace.posterior.data_vars))

    az.plot_trace(trace, var_names=[f"mu_{k}_{d}" for k in range(K) for d in range(x_test.shape[1])], figsize=(12, 8))
    plt.savefig('../plots/means_posterior.png', bbox_inches='tight')

    for k in range(K):
        axes = az.plot_trace(trace, var_names=[f"sigma_{k}_{i}_{j}"
                                        for i in range(x_test.shape[1])
                                        for j in range(x_test.shape[1])]
                                        , figsize=(12, 8))
        # monocolor
        for ax_row in axes:
            for ax in ax_row:
                for line in ax.get_lines():
                    line.set_color("blue")
        plt.savefig(f"../plots/covariance_posterior_{k}.png", bbox_inches='tight')

