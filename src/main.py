import os.path

import numpy as np
import pandas as pd
from gmm import MixtureModel
from utils import preprocess_data, split_data

if __name__ == '__main__':

    original_df = pd.read_csv('../data/iris.csv')
    new_name = '../data/iris_numeric.csv'

    if not os.path.exists(new_name):
        preprocess_data(original_df, new_name)

    numeric_iris = pd.read_csv(new_name)

    K = 3
    x_train, y_train, x_test, y_test = split_data(numeric_iris)
    mixture_model = MixtureModel(K, x_train, y_train)
    # mixture_model.train()
    mixture_model.predict(x_test, y_test)
