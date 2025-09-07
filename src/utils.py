import numpy as np
import pandas as pd

def init_cov_matrix(dim: int, small: bool=True):

    if small:
        return np.diag(np.random.rand(dim) + 0.1)
    else:
        sigma = np.random.rand(dim, dim)
        return sigma * sigma.transpose()

def _standardize_data(df: pd.DataFrame):

    for col in df.columns[:-1]: # col is just the column's name
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df

def preprocess_data(df: pd.DataFrame, out_name: str):

    df['species'], original_string = pd.factorize(df['species'])
    df.drop_duplicates()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna()
    df = _standardize_data(df)
    print(f"df.head() = \n {df.head()}")
    df.to_csv(out_name, index=False)

def split_data(df: pd.DataFrame):

    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    size = int(0.8 * df.shape[0])
    # print(f"train size = {size}")
    x_train, y_train = df.iloc[:size, :-1], df.iloc[:size, -1:]
    x_test, y_test = df.iloc[size:, :-1], df.iloc[size:, -1:]
    return x_train, y_train, x_test, y_test

def relabel_data(num_clusters, cluster_indices, num_classes):

    # assigning the true labels to each gaussian component, based on 'majority vote relabelling'
    cluster_labels = [0 for _ in range(num_clusters)]
    for i, cluster in enumerate(cluster_indices):
        frequencies = [0 for _ in range(num_classes)]
        for (_, true_label) in cluster:
            frequencies[true_label] += 1
        cluster_labels[i] = np.argmax(frequencies)
    return cluster_labels


