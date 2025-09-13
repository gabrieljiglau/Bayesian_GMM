import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import dirichlet

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

def dirichlet_pdf(weights):
    return dirichlet(weights)

def init_cluster_means(x_train, no_clusters):

    k_means = KMeans(n_clusters=no_clusters, n_init=1, max_iter=1, random_state=13)
    labels = k_means.fit_predict(x_train)

    return labels, k_means.cluster_centers_

def init_soft_counts(no_clusters, labels, x_train):

    """
    simulate the soft counts, since we used k-means (hard clustering algorithm) to initialize the clusters
    """

    additive = 1.4 / no_clusters
    soft_counts = np.full((no_clusters, x_train.shape[0]), additive)
    for index, x_row in enumerate(x_train.itertuples()):
        soft_counts[labels[index], index] += 1

    soft_counts /= ((additive * no_clusters) + 1) # normalization
    return soft_counts


def compute_weighted_mean(x_train, labels, no_clusters, soft_counts):

    print(f"type(soft_counts) = {type(soft_counts)}")  ## de ce primesc int aici ?

    weighted_means = np.zeros((no_clusters, x_train.shape[1]))

    for i, row in enumerate(x_train.itertuples()):
        current_cluster = labels[i]
        weighted_means[current_cluster,:] += np.array(row[1:]) * soft_counts[current_cluster, i]

    return weighted_means


def init_precision_matrix(x_train, labels, no_clusters):

    weighted_means = compute_weighted_mean(x_train, labels, no_clusters, x_train.shape[0])
    covariance_matrix = 0
    for i, x_row in enumerate(x_train.itertuples()):
        data_diff =  weighted_means[labels[i]] / x_row[1:]
        data_diff = data_diff.reshape(-1, 1)
        covariance_matrix += data_diff @ data_diff.transpose()

    # print(f"cov_matrix = {covariance_matrix}")
    covariance_matrix /= x_train.shape[0]
    precision_matrix = np.linalg.inv(covariance_matrix)
    # print(np.all(np.linalg.eigvals(precision_matrix) > 0)) # sanity check, all eigenvalues should be positive
    return np.linalg.inv(covariance_matrix)


## aici se plange
def init_priors(x_train, labels, no_clusters):

    priors, data_mean = [], []
    cols = x_train.columns[:-1]
    for i in range(len(cols)):
        dimension_mean = x_train[cols[i]].mean()
        data_mean.append(dimension_mean)

    priors.append(data_mean)
    priors.append(1)  # the confidence in our means; 1 ~ 'not so much'
    priors.append(x_train.shape[1] + 2)  # degree of freedom
    priors.append(init_precision_matrix(x_train, labels, no_clusters))

    return priors


