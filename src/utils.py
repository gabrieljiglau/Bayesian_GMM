import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import dirichlet
from scipy.linalg import fractional_matrix_power

def init_cov_matrix(dim: int, small: bool=True):

    if small:
        return np.diag(np.random.rand(dim) + 0.1)
    else:
        sigma = np.random.rand(dim, dim)
        return sigma * sigma.transpose()

def _standardize_data(df: pd.DataFrame):

    for col in df.columns[:-1]: # col is just the column's name
        df[col] = (df[col] - df[col].cluster_mean()) / df[col].std()

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

    """
    assigning the true labels to each gaussian component, based on 'majority vote relabelling'
    """
    cluster_labels = [0 for _ in range(num_clusters)]
    for i, cluster in enumerate(cluster_indices):
        frequencies = [0 for _ in range(num_classes)]
        for (_, true_label) in cluster:
            frequencies[true_label] += 1
        cluster_labels[i] = np.argmax(frequencies)
    return cluster_labels

def map_clusters(true_labels, cluster_assignments, no_clusters):

    cm = confusion_matrix(true_labels, cluster_assignments, labels=range(no_clusters))
    # print(f"confusion_matrix = {cm}")
    rows, cols = linear_sum_assignment(-cm)

    mapping = {cluster: label for cluster, label in zip(rows, cols)}
    new_assignments = np.array([mapping[cluster] for cluster in cluster_assignments])

    return new_assignments, mapping

def init_cluster_means(x_train, no_clusters):

    k_means = KMeans(n_clusters=no_clusters, n_init=1, max_iter=1, random_state=13)
    labels = k_means.fit_predict(x_train)

    return labels, k_means.cluster_centers_

def log_det_cholesky(matrix):

    cholesky_det = np.linalg.cholesky(matrix)
    return np.sum(np.log(np.diag(cholesky_det))) * 2

def init_responsibilities(no_clusters, labels, x_train):

    """
    simulate the soft counts, since we used k-means (hard clustering algorithm) to initialize the clusters
    """

    additive = 1.4 / no_clusters
    soft_counts = np.full((no_clusters, x_train.shape[0]), additive)
    for index, x_row in enumerate(x_train.itertuples()):
        soft_counts[labels[index], index] += 1

    soft_counts /= ((additive * no_clusters) + 1) # normalization
    return np.array(soft_counts)


def _weighted_mean(x_train, no_clusters, soft_counts):

    weighted_means = np.zeros((no_clusters, x_train.shape[1]))
    for k in range(no_clusters):
        norm_term = np.sum(soft_counts[k, :])
        for i, row in enumerate(x_train.itertuples()):
            weighted_means[k,:] += (np.array(row[1:]) * soft_counts[k, i]) / norm_term

        weighted_means[k, :] /= norm_term
    return weighted_means

def _data_mean(x_train):

    data_mean = np.zeros(x_train.shape[1])
    for i in range(x_train.shape[1]):
        dimension_mean = x_train.iloc[:, i].mean()
        data_mean[i] = dimension_mean
    return data_mean

def init_priors(x_train, labels, no_clusters, soft_counts, dim_data, epsilon):

    """
    return priors: data_mean, strength_mean (the confidence in the data_mean), degree_of_freedom, precision_matrix
    """

    return [_data_mean(x_train), 1, x_train.shape[1] + 2, (np.eye(dim_data) * epsilon) + np.eye(dim_data)]

def gaussian_pdf(instance, dim_data, covariance, mean):
    # instance is a real valued vector
    denominator =  ((2 * np.pi) ** dim_data / 2) * np.sqrt(np.linalg.det(covariance))
    diff = instance - mean
    cov_inverse = fractional_matrix_power(covariance, -1)
    exp_term = np.exp(-1/2 * (diff.transpose() @ cov_inverse @ diff))

    return exp_term / denominator

def compute_log_likelihood(x_train, dim_data, cluster_means, cov_matrices, mixing_weights):

    log_likelihood = 0
    for row in x_train.itertuples():
        prob_sum = 0
        for i in range(len(cluster_means)):
            x_values = np.array(row[1:])
            prob_sum += mixing_weights[i] * gaussian_pdf(x_values, dim_data, cov_matrices[i], cluster_means[i])
        log_likelihood += np.log(prob_sum + 1e-12)

    return log_likelihood

def get_free_parameters(K, d):
    ## believe me
    """
    :return: the number of free parameters for a gaussian mixture model, with K clusters, d-dim_data
    """

    return int(K * d + K * d * (d + 1)/2 + (K - 1))


def akaike_information_criterion(log_likelihood, K, dim_data):
    """
    :return: the badness of fit for the model to the dataset;
    TLDR: the higher this number is, the worse
    """
    return -2 * log_likelihood + 2 * get_free_parameters(K, dim_data)

