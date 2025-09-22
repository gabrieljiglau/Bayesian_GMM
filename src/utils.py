import os
import numpy as np
import pandas as pd
from numpy import shape
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import invwishart, multivariate_normal
from scipy.linalg import fractional_matrix_power
from scipy.special import gammaln

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
def sample_covariance(degrees_of_freedom, precision_matrix):
    return invwishart.rvs(df=degrees_of_freedom, scale=np.linalg.inv(precision_matrix))

def sample_mean(miu_point_estimate, covariance_matrix, strength_mean):
    covariance_matrix = np.array(covariance_matrix)
    covariance_matrix /= strength_mean
    return multivariate_normal(mean=miu_point_estimate, cov=covariance_matrix).rvs()

def hyperparameters_exist(out_strength_means='../models/bayesian_GMM/strength_means.npy',
              out_freedom_degrees='../models/bayesian_GMM/freedom_degrees.npy',
              out_means='../models/bayesian_GMM/means.npy',
              out_precision_matrices='../models/bayesian_GMM/precision_matrices.npy',
              out_weights='../models/bayesian_GMM/mixing_weights.npy'):
    return (os.path.exists(out_strength_means) and os.path.exists(out_freedom_degrees) and os.path.exists(out_means)
            and os.path.exists(out_precision_matrices) and os.path.exists(out_weights))


def _weighted_mean(x_train, no_clusters, soft_counts):

    weighted_means = np.zeros((no_clusters, x_train.shape[1]))
    for k in range(no_clusters):
        norm_term = np.sum(soft_counts[k, :])
        for i, row in enumerate(x_train.itertuples()):
            weighted_means[k,:] += (np.array(row[1:]) * soft_counts[k, i])

        weighted_means[k, :] /= norm_term
    return weighted_means

def _data_mean(x_train):

    data_mean = np.zeros(x_train.shape[1])
    for i in range(x_train.shape[1]):
        dimension_mean = x_train.iloc[:, i].mean()
        data_mean[i] = dimension_mean
    return data_mean

def init_priors(no_clusters, x_train, dim_data, epsilon):

    """
    return priors: prior on weights, data_mean, strength_mean (the confidence in the data_mean),
    degree_of_freedom, scale_matrix
    """

    return [np.ones(no_clusters), _data_mean(x_train), 1, x_train.shape[1] + 1,
            (np.eye(dim_data) * epsilon) + np.eye(dim_data)]

def gaussian_pdf(instance, dim_data, covariance, mean):
    # instance is a real valued vector
    denominator =  (2 * np.pi) ** (dim_data / 2) * np.sqrt(np.linalg.det(covariance))
    diff = instance - mean
    cov_inverse = fractional_matrix_power(covariance, -1)
    exp_term = np.exp(-1/2 * (diff.transpose() @ cov_inverse @ diff))

    return exp_term / denominator

def log_det_cholesky(matrix):

    cholesky_det = np.linalg.cholesky(matrix)
    return np.sum(np.log(np.diag(cholesky_det))) * 2

def student_t_pdf(x_in, degrees_of_freedom, dim_data, cluster_mean, scale_matrix):

    nominator = gammaln((degrees_of_freedom + dim_data) / 2)
    denominator = gammaln(degrees_of_freedom / 2) * ((degrees_of_freedom * np.pi) ** dim_data / 2)
    denominator *= np.sqrt(np.linalg.det(scale_matrix))

    diff = (x_in - cluster_mean).reshape(-1, 1)
    free_term = (1 + (1 / degrees_of_freedom) * (diff.transpose() @ np.linalg.inv(scale_matrix) @ diff))
    # print(f"degrees_of_freedom = {degrees_of_freedom}, dim_data = {dim_data}")
    free_term **= ((degrees_of_freedom + dim_data) / -2)

    return (nominator / denominator) * free_term

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

