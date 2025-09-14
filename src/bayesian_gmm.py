import numpy as np
from scipy.stats import dirichlet
from scipy.special import gammaln, psi  # psi is the logarithmic derivative of the gamma function
from utils import (init_cluster_means, init_precision_matrix, init_priors, init_responsibilities, dirichlet_pdf,
                   compute_weighted_mean)

class NormalInverseWishartDistribution:

    def __init__(self, strength_mean=None, mean=None, freedom_degrees=None, precision_matrix=None):

        self._strength_mean = strength_mean
        self._mean = mean
        self._freedom_degrees = freedom_degrees
        self._precision_matrix = precision_matrix

    @property
    def strength_mean(self):
        return self._strength_mean

    @strength_mean.setter
    def strength_mean(self, new_strength_mean):
        self._strength_mean = new_strength_mean

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, new_mean):
        self._mean = new_mean

    @property
    def precision_matrix(self):
        return self._precision_matrix

    @precision_matrix.setter
    def precision_matrix(self, new_precision_matrix):
        self._precision_matrix = new_precision_matrix

    @property
    def freedom_degrees(self):
        return self._freedom_degrees

    @freedom_degrees.setter
    def freedom_degrees(self, new_freedom_degrees):
        self._freedom_degrees = new_freedom_degrees


class BayesianMixtureModel:

    def __init__(self, no_clusters: int):

        self.no_clusters = no_clusters
        self.weights = [(1 / no_clusters) for _ in range(no_clusters)]
        self.weights_pdf = dirichlet_pdf(self.weights)
        self.responsibilities = []
        self.prior_hyperparameters = []
        self.means = []
        self.niw_posteriors = [NormalInverseWishartDistribution for _ in range(no_clusters)]


    def train(self, num_iterations, x_train, y_train=None):

        # parameter initialization
        dim_data = x_train.shape[1]
        labels, cluster_means = init_cluster_means(x_train, self.no_clusters)
        self.responsibilities = init_responsibilities(self.no_clusters, labels, x_train)
        # print(f"self.responsibilities = {self.responsibilities}")
        self.means = compute_weighted_mean(x_train, labels, self.no_clusters, self.responsibilities)

        # data_mean, strength_mean (the confidence in the data_mean), degree_of_freedom, precision_matrix
        self.prior_hyperparameters = init_priors(x_train, labels, self.no_clusters, self.responsibilities)

        for i, posterior in enumerate(self.niw_posteriors):
            posterior.mean = cluster_means[i]
            posterior.strength_mean = self.prior_hyperparameters[2] + 1
            posterior.freedom_degrees = dim_data + 2
            posterior.precision_matrix = self.prior_hyperparameters[3]

        for i in range(num_iterations):

            self.e_step(x_train, dim_data, cluster_means)
            self.variational_m_step(x_train)


    def predict(self, x_test, y_test):
        pass


    def compute_log_xi(self, cluster_idx, cluster_responsibility, cluster_mean, data_in, no_features):

        first_term = 0
        second_term = 0
        for dimension in range(1, no_features):
            first_term += psi(cluster_responsibility + 1 - dimension)

        first_term += no_features * np.log(2) + np.log(np.linalg.det(self.niw_posteriors[cluster_idx].precision_matrix))
        first_term /= 2

        first_term -= no_features / 2 * np.log(np.pi * 2)

        diff = data_in - cluster_mean
        diff_transpose = diff.reshape(-1, 1).tranpose()

        second_term += diff_transpose @ self.niw_posteriors[cluster_idx].precision_matrix @ diff
        second_term *= self.niw_posteriors[cluster_idx].freedom_degrees
        second_term += no_features / self.niw_posteriors[cluster_idx].strength_mean

        return first_term - second_term / 2


    def e_step(self, x_train, no_features, cluster_means):
        """
        update the responsibilities (i.e. the probability that point x_i belongs to each cluster)
        """
        total_responsibility = np.sum(self.responsibilities)
        # print(f"total_responsibility = {total_responsibility} \n")
        for k in range(self.no_clusters):
            for data_idx, row in enumerate(x_train.itertuples()):
                cluster_responsibility = np.sum(self.responsibilities[k, :])
                log_zi_prob = psi(cluster_responsibility) - psi(total_responsibility)
                log_xi_prob = self.compute_log_xi(k, cluster_responsibility, cluster_means[k], row[1:], no_features)
                self.responsibilities[k, data_idx] = log_zi_prob + log_xi_prob

        # subtracting the max to prevent overflows
        max_probs = []
        for k in range(self.no_clusters):
            max_probs.append(np.argmax(self.responsibilities[k, :]))
            self.responsibilities[k, :] -= max_probs[k]

        self.responsibilities = np.exp(self.responsibilities)

        # normalization
        for k in range(self.no_clusters):
            normalization_term = np.sum(np.exp(self.responsibilities[k, :]))
            for data_idx in range(x_train.shape[0]):
                self.responsibilities[k, data_idx] = np.exp(self.responsibilities[k, data_idx]) / normalization_term


    def build_sample_covariance(self, x_train, soft_counts):

        sample_covariance = np.zeros(self.no_clusters)
        for k in range(self.no_clusters):
            cov_dim = 0
            for idx, row in enumerate(x_train.itertuples()):
                diff = np.array(row[1:]) - self.means[k]
                diff_transpose = diff.reshape(-1, 1).transpose()

                cov_dim += (diff @ diff_transpose) * self.responsibilities[k, idx]

            sample_covariance[k] = cov_dim

        return sample_covariance


    def build_coefficient(self, k, soft_count, old_mean):

        first_term = ((self.niw_posteriors[k].strength_mean - soft_count) * soft_count) / self.niw_posteriors[k].strength_mean
        diff = self.niw_posteriors[k].mean - old_mean
        diff_transpose = diff.reshape(-1, 1).transpose()

        return (diff @ diff_transpose) * first_term


    def variational_m_step(self, x_train):
        """
        update the probabilities of the weights(from the dirichlet distribution),
        as well as all the parameters of the NormalInverseWishart: strength_mean, degree_of_freedom, precision_matrix
        """

        soft_counts = np.zeros(self.no_clusters)
        for k in range(self.no_clusters):
            soft_counts[k] = np.sum(self.responsibilities[k, :])

        # weights update
        for k in range(self.no_clusters):
            self.weights[k] += soft_counts[k]

        # new weighted mean
        weighted_means = np.zeros((self.no_clusters, x_train.shape[1]))
        for k in range(self.no_clusters):
            for idx, row in enumerate(x_train.itertuples()):
                weighted_means[k, :] += self.responsibilities[k, idx] * row[1:]

            weighted_means /= soft_counts[k]

        self.means = weighted_means

        sample_covariance = self.build_sample_covariance(x_train, soft_counts)

        # NIW parameter updates
        for k in range(self.no_clusters):
            self.niw_posteriors[k].strength_mean += soft_counts[k]
            self.niw_posteriors[k].freedom_degrees += soft_counts[k]

            old_mean = self.niw_posteriors[k].mean
            mean_contribution = (self.niw_posteriors[k].strength_mean - soft_counts[k]) * old_mean
            data_contribution = soft_counts[k] * self.means[k, :]

            self.niw_posteriors[k].mean = (mean_contribution + data_contribution) / self.niw_posteriors[k].strength_mean

            observed_data = soft_counts[k] * sample_covariance[k]
            uncertainty_coefficient = self.build_coefficient(k, soft_counts[k], old_mean)
            self.niw_posteriors[k].precision_matrix = (self.niw_posteriors[k].precision_matrix + observed_data +
                                                       uncertainty_coefficient)



