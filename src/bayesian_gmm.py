import numpy as np
from scipy.stats import dirichlet
from scipy.special import logsumexp, psi  # psi is the logarithmic derivative of the gamma function
from utils import init_cluster_means, init_priors, init_responsibilities, log_det_cholesky


class NormalInverseWishartDistribution:

    def __init__(self, strength_mean=None, mean=None, freedom_degrees=None, precision_matrix=None):
        self._strength_mean = strength_mean
        self._cluster_mean = mean
        self._freedom_degrees = freedom_degrees
        self._precision_matrix = precision_matrix

    @property
    def strength_mean(self):
        return self._strength_mean

    @strength_mean.setter
    def strength_mean(self, new_strength_mean):
        self._strength_mean = new_strength_mean

    @property
    def cluster_mean(self):
        return self._cluster_mean

    @cluster_mean.setter
    def cluster_mean(self, new_mean):
        self._cluster_mean = new_mean

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

    def __init__(self, no_clusters: int, epsilon=1e-14):

        self.no_clusters = no_clusters
        self.weights = [(1 / no_clusters) for _ in range(no_clusters)]
        self.dirichlet_prior = np.ones(self.no_clusters)
        self.prior_hyperparameters = []
        self.responsibilities = []
        self.niw_posteriors = [NormalInverseWishartDistribution() for _ in range(no_clusters)]
        self.epsilon = epsilon


    def train(self, num_iterations, x_train, y_train=None, eps=1e-6):

        # parameter initialization
        dim_data = x_train.shape[1]
        labels, cluster_means = init_cluster_means(x_train, self.no_clusters)

        self.responsibilities = init_responsibilities(self.no_clusters, labels, x_train)
        # priors: data_mean, strength_mean (the confidence in the data_mean), degree_of_freedom, precision_matrix
        self.prior_hyperparameters = init_priors(x_train, labels, self.no_clusters, self.responsibilities, dim_data)

        for i, posterior in enumerate(self.niw_posteriors):
            posterior.cluster_mean = cluster_means[i]
            posterior.strength_mean = self.prior_hyperparameters[1]
            posterior.freedom_degrees = self.prior_hyperparameters[2]
            posterior.precision_matrix = self.prior_hyperparameters[3][i] + np.eye(dim_data) * eps

        for i in range(num_iterations):
            print(f"Now at iteration {i}")

            self.e_step(x_train, dim_data)
            self.variational_m_step(x_train, dim_data)
            print(f"self.weights = {self.weights}")


    def predict(self, x_test, y_test):
        pass


    def compute_log_xi(self, cluster_idx, degrees_of_freedom, cluster_mean, data_in, no_features):

        first_term = 0
        second_term = 0
        for dimension in range(1, no_features + 1):
            first_term += psi((degrees_of_freedom + 1 - dimension) / 2)

        # print(f"self.niw_posteriors[cluster_idx].precision_matrix = {self.niw_posteriors[cluster_idx].precision_matrix} \n")

        log_det = log_det_cholesky(self.niw_posteriors[cluster_idx].precision_matrix + np.eye(no_features) * 1e-6)

        first_term += no_features * np.log(2) + log_det
        first_term /= 2
        first_term -= (no_features / 2) * np.log(np.pi * 2)

        diff = data_in - cluster_mean
        diff = diff.reshape(-1, 1)

        precision_matrix = self.niw_posteriors[cluster_idx].precision_matrix + np.eye(no_features) * 1e-6

        second_term += diff.transpose() @ (degrees_of_freedom * precision_matrix) @ diff
        second_term += no_features / self.niw_posteriors[cluster_idx].strength_mean
        second_term /= 2

        return first_term - second_term


    def e_step(self, x_train, no_features):
        """
        update the responsibilities (i.e. the probability that point x_i belongs to each cluster)
        """
        total_weight = np.sum(self.weights)
        print(f"total_weight = {total_weight} \n")
        # self.weights = self.weights / total_weight * self.no_clusters
        #print(f"inainte de for self.responsibilities = {self.responsibilities}")
        log_responsibilities = np.zeros((self.no_clusters, x_train.shape[0]))
        for k in range(self.no_clusters):
            for data_idx, row in enumerate(x_train.itertuples()):
                niu = self.niw_posteriors[k].freedom_degrees
                cluster_mean = self.niw_posteriors[k].cluster_mean
                cluster_weight = self.weights[k]
                # print(f"cluster_weight = {cluster_weight}")
                log_zi_prob = psi(cluster_weight) - psi(total_weight)
                log_xi_prob = self.compute_log_xi(k, niu, cluster_mean, row[1:], no_features)
                # print(f"log_zi_prob = {log_zi_prob}")
                # print(f"log_xi_prob = {log_xi_prob}")
                responsibility = log_zi_prob + log_xi_prob
                log_responsibilities[k, data_idx] = responsibility

        #print(f"before normalization = {log_responsibilities}")
        # normalization and subtracting the max to prevent overflows

        log_responsibilities -= logsumexp(log_responsibilities, axis=0, keepdims=True)
        self.responsibilities = np.exp(log_responsibilities)

        # print(f"after normalization: {self.responsibilities}")
        #print(f"self.responsibilities.sum(axis=) = {self.responsibilities.sum(axis=0)}")


    def build_sample_covariance(self, x_train, soft_counts, weighted_means):

        sample_covariance = []
        for k in range(self.no_clusters):
            cov_dim = 0
            for idx, row in enumerate(x_train.itertuples()):
                diff = np.array(row[1:]) - weighted_means[k]
                diff = diff.reshape(-1, 1)
                cov_dim += (diff @ diff.transpose()) * self.responsibilities[k, idx]

            # print(f"cov_dim = {cov_dim}")
            # print(f"soft_counts[k] = {soft_counts[k]}")  # should sumt up to 1
            sample_covariance.append(cov_dim / soft_counts[k])

        return np.array(sample_covariance)


    def build_coefficient(self, k, soft_count, weighted_mean):

        strength_mean = self.prior_hyperparameters[1]
        first_term = (strength_mean * soft_count) / (strength_mean + soft_count)
        diff = weighted_mean - self.prior_hyperparameters[0]
        diff = diff.reshape(-1, 1)

        return (diff @ diff.transpose()) * first_term


    def variational_m_step(self, x_train, dim_data):
        """
        update the probabilities of the weights(from the dirichlet distribution),
        as well as all the parameters of the NormalInverseWishart: strength_mean, degree_of_freedom, precision_matrix
        """

        # print(f"in m-step: \n self.responsibilities = {self.responsibilities}")

        soft_counts = np.zeros(self.no_clusters)
        for k in range(self.no_clusters):
            soft_counts[k] = np.sum(self.responsibilities[k, :])

        # weights update
        for k in range(self.no_clusters):
            self.weights[k] = self.dirichlet_prior[k] + soft_counts[k]

        # print(f"soft_counts = {soft_counts} \n ")
        print(f"self.weights = {self.weights}")

        # new weighted mean
        weighted_means = np.zeros((self.no_clusters, x_train.shape[1]))
        for k in range(self.no_clusters):
            for idx, row in enumerate(x_train.itertuples()):
                weighted_means[k, :] += self.responsibilities[k, idx] * np.array(row[1:])
            weighted_means[k, :] /= soft_counts[k]

        sample_covariance = self.build_sample_covariance(x_train, soft_counts, weighted_means)

        # NIW parameter updates
        for k in range(self.no_clusters):

            prior_data_mean = self.prior_hyperparameters[0]
            prior_strength_mean = self.prior_hyperparameters[1]
            self.niw_posteriors[k].strength_mean += soft_counts[k]
            self.niw_posteriors[k].freedom_degrees += soft_counts[k]

            mean_contribution = prior_strength_mean * prior_data_mean
            empirical_contribution = soft_counts[k] * weighted_means[k, :]
            self.niw_posteriors[k].cluster_mean = (mean_contribution + empirical_contribution) / self.niw_posteriors[k].strength_mean

            observed_data = soft_counts[k] * sample_covariance[k]
            uncertainty_coefficient = self.build_coefficient(k, weighted_means[k, :], self.prior_hyperparameters[0])

            precision_matrix_inv = np.linalg.inv(self.prior_hyperparameters[3][k])
            new_precision = precision_matrix_inv + observed_data + uncertainty_coefficient + 1e-6 * np.eye(dim_data)
            self.niw_posteriors[k].precision_matrix = np.linalg.inv(new_precision)



