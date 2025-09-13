import numpy as np
from scipy.stats import dirichlet
from scipy.special import gammaln, psi  # psi is the logarithmic derivative of the gamma function
from utils import (init_cluster_means, init_precision_matrix, init_priors, init_soft_counts, dirichlet_pdf,
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
        self.niw_posteriors = [NormalInverseWishartDistribution for _ in range(no_clusters)]


    def train(self, num_iterations, x_train, y_train=None):

        dim_data = x_train.shape[0]

        # parameter initialization
        labels, cluster_means = init_cluster_means(x_train, self.no_clusters)
        self.responsibilities = init_soft_counts(self.no_clusters, labels, x_train)
        print(f"self.responsibilities = {self.responsibilities}")
        weighted_means = compute_weighted_mean(x_train, labels, self.no_clusters, self.responsibilities)

        # data_mean, strength_mean (the confidence in the data_mean), degree_of_freedom, precision_matrix
        self.prior_hyperparameters = init_priors(x_train, labels, self.no_clusters)
        precision_matrix = init_precision_matrix(x_train, labels, self.no_clusters)

        for i, posterior in enumerate(self.niw_posteriors):
            posterior.mean = cluster_means[i]
            posterior.strength_mean = 1
            posterior.freedom_degrees = dim_data + 2
            posterior.precision_matrix = precision_matrix

        for i in range(num_iterations):

            self.e_step(x_train)
            self.variational_m_step(x_train, weighted_means)


    def e_step(self, x_train):
        """
        update the responsibilities (i.e. the probability that point x_i belongs to each cluster)
        """
        total_responsibility = np.sum(self.responsibilities[:, :])
        print(f"total_responsibility = {total_responsibility} \n")
        for cluster_idx in range(self.no_clusters):
            for data_idx, row in enumerate(x_train.itertuples()):
                # here the first term is the responsibility of the whole cluster
                cluster_responsibility = np.sum(self.responsibilities[cluster_idx, :])
                log_zi_prob = psi(cluster_responsibility) - psi(total_responsibility)
                log_xi_prob = None

        pass

    def variational_m_step(self, x_train, weighted_means):
        """
        update the probabilities of the weights(from the dirichlet distribution),
        as well as all the parameters of the NormalInverseWishart: strength_mean, degree_of_freedom, precision_matrix
        """

        pass























