import numpy as np
from scipy.stats import dirichlet
from scipy.special import gammaln, psi  # psi is the logarithmic derivative of the gamma function
from utils import dirichlet_pdf

class NormalInverseWishartDistribution:

    def __init__(self, strength_mean=None, mean=None, freedom_degrees=None, precision_matrix=None):

        self._strength_mean = 1 if strength_mean is None else self._strength_mean = strength_mean
        self._mean = 0 if mean is None else self._mean = mean
        self._freedom_degrees = 0 if freedom_degrees is None else self._freedom_degrees = freedom_degrees
        self._precision_matrix = [] if precision_matrix is None else self._precision_matrix = precision_matrix

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

    def __init__(self, no_clusters: int, mean_strength_prior, mean_prior, freedom_degree_prior, precision_matrix_prior):
        self.no_clusters = no_clusters
        self.weights = [(1 / no_clusters) for _ in range(no_clusters)]
        self.weights_dist = dirichlet_pdf(self.weights)
        self.responsibilities = []
        self.priors = [NormalInverseWishartDistribution for _ in range(no_clusters)]

    def train(self, num_iterations, x_train, y_train=None):

        # parameter initialization
        dim_data = x_train.shape[0]
        self.responsibilities = np.array((self.no_clusters, dim_data))
        for prior in self.priors:
            prior.freedom_degrees = dim_data + 2

        for i in range(num_iterations):


            pass

























