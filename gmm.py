import math
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from utils import *

class NormalDistribution:

    def __init__(self, mean, mixing_weight, dimension, covariance=None):
        self.mean = mean

        if not covariance:
            self._covariance = init_cov_matrix(dimension)
        else:
            self._covariance = covariance

        self.mixing_weight = mixing_weight
        self.dimension = dimension

    def density_function(self, data_point):
        # data_point is an observation, a real valued vector
        data_point = np.array(data_point)

        dim = self.dimension / 2
        sqrt_cov = fractional_matrix_power(self.covariance, 1/2)
        denominator =   ((2 * math.pi) ** dim) * np.abs(sqrt_cov)
        first_term = 1 / denominator

        diff = data_point - self.mean
        pow_cov = fractional_matrix_power(self.covariance, -1)
        second_term = math.exp(-1/2 * (diff.transpose() * pow_cov * diff))

        return first_term * second_term

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        self._covariance = cov


class MixtureModel:

    def __init__(self, no_clusters, dataset: pd.DataFrame):

        self.dim = dataset.shape[1]
        self.mixing_weights = np.ones(no_clusters) / no_clusters
        self.means =  dataset.sample(n=no_clusters, replace=False).to_numpy()
        self.gaussians = [NormalDistribution(self.means[i], self.mixing_weights[i], self.dim) for i in range(no_clusters)]


    def fit(self):
        pass


    def e_step(self):
        pass

    def m_step(self):
        pass










