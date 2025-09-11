import math
import os.path
import numpy as np
from utils import *
from scipy.linalg import fractional_matrix_power

class NormalDistribution:

    def __init__(self, mean, mixing_weight, dimension, covariance=None):

        self.mean = mean
        if covariance is None:
            self._covariance = init_cov_matrix(dimension)
        else:
            self._covariance = covariance
        self._mixing_weight = mixing_weight
        self.dimension = dimension

    # returns a scalar
    def density_function(self, instance):
        # instance is a real valued vector
        denominator =  ((2 * math.pi) ** self.dimension / 2) * np.sqrt(np.linalg.det(self.covariance))
        diff = instance - self.mean
        cov_inverse = fractional_matrix_power(self.covariance, -1)
        exp_term = math.exp(-1/2 * (diff.transpose() @ cov_inverse @ diff))

        return exp_term / denominator

    @property
    def mixing_weight(self):
        return self._mixing_weight

    @mixing_weight.setter
    def mixing_weight(self, mixing_weight):
        self._mixing_weight = mixing_weight

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        self._covariance = cov


class MixtureModel:

    def __init__(self, no_clusters: int, x_train, y_train):

        self.x_train = x_train
        self.y_train = y_train
        self.K = no_clusters
        self.means = self.x_train.sample(n=self.K, random_state=7, replace=False).to_numpy()
        self.gaussians = [NormalDistribution(self.means[i], 1 / self.K, self.x_train.shape[1]) for i in range(self.K)]
        self.responsibilities = np.zeros((self.K, self.x_train.shape[0])) # how many instances? the number of rows

    def train(self, iterations: int=20, out_means='../models/standard_GMM/means.npy', out_covariances='../models/standard_GMM/cov_matrices.npy',
              out_weights='../models/standard_GMM/mixing_weights.npy'):

        for i in range(iterations):
            print(f"Now at iteration {i}")

            self.e_step()
            self.m_step()

            self.evaluate_performance(self.x_train, self.y_train, training=True)

            # log likelihood should be increasing (it's a negative number)
            print(f"current log_likelihood \n = {self.compute_log_likelihood()}")

        np.save(out_means, self.means)

        cov_matrices = [gaussian.covariance for gaussian in self.gaussians]
        np.save(out_covariances, cov_matrices)

        mixing_weights = [gaussian.mixing_weight for gaussian in self.gaussians]
        np.save(out_weights, mixing_weights)
        return self

    def predict(self, x_test, y_test, out_means='../models/means.npy', out_covariances='../models/cov_matrices.npy',
                out_weights='../models/mixing_weights.npy'):

        if os.path.exists(out_means) and os.path.exists(out_covariances) and os.path.exists(out_weights):
            print(f"Loading the trained parameters \n")
            mixing_weights = np.load(out_weights)
            means = np.load(out_means)
            cov_matrices = np.load(out_covariances)

            self.gaussians = [NormalDistribution(means[i], mixing_weights[i], x_test.shape[0], cov_matrices[i])
                              for i in range(self.K)]
            self.evaluate_performance(x_test, y_test, training=False)

        else:
            print(f"Training the gaussian mixture model first \n")
            self.train()
            self.evaluate_performance(x_test, y_test, training=True)


    def evaluate_performance(self, x, y, training:bool):

        num_classes = y.nunique().iloc[0]
        x_np = x.to_numpy()
        y_np = y.to_numpy().flatten()

        cluster_indices = [[] for _ in range(self.K)]
        for idx, (x, y) in enumerate(zip(x_np, y_np)):

            result_probs = [gaussian.density_function(x) * gaussian.mixing_weight for gaussian in self.gaussians]
            result_probs /= np.sum(result_probs)
            cluster_idx = np.argmax(result_probs)
            cluster_indices[cluster_idx].append((idx, y))

        # assigning the true labels to each gaussian component
        cluster_labels = relabel_data(self.K, cluster_indices, num_classes)

        correct = 0
        for i, cluster in enumerate(cluster_indices):
            for (idx, true_label) in cluster:
                if cluster_labels[i] == true_label:
                    correct += 1

        if training:
            print(f"Training accuracy = {(correct / len(x_np)) * 100}")
        else:
            print(f"Testing accuracy = {(correct / len(x_np)) * 100}")

        return cluster_indices, cluster_labels


    def e_step(self):
        # pdf evaluations for each data point ~ P(z | x, theta)
        pdf_evals = np.zeros((self.K, self.x_train.shape[0]))
        for cluster in range(self.K):
            for row_index, row in enumerate(self.x_train.itertuples()):
                # the first element is the index in the dataframe
                x_values = np.array(row[1:])
                pdf_evals[cluster, row_index] = self.gaussians[cluster].density_function(x_values)

        # update the responsibilities (pdf evaluations weighted by the mixing_weights)
        for cluster in range(self.K):
            for instance_idx in range(self.x_train.shape[0]):
                nominator = self.gaussians[cluster].mixing_weight * pdf_evals[cluster, instance_idx]
                weights = np.array([g.mixing_weight for g in self.gaussians])
                denominator = np.sum(weights * pdf_evals[:, instance_idx])
                self.responsibilities[cluster, instance_idx] = nominator / denominator


    def m_step(self):
        # update the means, a vector from R^d, where d = K (the number of clusters)
        for cluster in range(self.K):
            nominator = np.zeros(self.x_train.shape[1])

            for row_index, row in enumerate(self.x_train.itertuples()):
                x_values = np.array(row[1:])
                nominator += self.responsibilities[cluster, row_index] * x_values  # gamma_ij * x_i
            self.means[cluster] = nominator / np.sum(self.responsibilities[cluster, :])

        # update mixing weights
        new_weights = [ float(np.sum(self.responsibilities[cluster, :])) / self.x_train.shape[0] for cluster in range(self.K)]
        for i in range(self.K):
            self.gaussians[i].mixing_weight = new_weights[i]

        # update covariance matrix
        for cluster_idx in range(len(self.gaussians)):
            nominator = np.zeros((self.x_train.shape[1], self.x_train.shape[1]))
            denominator = np.zeros((self.x_train.shape[1], self.x_train.shape[1]))

            for row_index, row in enumerate(self.x_train.itertuples()):
                data_diff = np.array(row[1:]) - self.means[cluster_idx]
                data_diff = data_diff.reshape(-1, 1) # (d, 1)

                nominator += self.responsibilities[cluster_idx, row_index] * (data_diff @ data_diff.transpose())
                denominator += self.responsibilities[cluster_idx , row_index]
            self.gaussians[cluster_idx].covariance = nominator / denominator


    def compute_log_likelihood(self):

        log_likelihood = 0
        for row in self.x_train.itertuples():
            prob_sum = 0
            for gaussian in self.gaussians:
                x_values = np.array(row[1:])
                prob_sum += gaussian.mixing_weight * gaussian.density_function(x_values)
            log_likelihood += np.log(prob_sum + 1e-12)

        return log_likelihood