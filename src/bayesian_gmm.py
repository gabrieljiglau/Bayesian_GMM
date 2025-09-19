import os
import numpy as np
from scipy.stats import dirichlet
from sklearn.metrics import accuracy_score
from scipy.special import logsumexp, psi  # psi is the logarithmic derivative of the gamma function
from utils import init_cluster_means, init_priors, init_responsibilities, log_det_cholesky, map_clusters, \
                   compute_log_likelihood, akaike_information_criterion, gaussian_pdf


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

    def __init__(self, no_clusters: int, epsilon=1e-6):

        self.no_clusters = no_clusters
        self.weights = [1 for _ in range(no_clusters)]
        self.dirichlet_prior = np.ones(self.no_clusters)
        self.prior_hyperparameters = []
        self.responsibilities = []
        self.niw_posteriors = [NormalInverseWishartDistribution() for _ in range(no_clusters)]
        self.epsilon = epsilon


    def train(self, num_iterations, x_train, y_train, eps=1e-6, out_means='../models/bayesian_GMM/means.npy',
              out_covariances='../models/bayesian_GMM/cov_matrices.npy', out_weights='../models/bayesian_GMM/mixing_weights.npy'):

        # parameter initialization
        dim_data = x_train.shape[1]
        labels, cluster_means = init_cluster_means(x_train, self.no_clusters)

        self.responsibilities = init_responsibilities(self.no_clusters, labels, x_train)
        # priors: data_mean, strength_mean (the confidence in the data_mean), degree_of_freedom, precision_matrix
        self.prior_hyperparameters = init_priors(x_train, labels, self.no_clusters, self.responsibilities, dim_data, self.epsilon)

        for i, posterior in enumerate(self.niw_posteriors):
            posterior.cluster_mean = cluster_means[i]
            posterior.strength_mean = self.prior_hyperparameters[1]
            posterior.freedom_degrees = self.prior_hyperparameters[2]
            posterior.precision_matrix = self.prior_hyperparameters[3]

        means = []
        cov_matrices = []
        for i in range(num_iterations):
            print(f"Now at iteration {i}")

            self.e_step(x_train, dim_data)
            self.variational_m_step(x_train, dim_data)
            acc = self.evaluate_performance(x_train, y_train)
            print(f"accuracy = {acc}")

            means = [posterior.cluster_mean for posterior in self.niw_posteriors]
            cov_matrices = [np.linalg.inv(posterior.precision_matrix) for posterior in self.niw_posteriors]
            log_likelihood = compute_log_likelihood(x_train, dim_data, means, cov_matrices, self.weights)
            print(f"log_likelihood = {log_likelihood}")

            # choose the model that minimizes aic
            aic = akaike_information_criterion(log_likelihood, self.no_clusters, dim_data)
            print(f"akaike_information_criterion = {aic}")

        np.save(out_means, means)
        np.save(out_covariances, cov_matrices)
        np.save(out_weights, self.weights)

    def predict(self, x_test, y_test,
                out_means='../models/bayesian_GMM/means.npy',
                out_covariances='../models/bayesian_GMM/cov_matrices.npy',
                out_weights='../models/bayesian_GMM/mixing_weights.npy'):

        if os.path.exists(out_means) and os.path.exists(out_covariances) and os.path.exists(out_weights):
            print(f"Loading the trained parameters \n")

            mixing_weights = np.load(out_weights)
            means = np.load(out_means)
            cov_matrices = np.load(out_covariances)

            acc = self.evaluate_performance(x_test, y_test, mixing_weights, means, cov_matrices, training=False)
            print(f"accuracy on test data = {acc}")
        else:
            print(f"The bayesian mixture model needs to be trained first !\n")


    def evaluate_performance(self, x, y, mixing_weights=None, means=None, cov_matrices=None, training=True):

        x_np = x.to_numpy()
        y_np = y.to_numpy().flatten()

        cluster_assignments = []

        if training:
            for idx in range(x_np.shape[0]):
                result_probs = self.responsibilities[:, idx]
                result_probs /= np.sum(result_probs)
                cluster_idx = np.argmax(result_probs)
                cluster_assignments.append(cluster_idx)
        else:

            ## wrong, because I need the full distribution over the parameters (miu, sigma, pi_k) in order to make inferences
            for idx, (x_in, y_true) in enumerate(zip(x_np, y_np)):
                result_probs = []
                result_instance = []
                for k in range(self.no_clusters):
                    result_instance.append(gaussian_pdf(x_in, len(x_in), cov_matrices[k], means[k]) * mixing_weights[k])
                result_probs.append(result_instance)
                result_probs /= np.sum(result_probs)
                cluster_idx = np.argmax(result_probs)
                cluster_assignments.append(cluster_idx)

        # assigning the true labels to each gaussian component
        cluster_labels, mapping = map_clusters(y_np, cluster_assignments, self.no_clusters)
        acc = accuracy_score(y_np, cluster_labels)

        return acc


    def compute_log_xi(self, cluster_idx, degrees_of_freedom, cluster_mean, data_in, no_features):

        first_term = 0
        second_term = 0
        for dimension in range(1, no_features + 1):
            first_term += psi((degrees_of_freedom + 1 - dimension) / 2)

        # print(f"self.niw_posteriors[cluster_idx].precision_matrix = {self.niw_posteriors[cluster_idx].precision_matrix} \n")

        precision_matrix = self.niw_posteriors[cluster_idx].precision_matrix + np.eye(no_features) * self.epsilon
        log_det = log_det_cholesky(precision_matrix)

        first_term += no_features * np.log(2.0) + log_det
        first_term /= 2.0
        first_term -= (no_features / 2.0) * np.log(np.pi * 2.0)

        diff = data_in - cluster_mean
        diff = diff.reshape(-1, 1)

        second_term += diff.transpose() @ (degrees_of_freedom * precision_matrix) @ diff
        trace = (degrees_of_freedom * np.trace(precision_matrix)) / self.niw_posteriors[cluster_idx].strength_mean
        second_term += trace

        return first_term - second_term / 2


    def e_step(self, x_train, no_features):
        """
        update the responsibilities (i.e. the probability that point x_i belongs to each cluster)
        """
        total_weight = np.sum(self.weights)
        log_responsibilities = np.zeros((self.no_clusters, x_train.shape[0]))

        for k in range(self.no_clusters):
            for data_idx, row in enumerate(x_train.itertuples()):
                niu = self.niw_posteriors[k].freedom_degrees
                cluster_mean = self.niw_posteriors[k].cluster_mean
                cluster_weight = self.weights[k]

                log_zi_prob = psi(cluster_weight) - psi(total_weight)
                log_xi_prob = self.compute_log_xi(k, niu, cluster_mean, row[1:], no_features)

                responsibility = log_zi_prob + log_xi_prob
                log_responsibilities[k, data_idx] = responsibility

        log_responsibilities -= logsumexp(log_responsibilities, axis=0, keepdims=True)
        self.responsibilities = np.exp(log_responsibilities)

        # sanity check, should sum up to 1
        # print(f"self.responsibilities.sum(axis=) = {self.responsibilities.sum(axis=0)}")


    def build_sample_covariance(self, x_train, soft_counts, weighted_means):

        sample_covariance = []
        for k in range(self.no_clusters):
            cov_dim = 0
            for idx, row in enumerate(x_train.itertuples()):
                diff = np.array(row[1:]) - weighted_means[k]
                diff = diff.reshape(-1, 1)
                cov_dim += (diff @ diff.transpose()) * self.responsibilities[k, idx]

            # print(f"cov_dim = {cov_dim}")
            # print(f"soft_counts[k] = {soft_counts[k]}")  # should sum up to 1
            sample_covariance.append(cov_dim / soft_counts[k])

        return np.array(sample_covariance)


    def build_coefficient(self, soft_count, weighted_mean):

        # priors: data_mean, strength_mean (the confidence in the data_mean), degree_of_freedom, precision_matrix
        strength_mean = self.prior_hyperparameters[1]
        first_term = (strength_mean * soft_count) / (strength_mean + soft_count)
        diff = (weighted_mean - self.prior_hyperparameters[0]).reshape(-1, 1)

        return first_term * (diff @ diff.transpose())


    def variational_m_step(self, x_train, dim_data):
        """
        update the probabilities of the weights(from the dirichlet distribution),
        as well as all the parameters of the NormalInverseWishart: strength_mean, degree_of_freedom, precision_matrix
        """

        soft_counts = np.zeros(self.no_clusters)
        for k in range(self.no_clusters):
            soft_counts[k] = np.sum(self.responsibilities[k, :])

        # weights update
        for k in range(self.no_clusters):
            self.weights[k] = self.dirichlet_prior[k] + soft_counts[k]

        # print(f"self.weights = {self.weights}")

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

            # priors: data_mean, strength_mean (the confidence in the data_mean), degree_of_freedom, precision_matrix
            self.niw_posteriors[k].strength_mean = self.prior_hyperparameters[1] + soft_counts[k]
            self.niw_posteriors[k].freedom_degrees = self.prior_hyperparameters[2] + soft_counts[k]

            mean_contribution = prior_strength_mean * prior_data_mean
            empirical_contribution = soft_counts[k] * weighted_means[k, :]
            self.niw_posteriors[k].cluster_mean = (mean_contribution + empirical_contribution) / self.niw_posteriors[k].strength_mean

            observed_data = soft_counts[k] * sample_covariance[k]
            uncertainty_coefficient = self.build_coefficient(soft_counts[k], weighted_means[k, :])

            precision_matrix_inv = np.linalg.inv(self.prior_hyperparameters[3])
            new_precision = precision_matrix_inv + observed_data + uncertainty_coefficient
            self.niw_posteriors[k].precision_matrix = np.linalg.inv(new_precision)

