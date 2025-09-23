import os
import numpy as np
import arviz as az
from scipy.stats import dirichlet
from sklearn.metrics import accuracy_score
from scipy.special import logsumexp, psi  # psi is the logarithmic derivative of the gamma function
from dataclasses import dataclass
from utils import *

@dataclass
class PriorHyperparameters:
        data_mean: np.ndarray
        strength_mean: int
        degrees_of_freedom: int
        scale_matrix: np.ndarray
        dirichlet_prior: np.ndarray


@dataclass
class PosteriorParameters:
        strength_mean: np.ndarray
        cluster_mean: np.ndarray
        degrees_of_freedom: int
        scale_matrix: np.ndarray

class BayesianMixtureModel:

    def __init__(self, no_clusters: int, priors = None,epsilon=1e-6):

        self.no_clusters = no_clusters

        if priors is not None:
            self.priors = priors
        else:
            self.priors = object.__new__(PriorHyperparameters)

        self.posteriors = [object.__new__(PosteriorParameters) for _ in range (no_clusters)]
        self.responsibilities = []
        self.epsilon = epsilon
        self.dirichlet_params = [1 / no_clusters for _ in range (no_clusters)]  # for the dirichlet distribution


    def train(self, num_iterations, x_train, y_train,
              out_strength_means='../models/bayesian_GMM/strength_means.npy',
              out_degrees_of_freedom='../models/bayesian_GMM/degrees_of_freedom.npy',
              out_means='../models/bayesian_GMM/means.npy',
              out_scale_matrices='../models/bayesian_GMM/scale_matrices.npy',
              out_weights='../models/bayesian_GMM/mixing_weights.npy'):

        dim_data = x_train.shape[1]
        self.responsibilities = np.zeros((self.no_clusters, x_train.shape[0]))

        # priors: prior on weights, data_mean, strength_mean (the confidence in the data_mean), degree_of_freedom, scale_matrix
        fixed_priors = init_priors(self.no_clusters, x_train, dim_data, self.epsilon)
        self.priors.dirichlet_prior = fixed_priors[0]
        self.priors.data_mean = fixed_priors[1]
        self.priors.strength_mean = fixed_priors[2]
        self.priors.degrees_of_freedom = fixed_priors[3]
        self.priors.scale_matrix = fixed_priors[4]

        cluster_means = x_train.sample(n=3, random_state=7, replace=False).to_numpy()
        for i in range(self.no_clusters):
            self.posteriors[i].strength_mean = self.priors.strength_mean
            self.posteriors[i].cluster_mean = cluster_means[i]
            self.posteriors[i].degrees_of_freedom = self.priors.degrees_of_freedom
            self.posteriors[i].scale_matrix = self.priors.scale_matrix

        scale_matrices = []
        strength_means = []
        degrees_of_freedom = []
        cluster_means = []
        for i in range(num_iterations):
            print(f"Now at iteration {i}")

            self.variational_e_step(x_train, dim_data)
            self.variational_m_step(x_train, dim_data)

            cluster_means = [posterior.cluster_mean for posterior in self.posteriors]
            scale_matrices = [posterior.scale_matrix for posterior in self.posteriors]
            strength_means = [posterior.strength_mean for posterior in self.posteriors]
            degrees_of_freedom = [posterior.degrees_of_freedom for posterior in self.posteriors]

            acc = self.evaluate_performance(x_train, y_train, self.dirichlet_params, strength_means, cluster_means,
                                            degrees_of_freedom, scale_matrices)
            print(f"accuracy = {acc}")

            log_likelihood = compute_log_likelihood(x_train, dim_data, cluster_means, scale_matrices,
                                                    self.dirichlet_params / np.sum(self.dirichlet_params))
            print(f"log_likelihood = {log_likelihood}")

            # choose the model that minimizes aic
            aic = akaike_information_criterion(log_likelihood, self.no_clusters, dim_data)
            print(f"akaike_information_criterion = {aic}")
            print(f"dirichlet_params = {self.dirichlet_params}")
            print(f"actual mixing weights = {self.dirichlet_params / np.sum(self.dirichlet_params)}")

            # print(f"self.responsibilities = {self.responsibilities[0]}")

        print(f"weights = {self.dirichlet_params}")

        np.save(out_strength_means, strength_means)
        np.save(out_degrees_of_freedom, degrees_of_freedom)
        np.save(out_means, cluster_means)
        np.save(out_scale_matrices, scale_matrices)
        np.save(out_weights, self.dirichlet_params)

    def predict(self, x_test, y_test, num_samples=30,
                out_strength_means='../models/bayesian_GMM/strength_means.npy',
                out_degrees_of_freedom='../models/bayesian_GMM/degrees_of_freedom.npy',
                out_means='../models/bayesian_GMM/means.npy',
                out_scale_matrices='../models/bayesian_GMM/scale_matrices.npy',
                out_weights='../models/bayesian_GMM/mixing_weights.npy'):

        if hyperparameters_exist(out_strength_means, out_degrees_of_freedom, out_means, out_scale_matrices, out_weights):
            print(f"Loading the trained parameters \n")

            # pe = single point_estimate (the output from MAP parameter estimation)
            pe_strength_means = np.load(out_strength_means)
            pe_degrees_of_freedom = np.load(out_degrees_of_freedom)
            pe_means = np.load(out_means)
            pe_precision_matrices = np.load(out_scale_matrices)
            pe_mixing_weights = np.load(out_weights)

            miu_samples = []
            sigma_samples = []

            # sample from the posterior
            for k in range(self.no_clusters):
                miu_sample = []
                sigma_sample = []
                for i in range(num_samples):
                    cov_matrix = sample_covariance(pe_degrees_of_freedom[k], np.linalg.inv(pe_precision_matrices[k]))
                    sigma_sample.append(cov_matrix)
                    miu_sample.append(sample_mean(pe_means[k], cov_matrix, pe_strength_means[k]))

                sigma_samples.append(sigma_sample)
                miu_samples.append(miu_sample)

            miu_samples = np.array(miu_samples)
            sigma_samples = np.array(sigma_samples)

            acc = self.evaluate_performance(x_test, y_test, pe_mixing_weights, pe_strength_means, np.mean(miu_samples, axis=1),
                                            pe_degrees_of_freedom, np.mean(sigma_samples, axis=1))
            print(f"accuracy on test data = {acc}")
            return miu_samples, sigma_samples
        else:
            raise RuntimeError(f"The bayesian mixture model needs to be trained first !\n")

    def evaluate_performance(self, x, y, dirichlet_params, strength_means, cluster_means, degrees_of_freedom, scale_matrices):

        x_np = x.to_numpy()
        y_np = y.to_numpy().flatten()

        cluster_assignments = []

        alpha = np.array(dirichlet_params, dtype=float)
        mixing_weights = alpha / np.sum(alpha)

        for idx, (x_in, y_true) in enumerate(zip(x_np, y_np)):
            result_probs = []
            result_instance = []
            for k in range(self.no_clusters):
                nominator = (strength_means[k] + 1) * scale_matrices[k]
                denominator = degrees_of_freedom[k] - len(x_in) + 1
                denominator *= strength_means[k]
                scale_matrix = nominator / denominator

                result_instance.append(student_t_pdf(x_in, degrees_of_freedom[k], len(x_in), cluster_means[k],
                                                     scale_matrix) * mixing_weights[k])
            result_probs.append(result_instance)
            # print(f"result_probs = {result_probs}")
            result_probs /= np.sum(result_probs)
            cluster_idx = np.argmax(result_probs)
            cluster_assignments.append(cluster_idx)

        # assigning the true labels to each gaussian component
        cluster_labels, mapping = map_clusters(y_np, cluster_assignments, self.no_clusters)
        acc = accuracy_score(y_np, cluster_labels)

        return acc


    def variational_e_step(self, x_train, dim_data):
        """
        update the responsibilities (the probability that point x_i belongs to each cluster) ~ student T distribution
        """

        log_responsibilities = np.zeros((self.no_clusters, x_train.shape[0]))
        for k in range(self.no_clusters):

            log_pi_expectation = psi(self.dirichlet_params[k]) - psi(np.sum(self.dirichlet_params))
            for idx, x_row in enumerate(x_train.itertuples()):

                nominator = (self.posteriors[k].strength_mean + 1) * self.posteriors[k].scale_matrix
                denominator = self.posteriors[k].degrees_of_freedom - dim_data + 1
                denominator *= self.posteriors[k].strength_mean

                scale_matrix = nominator / denominator

                log_pdf = student_t_pdf(np.array(x_row[1:]), self.posteriors[k].degrees_of_freedom, dim_data,
                                                self.posteriors[k].cluster_mean, scale_matrix)
                log_responsibilities[k, idx] = log_pi_expectation + np.log(log_pdf)


        log_responsibilities -= logsumexp(log_responsibilities, axis=0, keepdims=True)
        self.responsibilities = np.exp(log_responsibilities)
        # print(f"self.responsibilities[:, 5] = {self.responsibilities[:, 5]}")

        # sanity check, should sum up to 1
        # print(f"self.responsibilities.sum(axis=) = {self.responsibilities.sum(axis=0)}")


    def build_sample_covariance(self, x_train, soft_counts, weighted_means):

        covariance = []
        for k in range(self.no_clusters):
            cov_dim = 0
            for idx, row in enumerate(x_train.itertuples()):
                diff = np.array(row[1:]) - weighted_means[k]
                diff = diff.reshape(-1, 1)
                cov_dim += (diff @ diff.transpose()) * self.responsibilities[k, idx]

            # print(f"cov_dim = {cov_dim}")
            # print(f"soft_counts[k] = {soft_counts[k]}")  # should sum up to 1
            covariance.append(cov_dim / soft_counts[k])

        return np.array(covariance)


    def build_coefficient(self, soft_count, weighted_mean):

        first_term = (self.priors.strength_mean * soft_count) / (self.priors.strength_mean + soft_count)
        diff = (weighted_mean - self.priors.data_mean).reshape(-1, 1)

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
            self.dirichlet_params[k] = self.priors.dirichlet_prior[k] + soft_counts[k]

        # new weighted mean
        weighted_means = np.zeros((self.no_clusters, x_train.shape[1]))
        for k in range(self.no_clusters):
            for idx, row in enumerate(x_train.itertuples()):
                weighted_means[k, :] += self.responsibilities[k, idx] * np.array(row[1:])
            weighted_means[k, :] /= soft_counts[k]

        sample_cov = self.build_sample_covariance(x_train, soft_counts, weighted_means)

        # NIW posterior parameter updates
        for k in range(self.no_clusters):

            self.posteriors[k].strength_mean = self.priors.strength_mean + soft_counts[k]
            self.posteriors[k].degrees_of_freedom = self.priors.degrees_of_freedom + soft_counts[k]

            mean_contribution = self.priors.strength_mean * self.priors.data_mean
            empirical_contribution = soft_counts[k] * weighted_means[k, :]
            self.posteriors[k].cluster_mean = (mean_contribution + empirical_contribution) / self.posteriors[k].strength_mean

            observed_data = soft_counts[k] * sample_cov[k]
            uncertainty_coefficient = self.build_coefficient(soft_counts[k], weighted_means[k, :])

            self.posteriors[k].scale_matrix = self.priors.scale_matrix + observed_data + uncertainty_coefficient
