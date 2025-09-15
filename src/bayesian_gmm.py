import numpy as np
from scipy.stats import dirichlet
from scipy.special import gammaln, psi  # psi is the logarithmic derivative of the gamma function
from utils import init_cluster_means, init_priors, init_responsibilities


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

    def __init__(self, no_clusters: int):

        self.no_clusters = no_clusters
        self.weights = [(1 / no_clusters) for _ in range(no_clusters)]
        self.prior_hyperparameters = []
        self.responsibilities = []
        self.niw_posteriors = [NormalInverseWishartDistribution for _ in range(no_clusters)]


    def train(self, num_iterations, x_train, y_train=None):

        # parameter initialization
        dim_data = x_train.shape[1]
        labels, cluster_means = init_cluster_means(x_train, self.no_clusters)

        self.responsibilities = init_responsibilities(self.no_clusters, labels, x_train)
        # weighted_mean_0, strength_mean (the confidence in the data_mean), degree_of_freedom, precision_matrix
        self.prior_hyperparameters = init_priors(x_train, labels, self.no_clusters, self.responsibilities)

        for i, posterior in enumerate(self.niw_posteriors):
            posterior.cluster_mean = cluster_means[i]
            posterior.strength_mean = self.prior_hyperparameters[2] + 1
            posterior.freedom_degrees = dim_data + 2
            posterior.precision_matrix = self.prior_hyperparameters[3]

        # !!! tre' sa stii ce faci de la bun inceput, transformi sau nu x_train in numpy array sau il lasi dataframe ?

        for i in range(num_iterations):
            print(f"Now at iteration {i} \n")

            self.e_step(x_train, dim_data, cluster_means)
            self.variational_m_step(x_train)
            print(f"self.weights = {self.weights}")
            print(f"self.weighted_means = {self.prior_hyperparameters[0]}")


    def predict(self, x_test, y_test):
        pass


    def compute_log_xi(self, cluster_idx, cluster_responsibility, cluster_mean, data_in, no_features):

        first_term = 0
        second_term = 0
        for dimension in range(1, no_features):
            first_term += psi((cluster_responsibility + 1 - dimension) / 2)

        first_term += no_features * np.log(2) + np.log(np.linalg.det(self.niw_posteriors[cluster_idx].precision_matrix))
        first_term /= 2
        first_term -= (no_features / 2) * np.log(np.pi * 2)

        diff = data_in - cluster_mean
        diff = diff.reshape(-1, 1)

        second_term += (diff.transpose() @ self.niw_posteriors[cluster_idx].precision_matrix) @ diff
        second_term *= self.niw_posteriors[cluster_idx].freedom_degrees
        second_term += no_features / self.niw_posteriors[cluster_idx].strength_mean

        return first_term - second_term / 2


    def e_step(self, x_train, no_features, cluster_means):
        """
        update the responsibilities (i.e. the probability that point x_i belongs to each cluster)
        """
        total_weight = np.sum(self.weights)
        print(f"total_weight = {total_weight} \n")
        # print(f"inainte de for self.responsibilities = {self.responsibilities}")
        for k in range(self.no_clusters):
            for data_idx, row in enumerate(x_train.itertuples()):
                # TODO: debugging AICI, ceva se intampla cu cluster_weight
                cluster_weight = self.weights[k]
                # print(f"cluster_weight = {cluster_weight}")
                log_zi_prob = psi(cluster_weight) - psi(total_weight)
                log_xi_prob = self.compute_log_xi(k, cluster_weight, cluster_means[k], row[1:], no_features)
                # print(f"log_zi_prob = {log_zi_prob}")
                # print(f"log_xi_prob = {log_xi_prob}")
                self.responsibilities[k, data_idx] = log_zi_prob + log_xi_prob

        # print(f"dupa for self.responsibilities = {self.responsibilities}")
        # subtracting the max to prevent overflows
        for k in range(self.no_clusters):
            max_prob = np.argmax(self.responsibilities[k, :])
            self.responsibilities[k, :] -= max_prob

        self.responsibilities = np.exp(self.responsibilities)

        # normalization
        for k in range(self.no_clusters):
            normalization_term = np.sum(np.exp(self.responsibilities[k, :]))
            for data_idx in range(x_train.shape[0]):
                self.responsibilities[k, data_idx] = np.exp(self.responsibilities[k, data_idx]) / normalization_term


    def build_sample_covariance(self, x_train, soft_counts):

        sample_covariance = []
        for k in range(self.no_clusters):
            cov_dim = 0
            for idx, row in enumerate(x_train.itertuples()):
                diff = np.array(row[1:]) - self.means[k]  ## aici formula e gresita; ar trebui sa fie weighted_mean - prev_mean
                diff = diff.reshape(-1, 1)
                cov_dim += (diff @ diff.transpose()) * self.responsibilities[k, idx]

            print(f"cov_dim = {cov_dim}")
            print(f"soft_counts[k] = {soft_counts[k]}")  # de ce toate soft_count-urile sunt 0.99 ??
            sample_covariance.append(cov_dim / soft_counts[k])

        return np.array(sample_covariance)


    def build_coefficient(self, k, soft_count, old_mean):

        first_term = ((self.niw_posteriors[k].strength_mean - soft_count) * soft_count) / self.niw_posteriors[k].strength_mean
        diff = self.niw_posteriors[k].cluster_mean - old_mean
        diff = diff.reshape(-1, 1)

        return (diff @ diff.transpose()) * first_term


    def variational_m_step(self, x_train):
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
            self.weights[k] += soft_counts[k]

        print(f"soft_counts = {soft_counts} \n ")

        # !! ALL THE CLUSTERS HAVE THE SAME MEAN AS THE CENTER ????

        # new weighted mean
        weighted_means = np.zeros((self.no_clusters, x_train.shape[1]))
        for k in range(self.no_clusters):
            for idx, row in enumerate(x_train.itertuples()):
                # dot product, or just multiplication of the values with a scalar ??
                weighted_means[k, :] += self.responsibilities[k, idx] * np.array(row[1:])
            weighted_means /= soft_counts[k]

        sample_covariance = self.build_sample_covariance(x_train, soft_counts)

        # NIW parameter updates
        for k in range(self.no_clusters):
            self.niw_posteriors[k].strength_mean += soft_counts[k]
            self.niw_posteriors[k].freedom_degrees += soft_counts[k]

            old_mean = self.niw_posteriors[k].cluster_mean
            mean_contribution = (self.niw_posteriors[k].strength_mean - soft_counts[k]) * old_mean
            data_contribution = soft_counts[k] * weighted_means[k, :]

            self.niw_posteriors[k].cluster_mean = (mean_contribution + data_contribution) / self.niw_posteriors[k].strength_mean

            observed_data = soft_counts[k] * sample_covariance[k]
            uncertainty_coefficient = self.build_coefficient(k, soft_counts[k], old_mean)
            self.niw_posteriors[k].precision_matrix = (self.niw_posteriors[k].precision_matrix + observed_data +
                                                       uncertainty_coefficient)



