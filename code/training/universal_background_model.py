import numpy as np
from sklearn.mixture import GaussianMixture


class GaussianMixtureModel(object):
    def __init__(self, weights, means, covariances, components_number):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.components_number = components_number

    def save_ubm(self, path):
        file_1 = open(path + r"/models/ubm_means_" + str(
            self.components_number) + ".txt",
                      "wb")
        np.save(file_1, self.means)
        file_1.close

        file_2 = open(path + r"/models/ubm_covariances_" +
                      str(self.components_number) +
                      ".txt", "wb")
        np.save(file_2, self.covariances)
        file_2.close

        file_3 = open(path + r"/models/ubm_weights_" +
                      str(self.components_number) +
                      ".txt", "wb")
        np.save(file_3, self.weights)
        file_3.close


def create_ubm(features, components_number):
    gmm = GaussianMixture(n_components=components_number, covariance_type='diag',
                          max_iter=500, n_init=3, verbose=1)
    gmm.fit(features)

    return GaussianMixtureModel(gmm.weights_, gmm.means_, gmm.covariances_,
                                components_number)


def load_ubm(path, components_number):
    file_1 = open(path + r"/models/ubm_means_" + str(components_number) +
                  ".txt",
                  "rb")
    ubm_means = np.load(file_1)
    file_1.close

    file_2 = open(path + r"/models/ubm_covariances_" + str(components_number) +
                  ".txt", "rb")
    ubm_covariances = np.load(file_2)
    file_2.close

    file_3 = open(path + r"/models/ubm_weights_" + str(components_number) +
                  ".txt",
                  "rb")
    ubm_weights = np.load(file_3)
    file_3.close

    ubm = GaussianMixtureModel(ubm_weights, ubm_means, ubm_covariances,
                               components_number)
    return ubm
