import numpy as np
from sklearn.mixture import GaussianMixture


class Gmm(object):
    def __init__(self, weights, means, covs):
        self.weights = weights
        self.means = means
        self.covs = covs


def load_ubm():
    f = open("ubm_means.txt", "rb")
    ubm_means = np.load(f)
    f.close

    f = open("ubm_covs.txt", "rb")
    ubm_covs = np.load(f)
    f.close

    f = open("ubm_weights.txt", "rb")
    ubm_weights = np.load(f)
    f.close

    ubm = Gmm(ubm_weights, ubm_means, ubm_covs)
    return ubm


def save_ubm(ubm):
    f = open("ubm_means.txt", "wb")
    np.save(f, ubm.means)
    f.close

    f = open("ubm_covs.txt", "wb")
    np.save(f, ubm.covs)
    f.close

    f = open("ubm_weights.txt", "wb")
    np.save(f, ubm.weights)
    f.close


def gmm_params(features_list, n_components):
    """
    features_array = None
    for i in features_list:
        if features_array is None:
            features_array = i
        else:
            features_array = np.concatenate((features_array, i))
    """

    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=500, n_init=3, verbose=1)
    gmm.fit(features_list)  # se face estimarea parametrilor modelului folosing algortimul EM a.i. sa se obtina
    # maximum likelihood

    gmm_obj = Gmm(gmm.weights_, gmm.means_, gmm.covariances_)
    return gmm_obj

