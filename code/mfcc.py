import python_speech_features
import numpy as np
import speechpy
from sklearn.mixture import GaussianMixture
from scipy import signal
import vad


def extract_mfcc(signal, sample_rate, num_coeff):
    mfcc = python_speech_features.mfcc(signal, samplerate=sample_rate, numcep=num_coeff)
    norm_mfcc = speechpy.processing.cmvn(mfcc)
    deltas = python_speech_features.delta(norm_mfcc, 2)
    delta_delta = python_speech_features.delta(deltas, 2) # 2 este nr de frame care se ia in considerare in fata si in spatele frame ului curent

    final_coef = [norm_mfcc]
    final_coef.append(deltas)
    final_coef.append(delta_delta)

    final = np.hstack(final_coef)
    return final


def gmm_params(features_list, n_components):

    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=500, n_init=3, verbose=1)
    gmm.fit(features_list)  # se face estimarea parametrilor modelului folosing algortimul EM a.i. sa se obtina maximum likelihood

    means = gmm.means_
    covs = gmm.covariances_
    weights = gmm.weights_

    return means, covs, weights


def extract_features(paths_dictionary, num_mfcc):
    ff = []
    keys_list = paths_dictionary.keys()
    for i in keys_list:
        audioNr = len(paths_dictionary[i])
        for j in range(audioNr):
            vad_obj = vad.Vad(paths_dictionary[i][j], 2)
            sig, rate = vad_obj.get_speech_signal()
            if (rate != 16000):
                resampled = signal.resample(sig.astype(int), 16000)
            features = extract_mfcc(resampled, 16000, num_mfcc)
            ff.append(features)
            del vad_obj

    return ff
