import numpy as np
import math
import voice_activity_detector
import features_extraction


def log_likelihood_computation(features_per_signal, ubm):
    features_ = np.repeat(np.transpose(features_per_signal)[np.newaxis, :, :],
                          len(ubm.weights), axis=0)
    ubm_means_ = np.repeat(ubm.means[:, :, np.newaxis], np.shape(features_)[2],
                           axis=2)
    features_minus_means = np.transpose(np.subtract(features_, ubm_means_),
                                        (1, 2, 0))

    permute_sigma = np.repeat(np.transpose(ubm.covariances)[:, np.newaxis, :],
                              1, axis=2)

    term_1 = np.sum(np.log(permute_sigma), axis=0)

    permute_sigma = np.repeat(permute_sigma, np.shape(features_per_signal)[
        0], axis=1)

    term_2 = np.sum(np.multiply(features_minus_means,
                                np.divide(features_minus_means,
                                          permute_sigma)), axis=0)
    term_3 = len(ubm.means[0]) * np.log(math.pi * 2)

    term_1 = np.repeat(term_1, len(term_2), axis=0)
    term_3 = np.full(np.shape(term_1), term_3)

    terms_sum = np.add(np.add(term_1, term_2), term_3)

    const_variable = np.full(np.shape(term_1), (-0.5))
    lunweighted = np.multiply(const_variable, terms_sum)
    temp = np.transpose(lunweighted)

    ubm_weights_ = np.repeat(ubm.weights[:, np.newaxis], np.shape(temp)[1],
                             axis=1)
    log_likelihood = np.add(temp, np.log(ubm_weights_))

    return log_likelihood


def statistics_computation(log_likelihood, features):
    # maximum log-likelihood array
    amax = np.amax(log_likelihood, axis=0)

    sum_to_amax = np.log(np.sum(np.exp(np.subtract(log_likelihood, np.full(
        np.shape(log_likelihood), amax))), axis=0))

    logLikelihoodSum = np.add(np.full(np.shape(sum_to_amax), amax),
                              sum_to_amax)
    for_gamma = np.transpose(np.repeat(logLikelihoodSum[:, np.newaxis],
                                       np.shape(log_likelihood)[0], axis=1))
    gamma = np.transpose(np.exp(np.subtract(log_likelihood, for_gamma)))

    # Compute Baum-Welch statistics
    n = np.sum(gamma, axis=0)
    f = np.matmul(np.transpose(features), gamma)
    s = np.matmul(np.multiply(np.transpose(features), np.transpose(features)),
                  gamma)

    return n, f, s


def Baum_Welch_Statistic(paths_dictionary, num_mfcc, ubm, frame_duration,
                         step_duration):
    files_nr = 0
    n_c = []
    f_c = []
    keys_list = paths_dictionary.keys()
    for i in keys_list:
        audioNr = len(paths_dictionary[i])
        for j in range(audioNr):
            vad_obj = voice_activity_detector.Vad(paths_dictionary[i][j], 2)
            signal, rate = vad_obj.get_speech_signal()
            del vad_obj

            mfcc = features_extraction.FeaturesExtraction(num_mfcc, True,
                                                          frame_duration,
                                                          step_duration)
            features = mfcc.extract_mfcc_from_signal(signal, rate)
            log_likelihood_ = log_likelihood_computation(features, ubm)
            n, f, s = statistics_computation(log_likelihood_, features)
            n_c.append(np.copy(n))  # comp gaussiene
            f_c.append(np.copy(f))  # nr features, comp gaussiene
            files_nr += 1

    n = []
    f = []
    for s in range(files_nr):
        n.append(np.reshape(
            np.repeat(n_c[s][:, np.newaxis], np.shape(ubm.means)[1], axis=1),
            (1, -1), order='C'))
        f.append(np.subtract(f_c[s], np.multiply(np.transpose(
            np.repeat(n_c[s][:, np.newaxis], np.shape(ubm.means)[1],
                      axis=1)), np.transpose(ubm.means))))
        f[s] = np.reshape(f[s], (-1, 1), order='F')

    return n, f


def save_statistics(n, f, path, components_number):
    file = open(path + "/models/N_stat_" + str(components_number) + ".txt",
                "wb")
    np.save(file, n)
    file.close

    file = open(path + "/models/F_stat_" + str(components_number) + ".txt",
                "wb")
    np.save(file, f)
    file.close


def load_statistics(path, components_number):
    file = open(path + "/models/N_stat_" + str(components_number) + ".txt",
                "rb")
    n = np.load(file)
    file.close

    file = open(path + "/models/F_stat_" + str(components_number) + ".txt",
                "rb")
    f = np.load(file)
    file.close

    return n, f
