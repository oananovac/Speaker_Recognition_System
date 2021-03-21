import numpy as np
import math


def logLikelihood(features_per_frame, ubm):
    #  xMinusMu = repmat(x,1,1,numel(gmm.ComponentProportion)) - permute(gmm.mu,[1,3,2]);
    #   repmat face mai multe copii ale arrayului de features la final este un array 3D
    features_ = np.repeat(np.transpose(features_per_frame)[:, :, np.newaxis], len(ubm.weights), axis=2)
    ubm_means_ = np.transpose(np.repeat(np.transpose(ubm.means)[:, :, np.newaxis], np.shape(features_)[1], axis=2), (0, 2, 1))
    feaMinusMu = np.subtract(features_, ubm_means_)

    permuteSigma = np.transpose(np.repeat(np.transpose(ubm.covs)[:, :, np.newaxis], 1, axis=2), (0, 2, 1))
    l1 = np.sum(np.log(permuteSigma), axis=0)
    permuteSigma2 = np.repeat(permuteSigma, np.shape(features_per_frame)[0], axis=1)

    l2 = np.sum(np.multiply(feaMinusMu, np.divide(feaMinusMu, permuteSigma2)), axis=0)
    l3 = len(ubm.means[0]) * np.log(math.pi * 2)
    l1_final = np.repeat(l1, len(l2), axis=0)
    l3_final = np.full(np.shape(l1_final), l3)

    final_sum = np.add(np.add(l1_final, l2), l3_final)

    const_var = np.full(np.shape(l1_final), (-0.5))
    Lunweighted = np.multiply(const_var, final_sum)
    temp = np.transpose(Lunweighted)

    ubm_weights_ = np.repeat(ubm.weights[:, np.newaxis], np.shape(temp)[1], axis=1)
    L = np.add(temp, np.log(ubm_weights_))

    return L


def compute_statistics(logLikelihood, features):
    amax = np.amax(logLikelihood, axis=0)

    sum_to_amax = np.log(np.sum(np.exp(np.subtract(logLikelihood, np.full(np.shape(logLikelihood),amax))),axis=0))

    logLikelihoodSum = np.add(np.full(np.shape(sum_to_amax),amax), sum_to_amax)
    for_gamma =  np.transpose(np.repeat(logLikelihoodSum[:,np.newaxis], np.shape(logLikelihood)[0], axis = 1))
    gamma = np.transpose(np.exp(np.subtract(logLikelihood, for_gamma)))

    fea_t = np.transpose(features)
    # Compute Baum-Welch statistics
    n = np.sum(gamma, axis=0)
    f = np.matmul(fea_t, gamma)
    s = np.matmul(np.multiply(fea_t, fea_t), gamma)

    return n, f, s


def Baum_Welch_Statistic(features_list, ubm):
    numFiles = 0
    Nc = []
    Fc = []
    for features in features_list:
        logLikelihood_ = logLikelihood(features, ubm)
        n, f, s = compute_statistics(logLikelihood_, features)
        Nc.append(np.copy(n))  # comp gaussiene
        Fc.append(np.copy(f))  # nr features, comp gaussiene
        numFiles += 1

    N = []
    F = []
    for s in range(numFiles):
        N.append(np.reshape(np.repeat(Nc[s][:, np.newaxis], np.shape(ubm.means)[1], axis=1), (1, -1), order='F'))
        F.append(np.subtract(Fc[s], np.multiply(np.transpose(np.repeat(Nc[s][:, np.newaxis], np.shape(ubm.means)[1],
                                                                     axis=1)), np.transpose(ubm.means))))
        F[s] = np.reshape(F[s], (-1, 1), order='F')

    return N, F


def save_statistics(n, f):
    file = open("N_stat.txt", "wb")
    np.save(file, n)
    file.close
    file = open("F_stat.txt", "wb")
    np.save(file, f)
    file.close


def load_statistics():
    file = open("N_stat.txt", "rb")
    n = np.load(file)
    file.close

    file = open("F_stat.txt", "rb")
    f = np.load(file)
    file.close
    return n, f