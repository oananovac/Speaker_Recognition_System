import numpy as np
import math

def logLikelihood(features, gmm_ubm_means, gmm_ubm_weights, gmm_ubm_covs):
    #  xMinusMu = repmat(x,1,1,numel(gmm.ComponentProportion)) - permute(gmm.mu,[1,3,2]);
    #   repmat face mai multe copii ale arrayului de features la final este un array 3D
    features_ = np.repeat(np.transpose(features)[:,:,np.newaxis],len(gmm_ubm_weights), axis = 2)
    ubm_means_ = np.transpose(np.repeat(np.transpose(gmm_ubm_means)[:,:,np.newaxis],np.shape(features_)[1],axis = 2),(0,2,1))
    feaMinusMu = np.subtract(features_, ubm_means_)

    permuteSigma = np.transpose(np.repeat(np.transpose(gmm_ubm_covs)[:,:,np.newaxis],1,axis=2),(0,2,1))
    l1 = np.sum(np.log(permuteSigma), axis=0)
    permuteSigma2 = np.repeat(permuteSigma, np.shape(features)[0], axis = 1)

    l2 = np.sum(np.multiply(feaMinusMu, np.divide(feaMinusMu, permuteSigma2)), axis=0)
    l3 = len(gmm_ubm_means[0]) * np.log(math.pi * 2)
    l1_final = np.repeat(l1,len(l2),axis = 0)
    l3_final = np.full(np.shape(l1_final),l3)

    final_sum = np.add(np.add(l1_final,l2),l3_final)

    const_var = np.full(np.shape(l1_final),(-0.5))
    Lunweighted = np.multiply(const_var, final_sum)
    temp = np.transpose(Lunweighted)

    ubm_weights_ = np.repeat(gmm_ubm_weights[:,np.newaxis],np.shape(temp)[1],axis = 1)
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

    return n,f,s