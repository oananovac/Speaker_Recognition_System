import numpy as np
from scipy import signal
import mfcc
import baum_welch
from scipy.stats import norm
import vad
from scipy.linalg import eigh
from scipy.io import savemat

def Total_Variability_Space(N, F, ubm, numTdim, numIterations, numFeatures, numComp):
    sigma = np.zeros(shape=(len(ubm.covs)*len(ubm.covs[0])))
    k = 0
    for i in range(len(ubm.covs[0])):
        for j in range(len(ubm.covs)):
            sigma[k] = ubm.covs[j][i]
            k += 1
    count_covs = np.shape(sigma)[0]

    np.random.seed(1)
    T = norm.ppf(np.random.rand(count_covs, numTdim))
    normT = np.linalg.norm(T)


    newT = np.divide(T, np.full(np.shape(T), normT))
    t = {"norm": newT, "label": "t"}
    savemat("normT.mat", t)
    I = np.eye(numTdim, dtype=float)

    for i in range(numIterations):

        Ey = []
        Eyy = []
        Linv = []
        final_matrix = []

        print("Iteratia :" + str(i))
        rep_sigma  = np.repeat(np.transpose(sigma)[:,np.newaxis], np.shape(newT)[1], axis=1)
        TtimesInverseSSdiag = np.transpose(np.divide(newT, rep_sigma))

        # 1. Calculate the posterior distribution of the hidden variable

        for s in range(len(N)):
            mul = np.matmul(np.multiply(TtimesInverseSSdiag, np.repeat(N[s], np.shape(newT)[1], axis=0)), newT)
            L = np.add(I, mul)
            Linv.append(np.linalg.pinv(L))
            Ey.append(np.matmul((np.matmul(Linv[s], TtimesInverseSSdiag)), F[s]))
            Eyy.append(np.add(Linv[s], np.matmul(Ey[s], np.transpose(Ey[s]))))

        # 2. Accumlate statistics across the speakers
        Eymat = []
        for it in range(len(Ey)):
            if it == 0:
                Eymat = Ey[it]
            else:
                Eymat = np.concatenate((Eymat, Ey[it]), axis=1)
        FFmat = np.zeros(shape=(len(F[0]), len(F)))
        for k in range(len(F)):
            for l in range(len(F[0])):
                FFmat[l][k] = F[k][l]
        Kt = np.matmul(FFmat, np.transpose(Eymat))
        K = np.zeros(shape=(numTdim, numFeatures, numComp))
        i = 0

        for k in range(numTdim):
            for k1 in range(numFeatures):
                K[k][k1] = Kt[i]
                i += 1

        K = np.transpose(K, (0, 2, 1))
        newnT = []
        for c in range(numComp):
            AcLocal = np.zeros(shape=(numTdim, numTdim))
            for s in range(np.shape(N)[0]):
                nc = np.full(np.shape(Eyy[s]), N[s][0][c])
                AcLocal = np.add(AcLocal, np.multiply(nc, Eyy[s]))

            # 3. Update the Total Variability Space
            newnT.append(np.transpose(np.matmul(np.linalg.pinv(AcLocal), K[c])))

        for it in range(len(newnT)):
            if it == 0:
                final_matrix = newnT[it]
            else:
                final_matrix = np.concatenate((final_matrix, newnT[it]), axis=0)

        newT = final_matrix

    f = open("T_matrix.txt", "wb")
    np.save(f, final_matrix)
    f.close


def load_T_matrix():
    f = open("T_matrix.txt", "rb")
    matrix = np.load(f)
    f.close
    return matrix

def extract_ivector_from_signal(ubm, utternace_path , T_matrix, numTdim, numFeatures):
    sigma = np.zeros(shape=(len(ubm.covs) * len(ubm.covs[0])))
    k = 0
    for i in range(len(ubm.covs[0])):
        for j in range(len(ubm.covs)):
            sigma[k] = ubm.covs[j][i]
            k += 1

    rep_sigma = np.repeat(np.transpose(sigma)[:, np.newaxis], numTdim, axis=1)
    TS = np.divide(T_matrix, rep_sigma)
    TSi = np.transpose(TS)
    I = np.eye(128, dtype=float)

    vad_obj = vad.Vad(utternace_path, 2)
    signal, rate = vad_obj.get_speech_signal()
    features = mfcc.extract_mfcc_from_signal(signal, rate, numFeatures)
    logLikelihood = baum_welch.logLikelihood(features, ubm)
    n, f, s = baum_welch.compute_statistics(logLikelihood, features)
    ff = np.subtract(f, np.multiply(np.transpose(np.repeat(n[:, np.newaxis], np.shape(ubm.means)[1], axis=1)),
                                    np.transpose(ubm.means)))
    i1 = np.matmul(np.transpose(
        np.multiply(TS,
                    np.repeat(np.transpose(np.repeat(n, np.shape(features)[1]))[:, np.newaxis], 128, axis=1))),
        T_matrix)
    i2 = np.matmul(np.linalg.pinv(np.add(I, i1)), TSi)
    i3 = []
    for i in range(np.shape(ff)[1]):
        if i == 0:
            i3 = np.transpose(ff)[i]
        else:
            i3 = np.concatenate((i3, np.transpose(ff)[i]), axis=0)
    ivector = np.matmul(i2, i3)
    return ivector

def extract_i_vectors(ubm, train_paths, T_matrix, numTdim, numFeatures):  # numTdim = num gaussiene
    sigma = np.zeros(shape=(len(ubm.covs) * len(ubm.covs[0])))
    k = 0
    for i in range(len(ubm.covs[0])):
        for j in range(len(ubm.covs)):
            sigma[k] = ubm.covs[j][i]
            k += 1

    rep_sigma = np.repeat(np.transpose(sigma)[:, np.newaxis], numTdim, axis=1)

    speakers_list = train_paths.keys()
    TS = np.divide(T_matrix, rep_sigma)
    TSi = np.transpose(TS)
    I = np.eye(128, dtype=float)

    ivectors = {}
    ivector_per_speaker = []
    ivector_sp_2 = []
    num = 0
    ii1 = 0
    ii2 = 0
    for speaker in speakers_list:
        ivector_per_file = []
        """
        if num == 0:
            ii1 = len(train_paths[speaker])
        if num != 0:
            if ii1 != len(train_paths[speaker]):
                ii2 = len(train_paths[speaker])
        """
        for file in range(len(train_paths[speaker])):
            vad_obj = vad.Vad(train_paths[speaker][file], 2)
            # signal, rate = sf.read(paths_dictionary[i][j])
            signal, rate = vad_obj.get_speech_signal()
            features = mfcc.extract_mfcc_from_signal(signal, rate, numFeatures)
            logLikelihood = baum_welch.logLikelihood(features, ubm)
            n, f, s = baum_welch.compute_statistics(logLikelihood, features)
            ff = np.subtract(f, np.multiply(np.transpose(np.repeat(n[:, np.newaxis], np.shape(ubm.means)[1], axis=1)),
                                            np.transpose(ubm.means)))
            i1 = np.matmul(np.transpose(
                np.multiply(TS,
                            np.repeat(np.transpose(np.repeat(n, np.shape(features)[1]))[:, np.newaxis], 128, axis=1))),
                T_matrix)
            i2 = np.matmul(np.linalg.pinv(np.add(I, i1)), TSi)
            i3 = []
            for i in range(np.shape(ff)[1]):
                if i == 0:
                    i3 = np.transpose(ff)[i]
                else:
                    i3 = np.concatenate((i3, np.transpose(ff)[i]), axis=0)
            ivector_per_file.append(np.matmul(i2, i3))

        # i-vector per file appended to i-vector per speaker list
        ii = np.transpose(np.dstack(ivector_per_file)[0])
        """
        if ii1 == len(train_paths[speaker]):
            ivector_per_speaker.append(ii)
        if ii2 == len(train_paths[speaker]):
            ivector_sp_2.append(ii)
        """
        ivectors[speaker] = ii

        f = open(speaker + "_ivector.txt", "wb")
        np.save(f, ivectors[speaker])
        f.close

        num += 1
    """
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    ivector_array = np.dstack(ivector_per_speaker)
    ivector_array2 = np.dstack(ivector_sp_2)
    array = np.rollaxis(ivector_array, -1)
    f = open("ivectors.txt", "wb")
    np.save(f, array)
    f.close
    array2 = np.rollaxis(ivector_array2, -1)
    f = open("ivectors2.txt", "wb")
    np.save(f, array2)
    f.close
    """


def concatenate_ivectors(ivectors):
    ivector_list = ivectors

    cat_list = []
    it = 0
    for lists in ivector_list:
        if it == 0:
            cat_list = lists
        else:
            cat_list = np.concatenate((cat_list, lists), axis=0)
        it += 1

    return cat_list


def LDA_projection_matrix(ivectors):  # LDA projection matrix
    ivector_list = ivectors

    cat_list = concatenate_ivectors(ivector_list)

    projection_matrix = np.identity(len(ivector_list[0][0]))

    num_eigen_vectors = len(ivector_list)
    Sw = np.zeros(np.shape(projection_matrix))
    Sb = np.zeros(np.shape(projection_matrix))

    wbar = np.mean(cat_list, axis=0)

    for lists in ivector_list:
        ws = lists
        wsbar = np.mean(ws, axis=0)
        ws_sub = np.reshape(np.subtract(wsbar, wbar), (np.shape(wbar)[0], 1))
        ws_mul = np.matmul(ws_sub, np.transpose(ws_sub))

        Sb = np.add(Sb, ws_mul)

        ws_cov = np.cov(np.transpose(ws), bias=True)
        Sw = np.add(Sw, ws_cov)

    eigvals, eigvecs = eigh(Sb, Sw, eigvals_only=False)
    zipped_eig = zip(eigvals, eigvecs)
    sorted_zipped_eig = sorted(zipped_eig, reverse=True)
    sortedd = [element for _, element in sorted_zipped_eig]
    a_matrix = []

    for i in range(num_eigen_vectors):
        a_matrix.append(sortedd[i])
    a_matrix = np.dstack(a_matrix)
    a_matrix = np.rollaxis(a_matrix[0], -1)

    a_matrix = np.divide(a_matrix, np.repeat(np.linalg.norm(a_matrix, axis=1)[:, np.newaxis], len(a_matrix[0]), axis=1))

    ivectors_fin = np.matmul(a_matrix, np.transpose(cat_list))
    projection_matrix = np.matmul(a_matrix, projection_matrix)

    return projection_matrix, ivectors_fin


def isPD(B):
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearestPD(A):
    B = (A + np.transpose(A)) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(np.transpose(V), np.dot(np.diag(s), V))

    A2 = np.add(B, H) / 2

    A3 = np.add(A2, np.transpose(A2)) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def WCCN_projection_matrix(lda_projection_matrix, ivectors, utterances):
    num_eigen_vectors = len(ivectors)
    alpha = 0.9

    ivv = []
    index = 0
    utt_keys = utterances.keys()
    start = 0
    final = 0
    for i in utt_keys:
        final += utterances[i]
        iv = np.zeros((num_eigen_vectors, utterances[i]))
        for j in range(num_eigen_vectors):
            iv[j] = ivectors[j][start:final]
        ivv.append(iv)
        index += 1
        start += utterances[i]

    w_ = np.zeros((len(lda_projection_matrix), len(lda_projection_matrix)))

    for i in range(len(ivv)):
        w_ = np.add(w_, np.cov(ivv[i], bias=True))

    w_ = np.divide(w_, np.full((np.shape(w_)[0], np.shape(w_)[1]), num_eigen_vectors))
    w_ = np.add(np.multiply(np.full((np.shape(w_)[0], np.shape(w_)[1]), 1-alpha), w_),
                np.multiply(np.full((np.shape(w_)[0], np.shape(w_)[1]), alpha), np.identity(np.shape(w_)[0])))

    b_matrix = np.linalg.cholesky(np.linalg.pinv(w_)) #nearestPD(np.linalg.pinv(w_)))
    wccn_projection_matrix = np.matmul(b_matrix, lda_projection_matrix)

    return wccn_projection_matrix

def load_projection_matrix():
    f = open("projection_matrix.txt", "rb")
    p_matrix = np.load(f)
    f.close
    return p_matrix


