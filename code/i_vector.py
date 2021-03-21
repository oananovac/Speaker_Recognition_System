import numpy as np
from scipy import signal
import mfcc
import baum_welch
from scipy.stats import norm
import vad


def Total_Variability_Space(N, F, ubm, numTdim, numIterations, numFeatures, numComp):
    sigma = np.reshape(ubm.covs, (-1, 1))
    count_covs = np.shape(sigma)[0] * np.shape(sigma)[1]

    np.random.seed(1)
    T = norm.ppf(np.random.rand(count_covs, numTdim))
    normT = np.linalg.norm(T)
    newT = np.divide(T, np.full(np.shape(T), normT))
    I = np.eye(numTdim, dtype=float)

    for i in range(numIterations):

        Ey = []
        Eyy = []
        Linv = []
        final_matrix = []

        print("Iteratia :" + str(i))
        TtimesInverseSSdiag = np.transpose(np.divide(newT, np.repeat(sigma, np.shape(newT)[1], axis=1)))

        # 1. Calculate the posterior distribution of the hidden variable

        for s in range(np.shape(N)[0]):
            mul = np.matmul(np.multiply(TtimesInverseSSdiag, np.repeat(N[s], np.shape(newT)[1], axis=0)), newT)
            L = np.add(I, mul)
            Linv.append(np.linalg.pinv(L))
            Ey.append(np.matmul((np.matmul(Linv[s], TtimesInverseSSdiag)), F[s]))
            Eyy.append(np.add(Linv[s], np.matmul(Ey[s], np.transpose(Ey[s]))))

        # 2. Accumlate statistics across the speakers

        Eymat = np.array(Ey).reshape(len(Ey), -1)
        FFmat = np.array(F).reshape(np.shape(F)[1], -1)
        Kt = np.matmul(FFmat, Eymat)
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


def extract_i_vectors(ubm, train_paths, T_matrix, numTdim, numFeatures):  # numTdim = num gaussiene
    sigma = np.reshape(ubm.covs, (-1, 1))
    speakers_list = train_paths.keys()
    I = np.eye(numTdim, dtype=float)

    TS = np.divide(T_matrix, np.repeat(sigma, np.shape(T_matrix)[1], axis=1))
    TSi = np.transpose(TS)

    ivector_per_speaker = []
    for speaker in speakers_list:
        ivector_per_file = []
        for file in range(len(train_paths[speaker])):
            vad_obj = vad.Vad(train_paths[speaker][file], 2)
            sig, rate = vad_obj.get_speech_signal()

            if (rate != 16000):
                resampled = signal.resample(sig, 16000)

            features = mfcc.extract_mfcc(resampled, 16000, numFeatures)
            logLikelihood = baum_welch.logLikelihood(features, ubm)
            n, f, s = baum_welch.compute_statistics(logLikelihood, features)

            ff = np.subtract(f, np.multiply(np.transpose(np.repeat(n[:, np.newaxis], np.shape(ubm.means)[1], axis=1)),
                                            np.transpose(ubm.means)))

            i1 = np.matmul(np.transpose(
                np.multiply(TS, np.repeat(np.repeat(n, np.shape(features)[1])[:, np.newaxis], numTdim, axis=1))),
                           T_matrix)
            i2 = np.matmul(np.linalg.pinv(np.add(I, i1)), TSi)
            i3 = []
            for i in range(np.shape(ff)[1]):
                if i == 0:
                    i3 = np.transpose(ff)[i]
                else:
                    i3 = np.concatenate((i3, np.transpose(ff)[i]), axis=0)
            ivector_per_file.append(np.matmul(i2, i3))
            del vad_obj

        # i-vector per file appended to i-vector per speaker list
        ivector_per_speaker.append(ivector_per_file)
