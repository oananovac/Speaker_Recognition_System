import numpy as np
import mfcc
import baum_welch
from os import walk
import scipy.io.wavfile as spiowav
from scipy import signal
import os
from scipy.stats import norm

###ubm_file_path = r"D:\licenta\voice biometry\models\GMM.txt.gz"

def load_ubm(ubm_file_path):
    gmm_ubm = np.loadtxt(ubm_file_path, dtype=np.float32)

    means_and_covs_dim = (gmm_ubm.shape[1] - 1) / 2
    weights_ubm = gmm_ubm[:, 0]
    means_ubm = gmm_ubm[:, 1:(int(means_and_covs_dim)+1)] #fiecare linie si coloana incepand cu col 1 pana la coloana means_and_covs_dim +1
    covs_ubm = gmm_ubm[:, (int(means_and_covs_dim)+1):] #fiecare linie si restul coloanelor

    ubm = dict()
    ubm['weights_ubm'] = weights_ubm
    ubm['means_ubm'] = means_ubm
    ubm['covs_ubm'] = covs_ubm
    return ubm


def get_train_test_paths(directory):
    train_data = {}
    test_data = {}

    path_train = []
    path_test = []

    test_list = ["F09", "F10", "M09", "M10"]

    for entry in os.scandir(directory):
        for newEntry in os.scandir(entry):
            if(newEntry.name == 'MIC'):
                for newNewEntry in os.scandir(newEntry):
                    for files in os.scandir(newNewEntry):
                        if(files.path.endswith('.wav')):
                            if(newNewEntry.name in test_list):
                                path_test.append(files.path)
                            else:
                                path_train.append(files.path)
                    if (newNewEntry.name in test_list):
                        test_data[newNewEntry.name] = []
                        test_data[newNewEntry.name].extend(path_test)
                        path_test.clear()
                    else:
                        train_data[newNewEntry.name] = []
                        train_data[newNewEntry.name].extend(path_train)
                        path_train.clear()


    return train_data, test_data

def extract_features(paths_dictionary):
    final_fatures = None
    keys_list = paths_dictionary.keys()
    for i in keys_list:
        audioNr = len(paths_dictionary[i])
        for j in range(audioNr):
            path = paths_dictionary[i][j]
            rate, sig = spiowav.read(paths_dictionary[i][j])
            if ( rate != 16000):
                resampled = signal.resample(sig, 16000)
################################## AICI TREBUIE VAD ####################################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            features = mfcc.coeff_from_signal(resampled, 16000)
            features_array = features#.flatten()
            if final_fatures is None:
                final_fatures = features_array
            else:
                final_fatures = np.concatenate((final_fatures, features_array))

    return final_fatures

def Baum_Welch_Statistic(paths_dictionary, ubm_means, ubm_covs, ubm_weights):
    keys_list = paths_dictionary.keys()
    numFiles = 0
    Nc = []
    Fc = []
    for i in keys_list:
        audioNr = len(paths_dictionary[i])
        for j in range(audioNr):
            rate, sig = spiowav.read(paths_dictionary[i][j])
            if (rate != 16000):
                resampled = signal.resample(sig, 16000)
            ################################## AICI TREBUIE VAD #######################################

            features = mfcc.coeff_from_signal(resampled, 16000)

            #for i in features:
            logLikelihood = baum_welch.logLikelihood(features,ubm_means, ubm_weights, ubm_covs)
            n, f, s = baum_welch.compute_statistics(logLikelihood, features)
            Nc.append(np.copy(n)) # comp gaussiene
            Fc.append(np.copy(f)) # nr features, comp gaussiene
            numFiles += 1

    N = []
    F = []
    for s in range(numFiles):
        N.append(np.reshape(np.repeat(Nc[s][:, np.newaxis], np.shape(ubm_means)[1],axis=1),(1,-1), order='F'))
        F.append(np.subtract(Fc[s], np.multiply(np.transpose(np.repeat(Nc[s][:, np.newaxis], np.shape(ubm_means)[1],
                                                                     axis=1)), np.transpose(ubm_means))))
        F[s] = np.reshape(F[s],(-1,1),order='F')

    return N,F

def Total_Variability_Space(N,F, ubm_covs, numTdim, numIterations, numFeatures, numComp):
    sigma = np.reshape(ubm_covs,(-1,1))
    count_covs = np.shape(sigma)[0] * np.shape(sigma)[1]

    np.random.seed(1)
    T = norm.ppf(np.random.rand(count_covs,numTdim))
    normT = np.linalg.norm(T)
    newT = np.divide(T, np.full(np.shape(T),normT))
    I = np.eye(numTdim, dtype=float)



    for i in range(numIterations):

        Ey = []
        Eyy = []
        Linv = []
        finalT = []

        print("iteratia :" + str(i))
        TtimesInverseSSdiag = np.transpose(np.divide(newT, np.repeat(sigma,np.shape(newT)[1],axis=1)))

        # 1. Calculate the posterior distribution of the hidden variable

        for s in range(np.shape(N)[0]):
            mul = np.matmul(np.multiply(TtimesInverseSSdiag,np.repeat(N[s],np.shape(newT)[1],axis=0)),newT)
            L = np.add(I,mul)
            Linv.append(np.linalg.pinv(L))
            Ey.append(np.matmul((np.matmul(Linv[s],TtimesInverseSSdiag)),F[s]))
            Eyy.append(np.add(Linv[s], np.matmul(Ey[s],np.transpose(Ey[s]))))

        # 2. Accumlate statistics across the speakers

        Eymat = np.array(Ey).reshape(len(Ey),-1)
        FFmat = np.array(F).reshape(np.shape(F)[1],-1)
        Kt = np.matmul(FFmat, Eymat)
        K = np.zeros(shape = (numTdim,numFeatures, numComp))
        i = 0

        for k in range(numTdim):
            for k1 in range(numFeatures):
                K[k][k1] = Kt[i]
                i+=1

        K = np.transpose(K, (0,2,1))
        newnT = []
        for c in range(numComp):
            AcLocal = np.zeros(shape = (numTdim, numTdim))
            for s in range(np.shape(N)[0]):
                nc = np.full(np.shape(Eyy[s]),N[s][0][c])
                AcLocal = np.add(AcLocal, np.multiply(nc,Eyy[s]))

            # 3. Update the Total Variability Space
            newnT.append(np.transpose(np.matmul(np.linalg.pinv(AcLocal),K[c])))

        for it in range(len(newnT)):
            if it == 0:
                finalT = newnT[it]
            else:
                finalT = np.concatenate((finalT, newnT[it]), axis=0)

        newT = finalT

    f = open("T_matrix.txt", "wb")
    np.save(f, finalT)
    f.close


#---------------------------- EXTRAGERE FEATURES ---------------------------------
"""

train_paths, test_paths = get_train_test_paths(r"D:\licenta\SPEECH DATA")
train_features = extract_features(train_paths)
f = open("train_features.txt", "wb")
np.save(f, train_features)
f.close
"""


#------------------------------------- UBM GMM --------------------------------------
"""
f = open("train_features.txt", "rb")
feature_array = np.load(f)
f.close

ubm_means, ubm_covs, ubm_weights = mfcc.gmm_params(feature_array, 128)
f = open("ubm_means.txt", "wb")
np.save(f, ubm_means)
f.close


f = open("ubm_covs.txt", "wb")
np.save(f, ubm_covs)
f.close


f = open("ubm_weights.txt", "wb")
np.save(f, ubm_weights)
f.close
"""
#------------------------ LOAD UBM DATA ----------------------------------------
"""
f = open("ubm_means.txt", "rb")
ubm_means = np.load(f)
f.close

f = open("ubm_covs.txt", "rb")
ubm_covs = np.load(f)
f.close

f = open("ubm_weights.txt", "rb")
ubm_weights = np.load(f)
f.close
"""
#train_paths, test_paths = get_train_test_paths(r"D:\licenta\SPEECH DATA")

#--------------------------------- Baum Welch Statistics ----------------------------------------------
"""
N, F = Baum_Welch_Statistic(train_paths, ubm_means, ubm_covs, ubm_weights)

f = open("N_stat.txt", "wb")
np.save(f,N)
f.close
f = open("F_stat.txt", "wb")
np.save(f,F)
f.close
"""

# -------------------------------- load N and F from Baum Welch Statistics ---------------------------------------
"""

f = open("N_stat.txt", "rb")
N = np.load(f)
f.close

f = open("F_stat.txt", "rb")
F = np.load(f)
f.close
"""

#Total_Variability_Space(N,F,ubm_covs,128, 20, 39, 128)