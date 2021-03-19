import numpy as np
import ubm
import scipy.io.wavfile as spiowav
from scipy import signal
import mfcc
import baum_welch

def extract_i_vectors(ubm_covs, ubm_means,ubm_weights ,train_paths, T_matrix, numTdim): #numTdim = num gaussiene
    sigma = np.reshape(ubm_covs, (-1, 1))
    speakers_list = train_paths.keys()
    I = np.eye(numTdim, dtype=float)

    TS = np.divide(T,np.repeat(sigma,np.shape(T)[1],axis=1) )
    TSi = np.transpose(TS)

    i= 0
    ivector_per_speaker = []
    for speaker in speakers_list:
        ivector_per_file = []
        for file in range(len(train_paths[speaker])):
            rate, sig = spiowav.read(train_paths[speaker][file])

            if (rate != 16000):
                resampled = signal.resample(sig, 16000)

            ################################## AICI TREBUIE VAD ####################################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!

            features = mfcc.coeff_from_signal(resampled, 16000)
            logLikelihood = baum_welch.logLikelihood(features, ubm_means, ubm_weights, ubm_covs)
            n, f, s = baum_welch.compute_statistics(logLikelihood, features)

            f = np.subtract(f, np.multiply(np.transpose(np.repeat(n[:, np.newaxis],np.shape(ubm_means)[1],axis=1)),np.transpose(ubm_means)))

            i1 = np.matmul(np.transpose(np.multiply(TS,np.repeat(np.repeat(n, np.shape(features)[1])[:,np.newaxis], numTdim, axis = 1))), T_matrix)
            i2 = np.matmul(np.linalg.pinv(np.add(I,i1)),TSi)
            i3 = []
            for i in range(np.shape(f)[1]):
                if i == 0:
                    i3 = np.transpose(f)[i]
                else:
                    i3 = np.concatenate((i3,np.transpose(f)[i]),axis =0)
            ivector_per_file.append(np.matmul(i2,i3))
        # i-vector per file appended to i-vector per speaker list
        ivector_per_speaker.append(ivector_per_file)





#################### LOAD PATHS ########################
train_paths, test_paths = ubm.get_train_test_paths(r"D:\licenta\SPEECH DATA")

################## LOAD UBM DATA #######################
f = open("ubm_means.txt", "rb")
ubm_means = np.load(f)
f.close

f = open("ubm_covs.txt", "rb")
ubm_covs = np.load(f)
f.close

f = open("ubm_weights.txt", "rb")
ubm_weights = np.load(f)
f.close

############### LOAD T MATRIX ###################
f = open("T_matrix.txt", "rb")
T = np.load(f)
f.close

f = open("N_stat.txt", "rb")
N = np.load(f)
f.close

f = open("F_stat.txt", "rb")
F = np.load(f)
f.close


extract_i_vectors(ubm_covs,ubm_means,ubm_weights, train_paths, T, 128)

speakers_list = train_paths.keys()