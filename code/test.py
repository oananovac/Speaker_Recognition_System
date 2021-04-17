import mfcc
from sklearn.mixture import GaussianMixture
import numpy as np
import python_speech_features
import scipy.io.wavfile as spiowav
import struct
import contextlib
import wave
import soundfile as sf
from scipy import signal
import resampy
from sklearn import preprocessing
import gmm
import train
import pickle
import gmm
import baum_welch
from scipy.io import savemat
import i_vector
import math
import vad


def extract_mfcc_from_signal(signal, sample_rate, num_coeff):
    val_nfft = int(math.pow(2, math.ceil(math.log(48000 * 0.025, 2))))
    mfcc = python_speech_features.mfcc(signal, samplerate=sample_rate, numcep=num_coeff, winfunc=np.hanning, nfft=val_nfft)
    norm_mfcc = preprocessing.scale(mfcc)  # , axis=1)
    deltas = python_speech_features.delta(norm_mfcc, 2)
    double_deltas = python_speech_features.delta(deltas, 2)
    combined = np.hstack((norm_mfcc, deltas, double_deltas))

    return combined


def extract_all_features(paths_dictionary, num_mfcc):
    ff = np.asarray(())
    keys_list = paths_dictionary.keys()
    for i in keys_list:
        audioNr = len(paths_dictionary[i])
        for j in range(audioNr):
            vad_obj = vad.Vad(paths_dictionary[i][j], 2)
            # signal, rate = sf.read(paths_dictionary[i][j])
            signal, rate = vad_obj.get_speech_signal()
            features = extract_mfcc_from_signal(signal, 48000, num_mfcc)
            if ff.size == 0:
                ff = features
            else:
                ff = np.vstack((ff, features))
    return ff


def ivector_prototype(ubm, T_matrix):
    sigma = np.zeros(shape=(len(ubm.covs) * len(ubm.covs[0])))
    k = 0
    for i in range(len(ubm.covs[0])):
        for j in range(len(ubm.covs)):
            sigma[k] = ubm.covs[j][i]
            k += 1

    list = []
    list.append('D:\licenta\SPEECH DATA\FEMALE\MIC\F01\mic_F01_sa1.wav')
    list.append('D:\licenta\SPEECH DATA\FEMALE\MIC\F01\mic_F01_sa2.wav')
    list.append('D:\licenta\SPEECH DATA\FEMALE\MIC\F01\mic_F01_si453.wav')

    ivector_per_file = []

    rep_sigma = np.repeat(np.transpose(sigma)[:, np.newaxis], 128, axis=1)
    TS = np.divide(T_matrix, rep_sigma)
    TSi = np.transpose(TS)
    I = np.eye(128, dtype=float)

    for i in range(len(list)):
        signal, rate = sf.read(list[i])
        sig = resampy.resample(signal, 3, 1)

        features = extract_mfcc_from_signal(sig, 16000, 20)
        logLikelihood = baum_welch.logLikelihood(features, ubm)
        n, f, s = baum_welch.compute_statistics(logLikelihood, features)

        ff = np.subtract(f, np.multiply(np.transpose(np.repeat(n[:, np.newaxis], np.shape(ubm.means)[1], axis=1)),
                                        np.transpose(ubm.means)))
        rep = np.multiply(TS, np.repeat(np.transpose(np.repeat(n, np.shape(features)[1]))[:, np.newaxis], 128, axis=1))
        i1 = np.matmul(np.transpose(
            np.multiply(TS, np.repeat(np.transpose(np.repeat(n, np.shape(features)[1]))[:, np.newaxis], 128, axis=1))),
            T_matrix)
        i2 = np.matmul(np.linalg.pinv(np.add(I, i1)), TSi)
        i3 = []
        for i in range(np.shape(ff)[1]):
            if i == 0:
                i3 = np.transpose(ff)[i]
            else:
                i3 = np.concatenate((i3, np.transpose(ff)[i]), axis=0)
        ivector_per_file.append(np.matmul(i2, i3))

    return ivector_per_file


#rate, sig = spiowav.read(r'D:\licenta\SPEECH DATA\FEMALE\MIC\F01\mic_F01_sa1.wav')
"""decimal_signal = np.empty(int(len(sig)/2), dtype=object)
for i in range(len(sig)):
    decimal_signal[i] = struct.unpack('n', sig[2*i:2*i+2])[0]
    if 2 * i + 2 == len(sig):
        break

sig,rate = sf.read(r'D:\licenta\SPEECH DATA\FEMALE\MIC\F01\mic_F01_sa1.wav')
f = resampy.resample(sig, 3, 1)
extract_mfcc_from_signal(f, 16000, 20)
"""

train = train.TrainClass(r"D:\licenta\SPEECH DATA", 13, 128, 128, 20)
train_paths, test_paths = train.get_train_test_paths()

train_fea = extract_all_features(train_paths, train.num_mfcc)
f = open("train_features.txt", "wb")
np.save(f, train_fea)
f.close

#f = open("train_features1.txt", "rb")
#features_list = np.load(f)
#f.close

#gmm = GaussianMixture(n_components=128, covariance_type='diag', max_iter=500, n_init=3, verbose=1)
#gmm.fit(features_list)
#picklefile = "ubm.gmm"
#pickle.dump(gmm, open(picklefile, 'w'))
#model = pickle.load(open(picklefile, 'r'))

#means = gmm.means_
#covs = gmm.covariances_
#weights = gmm.weights_

f = open("ubm_means1.txt", "rb")
means = np.load(f)
f.close

f = open("ubm_covs1.txt", "rb")
covs = np.load(f)
f.close

f = open("ubm_weights1.txt", "rb")
weights = np.load(f)
f.close


#sig1,rate1 = sf.read(r'D:\licenta\SPEECH DATA\FEMALE\MIC\F01\mic_F01_sa1.wav')
#f = resampy.resample(sig1, 3, 1)

#fea  = extract_mfcc_from_signal(f,16000,20)

#aa = { "fea11": fea, "label": "fea" }
#savemat("fea.mat", aa)
"""
mean = {"me": means, "label": "m"}
savemat("me.mat", mean)
we = {"we": weights, "label": "w"}
savemat("we.mat", we)
co = {"co": covs, "label": "c"}
savemat("co.mat", co)
"""
ubm = gmm.Gmm(weights, means, covs)
#log = baum_welch.logLikelihood(fea, ubm)
#n, f, s = baum_welch.compute_statistics(log, fea)


"""
list = []
list.append('D:\licenta\SPEECH DATA\FEMALE\MIC\F01\mic_F01_sa1.wav')
list.append('D:\licenta\SPEECH DATA\FEMALE\MIC\F01\mic_F01_sa2.wav')
list.append('D:\licenta\SPEECH DATA\FEMALE\MIC\F01\mic_F01_si453.wav')

feaa = []
for i in list:
    sigg, ratee = sf.read(i)
    f = resampy.resample(sigg, 3, 1)
    fea = extract_mfcc_from_signal(f, 16000, 20)
    feaa.append(fea)

feaa1 = {"fe": feaa[0], "label": "f"}
savemat("f1.mat", feaa1)
feaa2 = {"fe": feaa[1], "label": "f"}
savemat("f2.mat", feaa2)
feaa3 = {"fe": feaa[2], "label": "f"}
savemat("f3.mat", feaa3)
N, F = baum_welch.Baum_Welch_Statistic(feaa, ubm)

"""
#N, F = baum_welch.Baum_Welch_Statistic(features_list, ubm)
# i_vector.Total_Variability_Space(N, F, ubm, train.num_T_space, train.num_T_iter, 60, train.num_gaussians)

"""
n1 = {"n": N, "label": "n"}
savemat("n1.mat", n1)
f1 = {"f": F, "label": "f"}
savemat("ff1.mat", f1)

#n, f = baum_welch.load_statistics()
n1 = np.dstack(N)
n1 = np.rollaxis(n1, -1)
f1 = np.dstack(F)
f1 = np.rollaxis(f1, -1)

a = [[1, 2], [2,8]]
aa = np.linalg.pinv(a)
"""
#i_vector.Total_Variability_Space(n1, f1, ubm, train.num_T_space, 2, 60, train.num_gaussians)

T = i_vector.load_T_matrix()
#t = {"norm": T, "label": "t"}
#savemat("normT.mat", t)

#aaa = ivector_prototype(ubm,T)

val_nfft = math.pow(2, math.ceil(math.log(48000 * 0.025, 2)))

print("aa")