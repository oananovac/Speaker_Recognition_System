import python_speech_features
import numpy as np
import speechpy
from sklearn.mixture import GaussianMixture
from scipy import signal
import vad
import math
from sklearn import preprocessing


def extract_mfcc_from_signal(signal, sample_rate, num_coeff):
    val_nfft = int(math.pow(2, math.ceil(math.log(48000 * 0.025, 2))))
    mfcc = python_speech_features.mfcc(signal, samplerate=sample_rate, numcep=num_coeff, winfunc=np.hanning, nfft=val_nfft)
    norm_mfcc = preprocessing.scale(mfcc)  # folosit pentru un set de date (de dimensiune mare) in care acestea sunt
    # foarte raspandite, le centreaza pe 0 si varianta 1
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
            features = extract_mfcc_from_signal(signal, rate, num_mfcc)
            if ff.size == 0:
                ff = features
            else:
                ff = np.vstack((ff, features))
            del vad_obj

    return ff