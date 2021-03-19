# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:57:19 2021

@author: Acer
"""

import python_speech_features
import wave
import contextlib
import numpy as np
import speechpy
from sklearn.mixture import GaussianMixture

def mfcc_features():
    with contextlib.closing(wave.open('../resources/final.wav', 'rb')) as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        signal = np.fromstring(pcm_data, "Int16")

       # pre_emphasis = 0.97
       # emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        mfcc = python_speech_features.mfcc(signal, samplerate = sample_rate)


        #mfcc_means = np.round(mfcc.mean(axis=0), 3)

        #for i in range(len(mfcc_means)):
        #    print ("vectorul " + str(i))
        #    print(mfcc_means[i])

        #index = 0
        """
        f = open("coeff.txt","w")
        
        #print("Mfcc coefficients:")
        f.write("Mfcc coefficients:\n")
        for i in mfcc:
            f.write("vectorul " + str(index))
            f.write(str(i)  + "\n")
            #print ("vectorul " + str(index))
            #print(i)
            index +=1
         
        index = 0
       # print("Delta coefficients:\n") 
        f.write("Delta coefficients:\n")
        for i in deltas:
            f.write("vectorul " + str(index))
            f.write(str(i) + "\n")
            #print ("vectorul " + str(index))
            #print(i)
            index +=1
            
        index = 0
        #print("Delta delta coefficients:\n") 
        f.write("Delta delta coefficients:\n")
        for i in delta_delta:
            f.write("vectorul " + str(index))
            f.write(str(i) + "\n")
            #print ("vectorul " + str(index))
            #print(i)
            index +=1
            
            
        """

        norm = speechpy.processing.cmvn(mfcc)
        deltas = python_speech_features.delta(norm, 2)
        delta_delta = python_speech_features.delta(deltas, 2)
        """
        feature_cube = speechpy.feature.extract_derivative_feature(norm)
       # fea = feature_cube.transpose(2,0,1)#.reshape(-1,39)
        fea = np.ravel(feature_cube, order = 'K')
        
        print(norm)
        print("Cubul")
        print(feature_cube)
        print(fea)
        
        """

        final_coef = [norm]
        final_coef.append(deltas)
        final_coef.append(delta_delta)

        final = np.hstack(final_coef)


        """
        
        print(final)
        print(final[0])
        
        print("Frames: " + str(len(final)) + "\nCoeff: " + str(len(final[0])))
        
        """

        return final

def coeff_from_signal(signal, sample_rate):
    mfcc = python_speech_features.mfcc(signal, samplerate=sample_rate)
    norm = speechpy.processing.cmvn(mfcc)
    deltas = python_speech_features.delta(norm, 2)
    delta_delta = python_speech_features.delta(deltas, 2) # 2 este nr de frame care se ia in considerare in fata si in spatele frame ului curent

    final_coef = [norm]
    final_coef.append(deltas)
    final_coef.append(delta_delta)

    final = np.hstack(final_coef)
    return final

def gmm_params(features_list, n_components): # n comp e 16
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=500, n_init=3, verbose=1)
    gmm.fit(features_list)  # se face estimarea parametrilor modelului folosing algortimul EM a.i. sa se obtina maximum likelihood
    #print(gmm.n_components)

    means = gmm.means_
    covs = gmm.covariances_
    weights = gmm.weights_
    #print(means.shape)
    #print(covs.shape)
    #print(weights.shape)

    total =0
    for i in range(0, len(weights)):
        total = total + weights[i]

    print(total)
    return means, covs, weights

"""

"""