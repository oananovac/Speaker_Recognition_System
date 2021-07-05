import numpy as np
from scipy.linalg import eigh
import voice_activity_detector
import features_extraction
import statistics
import utils


def get_sigma(ubm, space_dimension):
    sigma = np.zeros(shape=(len(ubm.covariances) * len(ubm.covariances[0])))

    k = 0
    for i in range(len(ubm.covariances[0])):
        for j in range(len(ubm.covariances)):
            sigma[k] = ubm.covariances[j][i]
            k += 1

    repeat_sigma = np.repeat(np.transpose(sigma)[:, np.newaxis],
                             space_dimension, axis=1)

    return repeat_sigma


def save_i_vector_model(path, i_vector, speaker, components_number):
    f = open(
        path + "/ivectors/" + speaker + "_ivector_model_" +
        str(components_number) + ".txt",
        "wb")
    np.save(f, i_vector)
    f.close


def load_i_vector_model(path, speaker, components_number):
    f = open(
        path + "/ivectors/" + speaker + "_ivector_model_" +
        str(components_number) + ".txt",
        "rb")
    i_vector = np.load(f)
    f.close
    return i_vector


def save_i_vectors(path, i_vectors, speaker, components_number):
    f = open(
        path + "/ivectors/" + speaker + "_ivector_" + str(
            components_number) +
        ".txt",
        "wb")
    np.save(f, i_vectors)
    f.close


def extract_i_vector_from_signal(ubm, utterance_path, t_matrix,
                                 space_dimension,
                                 mfcc_number, frame_duration, step_duration,
                                 sigma):

    t_matrix_divides_sigma = np.divide(t_matrix, sigma)
    t_matrix_divides_sigma_transpose = np.transpose(t_matrix_divides_sigma)
    identity_matrix = np.eye(space_dimension, dtype=float)
    vad_object = voice_activity_detector.Vad(utterance_path, 2)
    signal_samples, sample_rate = vad_object.get_speech_signal()
    del vad_object

    mfcc = features_extraction.FeaturesExtraction(mfcc_number, True,
                                                  frame_duration,
                                                  step_duration)
    features = mfcc.extract_mfcc_from_signal(signal_samples, sample_rate)

    log_likelihood = statistics.log_likelihood_computation(features, ubm)
    n, f, s = statistics.statistics_computation(log_likelihood, features)

    # first order statistics are centered by the mean vector
    f = np.subtract(f, np.multiply(np.transpose(
        np.repeat(n[:, np.newaxis], np.shape(ubm.means)[1], axis=1)),
        np.transpose(ubm.means)))

    # i-vector computation

    i1 = np.matmul(np.transpose(
        np.multiply(t_matrix_divides_sigma,
                    np.repeat(
                        np.transpose(np.repeat(n, np.shape(features)[1]))[:,
                        np.newaxis], space_dimension, axis=1))), t_matrix)

    i2 = np.matmul(np.linalg.pinv(np.add(identity_matrix, i1)),
                   t_matrix_divides_sigma_transpose)

    i3 = []
    for i in range(np.shape(f)[1]):
        if i == 0:
            i3 = np.transpose(f)[i]
        else:
            i3 = np.concatenate((i3, np.transpose(f)[i]), axis=0)

    i_vector = np.matmul(i2, i3)

    return i_vector


def extract_i_vectors(path, ubm, train_paths, t_matrix, space_dimension,
                      mfcc_number, frame_duration, step_duration,
                      components_number):
    sigma = get_sigma(ubm, space_dimension)

    speakers_list = train_paths.keys()

    ivectors = {}
    for speaker in speakers_list:
        ivector_per_file = []
        for file in range(len(train_paths[speaker])):
            ivector_per_file.append(extract_i_vector_from_signal(ubm,
                                                                 train_paths[speaker][file],
                                                                 t_matrix,
                                                                 space_dimension,
                                                                 mfcc_number,
                                                                 frame_duration,
                                                                 step_duration,
                                                                 sigma))
        i_vectors = np.transpose(np.dstack(ivector_per_file)[0])
        # ivectors[speaker] = i_vectors
        save_i_vectors(path, i_vectors, speaker, components_number)


def LDA_projection_matrix(ivectors):  # LDA projection matrix
    ivector_list = ivectors

    cat_list = utils.concatenate_ivectors(ivector_list)

    projection_matrix = np.identity(len(ivector_list[0][0]))

    num_eigen_vectors = len(ivector_list)
    sw = np.zeros(np.shape(projection_matrix))
    sb = np.zeros(np.shape(projection_matrix))

    wbar = np.mean(cat_list, axis=0)

    for lists in ivector_list:
        ws = lists
        wsbar = np.mean(ws, axis=0)
        ws_sub = np.reshape(np.subtract(wsbar, wbar), (np.shape(wbar)[0], 1))
        ws_mul = np.matmul(ws_sub, np.transpose(ws_sub))

        sb = np.add(sb, ws_mul)

        ws_cov = np.cov(np.transpose(ws), bias=True)
        sw = np.add(sw, ws_cov)

    eigvals, eigvecs = eigh(sb, sw, eigvals_only=False)
    zipped_eig = zip(eigvals, eigvecs)
    sorted_zipped_eig = sorted(zipped_eig, reverse=True)
    sortedd = [element for _, element in sorted_zipped_eig]
    a_matrix = []

    for i in range(num_eigen_vectors):
        a_matrix.append(sortedd[i])
    a_matrix = np.dstack(a_matrix)
    a_matrix = np.rollaxis(a_matrix[0], -1)

    a_matrix = np.divide(a_matrix, np.repeat(
        np.linalg.norm(a_matrix, axis=1)[:, np.newaxis], len(a_matrix[0]),
        axis=1))

    ivectors_fin = np.matmul(a_matrix, np.transpose(cat_list))
    projection_matrix = np.matmul(a_matrix, projection_matrix)

    return projection_matrix, ivectors_fin


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

    w_ = np.divide(w_, np.full((np.shape(w_)[0], np.shape(w_)[1]),
                               num_eigen_vectors))
    w_ = np.add(
        np.multiply(np.full((np.shape(w_)[0], np.shape(w_)[1]), 1 - alpha),
                    w_),
        np.multiply(np.full((np.shape(w_)[0], np.shape(w_)[1]), alpha),
                    np.identity(np.shape(w_)[0])))

    b_matrix = np.linalg.cholesky(
        np.linalg.pinv(w_))  # nearestPD(np.linalg.pinv(w_)))
    wccn_projection_matrix = np.matmul(b_matrix, lda_projection_matrix)

    return wccn_projection_matrix


def load_projection_matrix(path, components_number):
    f = open(path + "/models/projection_matrix_" + str(components_number) +
             ".txt",
             "rb")
    p_matrix = np.load(f)
    f.close

    return p_matrix


def save_projection_matrix(path, components_number, p_matrix):
    f = open(path + "/models/projection_matrix_" + str(components_number) +
             ".txt",
             "wb")
    np.save(f, p_matrix)
    f.close
