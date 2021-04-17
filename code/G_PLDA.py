import numpy as np
import i_vector
import utils
from scipy.io import savemat
import pickle


class GPLDA_model:
    def __init__(self, mu, whitening_matrix, eigen_voices, sigma):
        self.mu = mu
        self.whitening_matrix = whitening_matrix
        self.eigen_voices = eigen_voices
        self.sigma = sigma


def whitening_matrix(ivector_matrix): # ZCA whitening
    S = np.cov(np.transpose(ivector_matrix))
    u, s, vh = np.linalg.svd(S)
    W = np.matmul(np.diag(np.divide(1, np.sqrt(s) + np.spacing(1))), vh)
    return W


def sort_ivectors_by_utterances(ivectors_list, utterance_per_speaker):

    unique_nr_utt = []
    for val in utterance_per_speaker.values():
        if val in unique_nr_utt:
            continue
        else:
            unique_nr_utt.append(val)

    unique_nr_utt = sorted(unique_nr_utt)
    f = np.zeros((len(ivectors_list), len(ivectors_list)))

    sortedd = []
    index = 0
    for i in range(len(unique_nr_utt)):
        sorted_iv = []
        for j in range(len(ivectors_list)):
            shape = np.shape(ivectors_list[j])[1]
            if shape == unique_nr_utt[i]:
                sorted_iv.append(ivectors_list[j])
                f[index] = np.sum(ivectors_list[j], axis=1)
                index += 1
        sortedd.append(sorted_iv)

    return sortedd, unique_nr_utt, f


def EM_algorithm(num_eigen_voices, N, S, num_iter, unique_len, ivectors_sorted, f):
    V = np.random.rand(num_eigen_voices, num_eigen_voices)
    lambda_ = np.linalg.pinv(S/N)


    for iter in range(num_iter):
        # E
        gamma = np.zeros((num_eigen_voices, num_eigen_voices))
        EyTotal = np.zeros((num_eigen_voices, num_eigen_voices))
        R = np.zeros((num_eigen_voices, num_eigen_voices))

        idx = 0
        for i in range(len(unique_len)):
            ivector_len = unique_len[i]
            iv = ivectors_sorted[i]

            M = np.linalg.pinv(np.add(ivector_len * np.matmul(np.transpose(V), np.matmul(lambda_, V)), np.eye(num_eigen_voices)))

            for speaker in range(len(iv)):
                Ey = np.matmul(np.matmul(np.matmul(M, np.transpose(V)), lambda_), np.transpose(f[idx]))
                Ey1 = np.transpose(Ey[np.newaxis])
                Eyy = np.matmul(Ey1, np.transpose(Ey1))
                R = np.add(R, ivector_len * np.add(M, Eyy))
                EyTotal[idx] = Ey
                idx += 1

                # minimum divergence
                gamma = np.add(gamma, np.add(M, Eyy))


        TT = np.matmul(np.transpose(EyTotal), f)

        # M
        V = np.matmul(np.transpose(TT), np.linalg.pinv(R))
        lambda_ = np.linalg.pinv(np.subtract(S, np.matmul(V, TT))/N)

        # minimum divergence
        gamma = gamma / num_eigen_voices
        V = np.matmul(V, np.linalg.cholesky(gamma))

    return V, lambda_


def save_gplda_model(gplda_object):
    filename = open('gplda_model.obj', 'wb')
    pickle.dump(gplda_object, filename)
    print("Ready")


def train_model(projection_matrix, ivectors_per_speaker, utterance_per_speaker, num_iter):
    ivectors_list_projected = []

    for i in ivectors_per_speaker:
        ivectors_list_projected.append(np.transpose(np.matmul(projection_matrix, np.transpose(i))))

    num_eigen_voices = len(ivectors_per_speaker)
    ivector_matrix = i_vector.concatenate_ivectors(ivectors_list_projected)

    N = len(ivector_matrix)
    mu = np.mean(np.transpose(ivector_matrix), axis=1)
    ivector_matrix = np.subtract(ivector_matrix, mu)
    w_matrix = whitening_matrix(ivector_matrix)
    ivector_matrix = np.matmul(w_matrix, np.transpose(ivector_matrix))
    ivector_matrix = np.divide(ivector_matrix, np.linalg.norm(ivector_matrix, axis=0))

    S = np.matmul(ivector_matrix, np.transpose(ivector_matrix))
    ivectors = utils.reshape_ivectors_per_speaker_utterances(ivector_matrix, utterance_per_speaker)
    sorted_ivectors, unique_len, f = sort_ivectors_by_utterances(ivectors, utterance_per_speaker)

    V, lambda_ = EM_algorithm(num_eigen_voices, N, S, num_iter, unique_len, sorted_ivectors, f)

    object_model = GPLDA_model(mu, w_matrix, V, np.linalg.pinv(lambda_))
    save_gplda_model(object_model)


def get_gplda_score(w1, w2):
    modelfile = open('gplda_model.obj', 'rb')
    model = pickle.load(modelfile)
    """
    mean = {"w1": w1, "label": "m"}
    savemat("w1.mat", mean)
    mean = {"w2": w2, "label": "m"}
    savemat("w2.mat", mean)
    mean = {"mu": model.mu, "label": "m"}
    savemat("gmu.mat", mean)
    mean = {"wt": model.whitening_matrix, "label": "m"}
    savemat("wt.mat", mean)
    mean = {"gs": model.sigma, "label": "m"}
    savemat("gs.mat", mean)
    mean = {"ge": model.eigen_voices, "label": "m"}
    savemat("ge.mat", mean)
    """
    # center data
    w1 = np.subtract(w1, model.mu)
    w2 = np.subtract(w2, model.mu)

    # whiten data
    w1 = np.matmul(model.whitening_matrix, w1)
    w2 = np.matmul(model.whitening_matrix, w2)

    w1 = w1 / np.linalg.norm(w1)
    w2 = w2 / np.linalg.norm(w2)

    vvt = np.matmul(model.eigen_voices, np.transpose(model.eigen_voices))
    svvt = np.add(model.sigma, vvt)

    c1 = np.concatenate((svvt, vvt), axis=1)
    c2 = np.concatenate((vvt, svvt), axis=1)

    term1 = np.linalg.pinv(np.concatenate((c1, c2), axis=0))
    term2 = np.linalg.pinv(svvt)

    w1w2 = np.concatenate((w1, w2), axis=0)
    t1 = np.matmul(np.matmul(np.transpose(w1w2), term1), w1w2)
    t2 = np.matmul(np.matmul(np.transpose(w1), term2), w1)
    t3 = np.matmul(np.matmul(np.transpose(w2), term2), w2)
    score = np.subtract(np.subtract(t1, t2), t3)
    return score

