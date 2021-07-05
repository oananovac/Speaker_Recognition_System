import numpy as np
import utils
import pickle


class GPLDAModel(object):
    def __init__(self, i_vector_mean, whitening_matrix, eigen_voices, sigma):
        self.i_vector_mean = i_vector_mean
        self.whitening_matrix = whitening_matrix
        self.eigen_voices = eigen_voices
        self.sigma = sigma


def whitening_matrix(i_vector_matrix):  # ZCA whitening
    covariance_i_vectors = np.cov(np.transpose(i_vector_matrix))
    u_matrix, singular_values, v_matrix = np.linalg.svd(covariance_i_vectors)
    w_matrix = np.matmul(np.diag(np.divide(1, np.sqrt(singular_values) +
                                           np.spacing(1))), v_matrix)
    return w_matrix


def sort_ivectors_by_utterances(ivectors_list, utterance_per_speaker):
    sorted_values = utils.get_unique_values_sorted(utterance_per_speaker)

    # the first order moment for speakers
    f = np.zeros((len(ivectors_list), len(ivectors_list)))

    sorted_ivectors = []
    index = 0
    for i in range(len(sorted_values)):
        sorted_iv = []

        for j in range(len(ivectors_list)):
            shape = np.shape(ivectors_list[j])[1]
            if shape == sorted_values[i]:
                sorted_iv.append(ivectors_list[j])
                f[index] = np.sum(ivectors_list[j], axis=1)
                index += 1

        sorted_ivectors.append(sorted_iv)

    return sorted_ivectors, sorted_values, f


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def em_algorithm(num_eigen_voices, speakers_number, S, iterations_number,
                 unique_values,
                 ivectors_sorted, f):
    eigen_voices_matrix = np.random.rand(num_eigen_voices, num_eigen_voices)
    lambda_ = np.linalg.pinv(S / speakers_number)

    for iter in range(iterations_number):
        # E
        gamma = np.zeros((num_eigen_voices, num_eigen_voices))
        all_ey = np.zeros((num_eigen_voices, num_eigen_voices))
        r_matrix = np.zeros((num_eigen_voices, num_eigen_voices))

        index = 0
        for i in range(len(unique_values)):
            ivector_len = unique_values[i]
            iv = ivectors_sorted[i]

            m_matrix = np.linalg.pinv(np.add(ivector_len * np.matmul(
                np.transpose(eigen_voices_matrix),
                np.matmul(lambda_, eigen_voices_matrix)),
                                             np.eye(num_eigen_voices)))

            for speaker in range(len(iv)):
                ey = np.matmul(np.matmul(np.matmul(m_matrix, np.transpose(
                    eigen_voices_matrix)), lambda_), np.transpose(f[index]))
                ey_transpose = np.transpose(ey[np.newaxis])

                eyy = np.matmul(ey_transpose, np.transpose(ey_transpose))

                r_matrix = np.add(r_matrix, ivector_len * np.add(m_matrix,
                                                                 eyy))
                all_ey[index] = ey
                index += 1

                # minimum divergence
                gamma = np.add(gamma, np.add(m_matrix, eyy))

        # M
        t_matrix = np.matmul(np.transpose(all_ey), f)
        eigen_voices_matrix = np.matmul(np.transpose(t_matrix),
                                        np.linalg.pinv(r_matrix))
        lambda_ = np.linalg.pinv(np.subtract(S, np.matmul(
            eigen_voices_matrix, t_matrix)) / speakers_number)

        # minimum divergence
        gamma = gamma / num_eigen_voices
        if is_pos_def(gamma):
            eigen_voices_matrix = np.matmul(eigen_voices_matrix,
                                            np.linalg.cholesky(gamma))

    return eigen_voices_matrix, lambda_


def save_gplda_model(path, gplda_object, components_number):
    filename = open(path + "/models/gplda_model_" + str(components_number) +
                    ".obj",
                    'wb')
    pickle.dump(gplda_object, filename)
    print("Ready")


def load_gplda_model(path, components_number):
    filename = open(path + "/models/gplda_model_" + str(components_number) +
                    ".obj",'rb')
    model = pickle.load(filename)
    return model


def train_model(path, components_number, projection_matrix,
                i_vectors_per_speaker, utterance_per_speaker,
                iterations_number):

    i_vectors_list_projected = []

    for i in i_vectors_per_speaker:
        i_vectors_list_projected.append(np.transpose(np.matmul(
            projection_matrix, np.transpose(i))))

    num_eigen_voices = len(i_vectors_per_speaker)
    i_vector_matrix = utils.concatenate_ivectors(i_vectors_list_projected)

    speakers_number = len(i_vector_matrix)

    i_vectors_mean = np.mean(np.transpose(i_vector_matrix), axis=1)
    i_vector_matrix = np.subtract(i_vector_matrix, i_vectors_mean)

    w_matrix = whitening_matrix(i_vector_matrix)
    i_vector_matrix = np.matmul(w_matrix, np.transpose(i_vector_matrix))
    i_vector_matrix = np.divide(i_vector_matrix, np.linalg.norm(
        i_vector_matrix, axis=0))

    # second order moment
    S = np.matmul(i_vector_matrix, np.transpose(i_vector_matrix))

    ivectors = utils.reshape_ivectors_per_speaker_utterances(
        i_vector_matrix, utterance_per_speaker)
    sorted_ivectors, unique_values, f = sort_ivectors_by_utterances(
        ivectors, utterance_per_speaker)

    eigen_voices_matrix, lambda_ = em_algorithm(num_eigen_voices,
                                                speakers_number, S,
                                                iterations_number,
                                                unique_values, sorted_ivectors,
                                                f)

    object_model = GPLDAModel(i_vectors_mean, w_matrix, eigen_voices_matrix,
                              np.linalg.pinv(lambda_))
    save_gplda_model(path, object_model, components_number)


def prepare_i_vector_for_scoring(w, model):
    # center data
    w = np.subtract(w, model.i_vector_mean)

    # whiten data
    w = np.matmul(model.whitening_matrix, w)
    w = w / np.linalg.norm(w)
    return w


def compute_gplda_score(model, w1, w2):
    w1 = prepare_i_vector_for_scoring(w1, model)
    w2 = prepare_i_vector_for_scoring(w2, model)

    product_vv = np.matmul(model.eigen_voices, np.transpose(
        model.eigen_voices))
    product_vv_plus_sigma = np.add(model.sigma, product_vv)

    c1 = np.concatenate((product_vv_plus_sigma, product_vv), axis=1)
    c2 = np.concatenate((product_vv, product_vv_plus_sigma), axis=1)

    term1 = np.linalg.pinv(np.concatenate((c1, c2), axis=0))
    term2 = np.linalg.pinv(product_vv_plus_sigma)

    w1w2 = np.concatenate((w1, w2), axis=0)
    t1 = np.matmul(np.matmul(np.transpose(w1w2), term1), w1w2)
    t2 = np.matmul(np.matmul(np.transpose(w1), term2), w1)
    t3 = np.matmul(np.matmul(np.transpose(w2), term2), w2)
    score = np.subtract(np.subtract(t1, t2), t3)

    return score
