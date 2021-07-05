import numpy as np


def reshape_ivectors_per_speaker_utterances(ivectors, utterances):
    num_eigen_vectors = len(ivectors)

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

    return ivv


def get_unique_values_sorted(utterance_per_speaker):
    unique_utterances_number = []
    for value in utterance_per_speaker.values():
        if value in unique_utterances_number:
            continue
        else:
            unique_utterances_number.append(value)

    return sorted(unique_utterances_number)


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


def load_ivectors(train_paths, root_path, components_number):
    keys_list = train_paths.keys()
    ivectors = []
    for i in keys_list:
        filename = root_path + "/ivectors/" + i + "_ivector_" \
                   + str(components_number) + ".txt"
        f = open(filename, "rb")
        ivectors.append(np.load(f))
        f.close
    return ivectors


def from_dictionary_to_array(dictionary):
    array = []
    for key in dictionary:
        if len(array) == 0:
            array = dictionary[key]
        else:
            array = np.concatenate([array, dictionary[key]])
    return array