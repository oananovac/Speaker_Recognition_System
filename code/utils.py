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