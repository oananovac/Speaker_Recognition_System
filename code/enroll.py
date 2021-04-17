import os
import i_vector
import gmm
import numpy as np
import vad
import mfcc
import baum_welch


class EnrollClass(object):
    def __init__(self, root_directory, num_mfcc):
        self.root_directory = root_directory
        self.ubm = self.get_ubm()
        self.t_matrix = self.get_t_matrix()
        self.projection_matrix = self.get_projection_matrix()
        self.num_mfcc = num_mfcc

    def get_enroll_paths(self):
        enroll_paths_per_speaker = {}
        enroll_paths = []
        enroll_list = ["F09", "M09"]
        for entry in os.scandir(self.root_directory):
            for newEntry in os.scandir(entry):
                if (newEntry.name == 'MIC'):
                    for newNewEntry in os.scandir(newEntry):
                        for files in os.scandir(newNewEntry):
                            if (files.path.endswith('.wav')):
                                if (newNewEntry.name in enroll_list):
                                    enroll_paths.append(files.path)
                        if newNewEntry.name in enroll_list:
                            enroll_paths_per_speaker[newNewEntry.name] = []
                            enroll_paths_per_speaker[newNewEntry.name].extend(enroll_paths)
                            enroll_paths.clear()

        return enroll_paths_per_speaker

    def get_t_matrix(self):
        return i_vector.load_T_matrix()

    def get_ubm(self):
        return gmm.load_ubm()

    def get_projection_matrix(self):
        return i_vector.load_projection_matrix()

    def save_ivector_model(self, ivector_model, label):
        f = open(label + "_ivector_model.txt", "wb")
        np.save(f, ivector_model)
        f.close

    def enroll_speaker(self, label, list_utt):
        ivector_mat = np.zeros((len(list_utt), len(self.projection_matrix)))
        index = 0
        for utt in list_utt:

            w = i_vector.extract_ivector_from_signal(self.ubm, utt, self.t_matrix, len(self.t_matrix[0]), self.num_mfcc)
            w = np.matmul(self.projection_matrix,np.transpose(w))
            ivector_mat[0] = w
            index += 1

        ivector_model = np.mean(ivector_mat, 0)
        self.save_ivector_model(ivector_model, label)

"""
enrollObj = EnrollClass(r"D:\licenta\SPEECH DATA", 13)
ll = enrollObj.get_enroll_paths()

list_utt_speaker = []
labels = list(ll.keys())
aa = labels[0]
for j in labels:
    for i in range(3):
        list_utt_speaker.append(ll[j][i])
    enrollObj.enroll_speaker(j, list_utt_speaker)
    list_utt_speaker.clear()

print("Ready")
"""