    import os
import mfcc
import numpy as np
import gmm
import baum_welch
import G_PLDA as gplda
import i_vector


class TrainClass(object):
    def __init__(self, root_directory, num_mfcc, num_gaussians, num_t_space, num_t_iter):
        self.root_directory = root_directory
        self.num_mfcc = num_mfcc
        self.num_gaussians = num_gaussians
        self.num_T_space = num_t_space
        self.num_T_iter = num_t_iter

    def get_train_test_paths(self):
        train_data = {}
        test_data = {}

        path_train = []
        path_test = []

        test_list = ["F09", "F10", "M09", "M10"]

        for entry in os.scandir(self.root_directory):
            for newEntry in os.scandir(entry):
                if (newEntry.name == 'MIC'):
                    for newNewEntry in os.scandir(newEntry):
                        for files in os.scandir(newNewEntry):
                            if (files.path.endswith('.wav')):
                                if (newNewEntry.name in test_list):
                                    path_test.append(files.path)
                                else:
                                    path_train.append(files.path)
                        if (newNewEntry.name in test_list):
                            test_data[newNewEntry.name] = []
                            test_data[newNewEntry.name].extend(path_test)
                            path_test.clear()
                        else:
                            train_data[newNewEntry.name] = []
                            train_data[newNewEntry.name].extend(path_train)
                            path_train.clear()

        return train_data, test_data

    def get_utterance_per_speaker(self, dict_paths):
        keys_list = dict_paths.keys()
        utterances = {}
        for i in keys_list:
            utterances[i] = len(dict_paths[i])
        return utterances



train = TrainClass(r"D:\licenta\SPEECH DATA", 13, 128, 128, 20)
train_paths, test_paths = train.get_train_test_paths()

############################################### EXTRACT FEATURES #####################################################

#train_features = mfcc.extract_all_features(train_paths, train.num_mfcc)
"""
f = open("train_features.txt", "wb")
np.save(f, train_features)
f.close
"""
#f = open("train_features.txt", "rb")
#features_list = np.load(f)
#f.close

################################################### UBM ##############################################################
"""
ubm = gmm.gmm_params(features_list, train.num_gaussians)
gmm.save_ubm(ubm)
"""
loaded_ubm = gmm.load_ubm()

########################################### BAUM WELCH STATISTICS ####################################################
"""
N, F = baum_welch.Baum_Welch_Statistic(train_paths, train.num_mfcc, loaded_ubm)
baum_welch.save_statistics(N, F)
"""
n, f = baum_welch.load_statistics()

############################################### I-VECTORS ############################################################
#i_vector.Total_Variability_Space(n, f, loaded_ubm, train.num_T_space, train.num_T_iter, 3*train.num_mfcc,
#                                 train.num_gaussians)
#matrix = i_vector.load_T_matrix()
#i_vector.extract_i_vectors(loaded_ubm, train_paths, matrix, train.num_T_space, train.num_mfcc)
#print("ready")

#f = open("F05_ivector.txt", "rb")
#ivectors = np.load(f)
#f.close

keys_list = train_paths.keys()
ivectors = []
for i in keys_list:
    filename = str(i) + "_ivector.txt"
    f = open(filename, "rb")
    ivectors.append(np.load(f))
    f.close()

############################################ PROJECTION MATRIX #######################################################
"""
utterances = train.get_utterance_per_speaker(train_paths)
p_mat, iv = i_vector.LDA_projection_matrix(ivectors)

projection_matrix = i_vector.WCCN_projection_matrix(p_mat, iv, utterances)

f = open("projection_matrix.txt", "wb")
np.save(f, projection_matrix)
f.close
"""
f = open("projection_matrix.txt", "rb")
p_matrix = np.load(f)
f.close

############################################### TRAIN GPLDA ##########################################################

utterances = train.get_utterance_per_speaker(train_paths)
gplda.train_model(p_matrix, ivectors, utterances, 5)



"""
f = open("ivectors.txt", "rb")
ivector1 = np.load(f, allow_pickle=True)
f.close

f = open("ivectors2.txt", "rb")
ivector2 = np.load(f, allow_pickle=True)
f.close

p_mat, iv, dim = i_vector.LDA_projection_matrix(ivector1, ivector2)
i_vector.WCCN_projection_matrix(p_mat, iv, dim)

print("Ready")
"""