import enroll
import i_vector
import G_PLDA
import numpy as np

class VerifyClass(object):
    def __init__(self, signal_path, label):
        self.signal_path = signal_path
        self. label = label

    def get_current_ivector(self):
        enroll_obj = enroll.EnrollClass(r"D:\licenta\SPEECH DATA", 13)
        w = i_vector.extract_ivector_from_signal(enroll_obj.ubm, self.signal_path, enroll_obj.t_matrix, len(enroll_obj.t_matrix[0]), enroll_obj.num_mfcc)
        w = np.matmul(enroll_obj.projection_matrix, np.transpose(w))
        return w

    def get_label_model(self):
        filename = self.label + "_ivector_model.txt"
        f = open(filename, "rb")
        ivector = np.load(f)
        f.close
        return ivector

    def get_score(self):
        w_test = self.get_current_ivector()
        w_labeled = self.get_label_model()

        score = G_PLDA.get_gplda_score(w_labeled, w_test)
        return score


ver = VerifyClass(r"D:\licenta\SPEECH DATA\FEMALE\MIC\F09\mic_F09_si1991.wav", "F09")
score = ver.get_score()