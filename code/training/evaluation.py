import random
import numpy as np
import matplotlib.pyplot as plt
from intersect import intersection
from enum import Enum
import i_vector_extraction
import gplda_model
import utils
import records_files_handler
import universal_background_model
import total_variability_space
import enroll_and_verify


class Errors(Enum):
    FAR = "far"
    FRR = "frr"


class EvaluationClass(object):
    def __init__(self, components_number, mfcc_number, space_dimension):
        self.root_directory = None
        self.components_number = components_number
        self.mfcc_number = mfcc_number
        self.space_dimension = space_dimension

    def set_root_directory(self, root):
        self.root_directory = root

    def split_paths(self, evaluate_paths):
        enroll_paths = {}
        test_paths = {}
        speakers = evaluate_paths.keys()
        for speaker in speakers:
            random.shuffle(evaluate_paths[speaker])
            enroll_paths[speaker] = evaluate_paths[speaker][0:3]
            test_paths[speaker] = evaluate_paths[speaker][3:]

        return enroll_paths, test_paths

    def get_far_frr_paths(self, test_paths, speaker, rate_type):
        list_paths = []
        if rate_type == Errors.FRR:
            list_paths = test_paths[speaker]
        elif rate_type == Errors.FAR:
            for i in test_paths.keys():
                if i != speaker:
                    list_paths = np.concatenate((list_paths, test_paths[i]))

        return list_paths

    def compute_scores(self, model, test_paths, rate_type, ubm, t_matrix, \
                       num_mfcc,
                       projection_matrix):
        speakers = test_paths.keys()
        scores = {}

        sigma = i_vector_extraction.get_sigma(ubm, self.space_dimension)

        for speaker in speakers:
            list_paths = self.get_far_frr_paths(test_paths, speaker, rate_type)
            i_vector_speaker = i_vector_extraction.load_i_vector_model(
                self.root_directory, speaker, self.components_number)

            scores[speaker] = np.zeros(len(list_paths))
            index = 0

            for file in list_paths:
                w = i_vector_extraction.extract_i_vector_from_signal(ubm,
                                                                     file,
                                                                     t_matrix,
                                                                     len(
                                                                         t_matrix[
                                                                             0]),
                                                                     num_mfcc,
                                                                     0.025,
                                                                     0.01,
                                                                     sigma)
                w = np.matmul(projection_matrix, np.transpose(w))

                score = gplda_model.compute_gplda_score(model, w,
                                                        i_vector_speaker)

                scores[speaker][index] = score
                index += 1

        return scores

    def save_scores(self, error_type, array):
        f = open(self.root_directory + "/models/" + error_type.value +
                 "_scores_" + str(self.components_number) + ".txt", "wb")
        np.save(f, array)
        f.close

    def load_scores(self, error_type):
        f = open(self.root_directory + "/models/" + error_type.value +
                 "_scores_" + str(self.components_number) + ".txt", "rb")
        array = np.load(f)
        f.close
        return array

    def save_rates(self, error_type, array):
        f = open(self.root_directory + "/models/" + error_type.value +
                 "_rates_" + str(self.components_number) + ".txt", "wb")
        np.save(f, array)
        f.close

    def load_rates(self, error_type):
        f = open(self.root_directory + "/models/" + error_type.value +
                 "_rates_" + str(self.components_number) + ".txt", "rb")
        array = np.load(f)
        f.close
        return array

    def save_thresholds(self, array):
        f = open(self.root_directory + "/models/" + "_thresholds_" +
                 str(
                     self.components_number) + ".txt", "wb")
        np.save(f, array)
        f.close

    def load_thresholds(self):
        f = open(self.root_directory + "/models/" + "_thresholds_" +
                 str(self.components_number) + ".txt", "rb")
        array = np.load(f)
        f.close
        return array

    def compute_error_rates(self, model, test_paths, ubm, t_matrix,
                            mfcc_number,
                            projection_matrix):

        frr_scores_dict = self.compute_scores(model, test_paths, Errors.FRR,
                                              ubm,
                                              t_matrix, mfcc_number,
                                              projection_matrix)
        frr_scores = utils.from_dictionary_to_array(frr_scores_dict)
        self.save_scores(Errors.FRR, frr_scores)

        far_scores_dict = self.compute_scores(model, test_paths, Errors.FAR,
                                              ubm,
                                              t_matrix, mfcc_number,
                                              projection_matrix)
        far_scores = utils.from_dictionary_to_array(far_scores_dict)
        self.save_scores(Errors.FAR, far_scores)

        frr_scores = self.load_scores(Errors.FRR)
        far_scores = self.load_scores(Errors.FAR)

        min_score = min(min(frr_scores), min(far_scores))
        max_score = max(max(frr_scores), max(far_scores))

        threshold_to_test = np.array(
            [np.linspace(min_score, max_score, 10000)])

        frr_scores = np.array([frr_scores])
        far_scores = np.array([far_scores])

        frr = np.mean(
            np.greater(np.transpose(frr_scores), threshold_to_test).astype(
                int),
            axis=0)
        far = np.mean(
            np.less(np.transpose(far_scores), threshold_to_test).astype(int),
            axis=0)

        self.save_rates(Errors.FAR, far)
        self.save_rates(Errors.FRR, frr)
        self.save_thresholds(np.transpose(threshold_to_test))

    def plot_rates(self):
        far = self.load_rates(Errors.FAR)
        frr = self.load_rates(Errors.FRR)
        thresholds = self.load_thresholds()

        x, y = intersection(thresholds, frr, thresholds, far)
        plt.figure()
        plt.plot(thresholds, frr, 'b', label="FRR")
        plt.plot(thresholds, far, 'g', label="FAR")
        plt.plot(x, y, 'ro', label='EER')

        plt.ylabel('Error Rate')
        plt.xlabel('Threshold')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.plot(np.array(far), np.array(frr), 'b')
        plt.ylabel('False Rejection Rates')
        plt.xlabel('False Acceptance Rates')
        plt.title("Detection Errors Tradeoff (DET) curves")
        plt.show()

    def enroll_and_test(self):
        """
        files_handler = records_files_handler.FilesHandler()
        files_handler.set_ptdb_root_directory()
        train_paths, test_paths = files_handler.get_train_test_paths()
        """

        files_handler = records_files_handler.FilesHandler()
        files_handler.set_members()
        evaluate_paths = files_handler.evaluate_paths
        enroll_paths, test_paths = self.split_paths(
            evaluate_paths)

        ubm = universal_background_model.load_ubm(self.root_directory,
                                                  self.components_number)
        t_matrix = total_variability_space.load_t_matrix(self.root_directory,
                                                         self.components_number)
        projection_matrix = i_vector_extraction.load_projection_matrix(
            self.root_directory, self.components_number)

        enroll_object = enroll_and_verify.Enrollment(self.components_number,
                                                     self.mfcc_number,
                                                     self.space_dimension)
        enroll_object.set_root_directory(self.root_directory)

        for speaker in enroll_paths:
            i_vector_model = enroll_object.create_speaker_model(enroll_paths[
                                                                    speaker],
                                                                ubm, t_matrix,
                                                                projection_matrix)
            i_vector_extraction.save_i_vector_model(self.root_directory,
                                                    i_vector_model, speaker,
                                                    self.components_number)

        model = gplda_model.load_gplda_model(self.root_directory,
                                             self.components_number)
        self.compute_error_rates(model, test_paths, ubm, t_matrix,
                                 self.mfcc_number, projection_matrix)


def main():
    evaluation_object = EvaluationClass(128, 13, 200)
    evaluation_object.set_root_directory(
        "D:\licenta\LICENSE ""WORK\Speaker-Recognition\code")
    # "/home/onovac/licenta/Speaker-Recognition/code")

    evaluation_object.plot_rates()


main()
