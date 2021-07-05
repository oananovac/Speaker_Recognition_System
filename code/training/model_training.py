import features_extraction
import records_files_handler
import universal_background_model
import statistics
import total_variability_space
import i_vector_extraction
import utils
import gplda_model


class TrainClass(object):
    def __init__(self, mfcc_number,
                 gaussians_components_number,
                 total_variability_space_dimension,
                 t_matrix_iterations_number, frame_duration, step_duration):
        self.root_directory = None
        self.mfcc_number = mfcc_number
        self.gaussians_components_number = gaussians_components_number
        self.total_variability_space_dimension = total_variability_space_dimension
        self.t_matrix_iterations_number = t_matrix_iterations_number
        self.frame_duration = frame_duration
        self.step_duration = step_duration

    def set_root_directory(self, root):
        self.root_directory = root

    def get_utterance_per_speaker(self, dict_paths):
        keys_list = dict_paths.keys()
        utterances = {}
        for i in keys_list:
            utterances[i] = len(dict_paths[i])
        return utterances

    def train_model(self):
        """
        files_handler = records_files_handler.FilesHandler()
        files_handler.set_ptdb_root_directory()
        train_paths, test_paths = files_handler.get_train_test_paths()
        """

        files_handler = records_files_handler.FilesHandler()
        files_handler.set_members()
        train_paths = files_handler.train_paths
        dev_paths = files_handler.development_paths

        for speaker in dev_paths:
            for file in dev_paths[speaker]:
                train_paths[speaker].append(file)

        features_handler = features_extraction.FeaturesExtraction(
            self.mfcc_number, True, self.frame_duration, self.step_duration)
        features = features_handler.extract_all_features(train_paths)
        features_handler.save_all_features(self.root_directory, features)

        features = features_extraction.load_all_features(
            self.root_directory)
        print(len(features))
        print(len(features[0]))

        ubm = universal_background_model.create_ubm(
            features, self.gaussians_components_number)
       	ubm.save_ubm(self.root_directory)

        ubm = universal_background_model.load_ubm(self.root_directory,
                                                  self.gaussians_components_number)

        n, f = statistics.Baum_Welch_Statistic(train_paths,
                                               self.mfcc_number, ubm,
                                               self.frame_duration,
                                               self.step_duration)
        statistics.save_statistics(n, f, self.root_directory,
                                   self.gaussians_components_number)
        n, f = statistics.load_statistics(self.root_directory,
                                          self.gaussians_components_number)

        t_space = total_variability_space.TotalVariabilitySpace(
            self.total_variability_space_dimension,
            self.t_matrix_iterations_number)
        t_matrix = t_space.create_t_matrix(n, f, ubm, 3 * self.mfcc_number,
                                           self.gaussians_components_number)
        t_space.save_t_matrix(t_matrix, self.root_directory,
                              self.gaussians_components_number)
        t_matrix = total_variability_space.load_t_matrix(self.root_directory, self.gaussians_components_number)

        i_vector_extraction.extract_i_vectors(
            self.root_directory, ubm, train_paths, t_matrix,
            self.total_variability_space_dimension, self.mfcc_number,
            self.frame_duration, self.step_duration,
            self.gaussians_components_number)

        # train projection matrix
        ivectors = utils.load_ivectors(train_paths, self.root_directory,
                                       self.gaussians_components_number)
        utterances_per_speaker = self.get_utterance_per_speaker(train_paths)

        lda_matrix, lda_ivectors = \
            i_vector_extraction.LDA_projection_matrix(ivectors)
        projection_matrix = i_vector_extraction.WCCN_projection_matrix(
            lda_matrix, lda_ivectors, utterances_per_speaker)

        i_vector_extraction.save_projection_matrix(
            self.root_directory,
            self.gaussians_components_number,
            projection_matrix)
        projection_matrix = i_vector_extraction.load_projection_matrix(
            self.root_directory, self.gaussians_components_number)

        gplda_model.train_model(self.root_directory,
                                self.gaussians_components_number,
                                projection_matrix, ivectors,
                                utterances_per_speaker, 5)


def main():
    train_object = TrainClass(13, 128, 200, 20, 0.025, 0.01)
    train_object.set_root_directory(
        r"/home/onovac/licenta/Speaker-Recognition/code")#"D:\licenta\LICENSE "
                                    #"WORK\Speaker-Recognition/code")

    train_object.train_model()


main()
