import numpy as np
import i_vector_extraction
import universal_background_model
import total_variability_space
import database_handler
import gplda_model

THRESHOLD = 3.2256


class ModelClass(object):
    def __init__(self, components_number, mfcc_number, space_dimension):
        self.root_directory = None
        self.components_number = components_number
        self.mfcc_number = mfcc_number
        self.space_dimension = space_dimension

    def set_root_directory(self, root):
        self.root_directory = root

    def create_speaker_model(self, records, ubm, t_matrix, projection_matrix):
        i_vectors_matrix = np.zeros((len(records), len(projection_matrix)))
        index = 0
        sigma = i_vector_extraction.get_sigma(ubm, self.space_dimension)
        for record in records:
            w = i_vector_extraction.extract_i_vector_from_signal(ubm, record,
                                                     t_matrix,
                                                     len(t_matrix[0]),
                                                     self.mfcc_number,
                                                                 0.025,
                                                                 0.01, sigma)
            w = np.matmul(projection_matrix, np.transpose(w))
            i_vectors_matrix[index] = w
            index += 1

        i_vector_model = np.mean(i_vectors_matrix, 0)
        return i_vector_model

    def load_model_parameters(self, components_number):
        ubm = universal_background_model.load_ubm(self.root_directory,
                                                  components_number)
        t_matrix = total_variability_space.load_t_matrix(self.root_directory,
                                                         components_number)
        projection_matrix = i_vector_extraction.load_projection_matrix(
            self.root_directory, components_number)
        return ubm, t_matrix, projection_matrix


class Enrollment(ModelClass):
    def __init__(self,components_number, mfcc_number, space_dimension):
        ModelClass.__init__(self, components_number, mfcc_number,
                             space_dimension)

    def enroll_speaker(self, records_paths, username):
        database = database_handler.Database()
        database.database_connection()
        ubm, t_matrix, projection_matrix = self.load_model_parameters(
            self.components_number)
        speaker_model = self.create_speaker_model(records_paths, ubm,
                                           t_matrix, projection_matrix)
        bytes_model = speaker_model.tostring()
        database.insert_speaker_model(username, bytes_model)


class Verification(ModelClass):
    def __init__(self,components_number, mfcc_number, space_dimension):
        ModelClass.__init__(self, components_number, mfcc_number,
                             space_dimension)

    def compute_score(self, new_model, speaker_model):
        model = gplda_model.load_gplda_model(self.root_directory,
                                             self.components_number)
        score = gplda_model.compute_gplda_score(model, new_model,
                                                speaker_model)
        return score

    def verify_speaker(self, record_path, username):
        database = database_handler.Database()
        database.database_connection()
        speaker_model_bytes = database.extract_speaker_model(username)
        speaker_model = np.frombuffer(speaker_model_bytes, dtype=np.float64)
        ubm, t_matrix, projection_matrix = self.load_model_parameters(
            self.components_number)

        paths = []
        paths.append(record_path)

        new_model = self.create_speaker_model(paths, ubm,
                                              t_matrix, projection_matrix)
        score = self.compute_score(new_model, speaker_model)

        if score <= THRESHOLD:
            return True
        else:
            return False
