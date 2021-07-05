import i_vector_extraction
import universal_background_model
import total_variability_space
import gplda_model
import enroll_and_verify


def load_model_parameters(components_number, root_directory):
    ubm = universal_background_model.load_ubm(root_directory,
                                              components_number)
    t_matrix = total_variability_space.load_t_matrix(root_directory,
                                                     components_number)
    projection_matrix = i_vector_extraction.load_projection_matrix(
        root_directory, components_number)
    return ubm, t_matrix, projection_matrix


def compute_score(new_model, speaker_model, components_number, root_directory):
    model = gplda_model.load_gplda_model(root_directory, components_number)
    score = gplda_model.compute_gplda_score(model, new_model, speaker_model)
    return score


enrollobj = enroll_and_verify.ModelClass(1024, 13, 200)
enrollobj.set_root_directory("D:\licenta\LICENSE ""WORK\Speaker-Recognition\code")
ubm, t_matrix, projection_matrix = load_model_parameters(1024,
                                                         "D:\licenta\LICENSE ""WORK\Speaker-Recognition\code")


records = []
records.append(r"D:\records\razvan\Recording_1.wav")
records.append(r"D:\records\razvan\Recording_2.wav")
records.append(r"D:\records\razvan\Recording_3.wav")

enroll_model = enrollobj.create_speaker_model(records, ubm, t_matrix, projection_matrix)
score = []
record_to_test = []
for i in range(7):
    if i > 3:
        record_to_test.append(r"D:\records\razvan\Recording_" + str(i)+
                              ".wav")
        new_model = enrollobj.create_speaker_model(record_to_test, ubm,
                                              t_matrix, projection_matrix)
        score.append(compute_score(new_model, enroll_model, 1024,
                                   "D:\licenta\LICENSE WORK\Speaker-Recognition\code"))


print("Ready")