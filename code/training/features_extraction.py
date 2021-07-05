import python_speech_features
import math
import numpy as np
from sklearn import preprocessing
import voice_activity_detector


class FeaturesExtraction(object):
    def __init__(self, mfcc_number, append_energy, frame_duration,
                 step_duration):
        self.mfcc_number = mfcc_number
        self.append_energy = append_energy
        self.frame_duration = frame_duration
        self.step_duration = step_duration

    def normalize(self, y, axis=0, variance=True):
        # y.shape = (steps, features)
        sequence = y.astype("float32")

        # zero mean
        mean = sequence.mean(axis=axis)
        sequence -= mean

        if variance:
            # unit Variance
            standard_deviation = sequence.std(axis=axis)
            standard_deviation = np.maximum(standard_deviation,
                                            np.finfo(np.float32).eps)
            sequence /= standard_deviation

        return sequence

    def extract_mfcc_from_signal(self, signal, sample_rate):
        nfft_value = int(
            math.pow(2, math.ceil(math.log(sample_rate * self.frame_duration,
                                           2))))

        mfcc = python_speech_features.mfcc(signal, samplerate=sample_rate,
                                           numcep=self.mfcc_number,
                                           winfunc=np.hanning,
                                           winlen=self.frame_duration,
                                           winstep=self.step_duration,
                                           nfft=nfft_value, ceplifter=0,
                                           appendEnergy=self.append_energy)

        norm_mfcc = preprocessing.scale(mfcc)
        # folosit pentru un set de date (de dimensiune mare) in care acestea
        # sunt
        # foarte raspandite, le centreaza pe 0 si varianta 1

        delta_mfcc = python_speech_features.delta(norm_mfcc, 2)
        delta_delta_mfcc = python_speech_features.delta(delta_mfcc, 2)
        features = np.hstack((norm_mfcc, delta_mfcc, delta_delta_mfcc))
        #features = self.normalize(features)

        return features

    def extract_all_features(self, signals_paths):
        all_features = np.asarray(())
        speakers_list = signals_paths.keys()
        for speaker in speakers_list:

            audio_numbers = len(signals_paths[speaker])
            for index in range(audio_numbers):

                vad_obj = voice_activity_detector.Vad(signals_paths[speaker][index], 2)
                signal, rate = vad_obj.get_speech_signal()
                del vad_obj

                features = self.extract_mfcc_from_signal(signal, rate)

                if all_features.size == 0:
                    all_features = features
                else:
                    all_features = np.vstack((all_features, features))

        return all_features

    def save_all_features(self, file_path, features):
        file = open(file_path + "/models/train_features_new.txt", "wb")
        np.save(file, features)
        file.close


def load_all_features(file_path):
    file = open(file_path + "/models/train_features_new.txt",
                "rb")
    features = np.load(file)
    file.close

    return features


