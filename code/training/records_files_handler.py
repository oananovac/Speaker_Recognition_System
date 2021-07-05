import os
from pathlib import Path


class FilesHandler(object):
    def __init__(self):
        self.root_directory = None
        self.train_paths = None
        self.development_paths = None
        self.evaluate_paths = None

    def set_root_directory(self):
        path = Path(__file__).parents[3]
        self.root_directory = path.absolute().as_posix() + "/rodigits"
        print(self.root_directory)

    def set_ptdb_root_directory(self):
        path = Path(__file__).parents[3]
        self.root_directory = path.absolute().as_posix() + "/SPEECH DATA"
        print(self.root_directory)

    def get_paths(self, filename):
        file = open(self.root_directory + "/" + filename, "r")
        lines = file.readlines()

        paths_list = {}
        for line in lines:
            splited_line = line.split("/")
            if splited_line[0] not in paths_list.keys():
                paths_list[splited_line[0]] = []
            last = line[:-1]
            paths_list[splited_line[0]].append(self.root_directory + "/" + last
                                               + ".wav")
        file.close
        return paths_list

    def set_members(self):
        self.set_root_directory()
        self.train_paths = self.get_paths("trainSet.txt")
        self.development_paths = self.get_paths("devSet.txt")
        self.evaluate_paths = self.get_paths("evalSet.txt")

    def get_train_test_paths(self):  # setul de date PTDB-TUG
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