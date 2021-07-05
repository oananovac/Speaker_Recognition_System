import os
import sys
from pathlib import Path

path = Path(__file__).parents[1]
sys.path.insert(0, path.absolute().as_posix() + "/code/training")


import database_handler
import enroll_and_verify


def get_records_paths():
    path = os.getcwd() + r"\uploads"
    files = os.listdir(path)
    paths = []
    for file in files:
        paths.append(path + "\\" + file)
    return paths


def delete_files(paths):
    for path in paths:
        os.remove(path)


def check_username_exists(username):
    database = database_handler.Database()
    database.database_connection()
    if not database.check_speaker_exists(username):
        return False
    else:
        return True


def enroll_speaker(username, components_number, mfcc_number, space_dimension):
    if not check_username_exists(username):
        paths = get_records_paths()

        enroll_obj = enroll_and_verify.Enrollment(components_number,
                                                  mfcc_number,
                                                  space_dimension)
        enroll_obj.set_root_directory(path.absolute().as_posix() + "/code")
        enroll_obj.enroll_speaker( paths, username)
        delete_files(paths)
    else:
        return False


def verify_speaker(username, components_number, mfcc_number, space_dimension):
    if not check_username_exists(username):
        return False
    else:
        paths = get_records_paths()

        verify_obj = enroll_and_verify.Verification(components_number,
                                                  mfcc_number,
                                                  space_dimension)
        verify_obj.set_root_directory(path.absolute().as_posix() + "/code")
        decision = verify_obj.verify_speaker(paths[0], username)
        delete_files(paths)
        return decision


#load_model_parameters(128,path.absolute().as_posix() + "/code")

#check_username_exists("oana")