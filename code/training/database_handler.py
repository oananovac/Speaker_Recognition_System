import mysql.connector


class Database(object):
    def __init__(self):
        self.db = None
        self.db_cursor = None

    def database_connection(self):
        self.db = mysql.connector.connect(host="localhost", user="root",
                                          passwd="root",
                                          database="SpeakerRecognitionDataBase")
        self.db_cursor = self.db.cursor()

    def create_database_and_table(self):
        self.mycursor.execute("CREATE DATABASE ")

        self.mycursor.execute("CREATE TABLE SpeakersModels (username "
                              "VARCHAR(50) not null, model blob not null, "
                              "speakerID int PRIMARY KEY AUTO_INCREMENT)")

    def insert_speaker_model(self,username, model):
        self.db_cursor.execute("INSERT INTO SpeakersModels (username, model) "
                           "VALUES (%s,%s)", (username, model))
        self.db.commit()

    def check_speaker_exists(self, username):
        self.db_cursor.execute("SELECT * FROM SpeakersModels WHERE username "
                               "= %s", (username,))
        rows = self.db_cursor.fetchall()
        if not rows:
            return False
        else:
            return True

    def extract_speaker_model(self, username):
        self.db_cursor.execute("SELECT * FROM SpeakersModels WHERE "
                               "username = %s", (username,))
        row = self.db_cursor.fetchall()
        model = row[0][1]
        return model

