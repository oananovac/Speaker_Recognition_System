from flask import Flask, render_template, request
import os
from demo import model_handler

ALLOWED_EXTENSIONS = {'wav'}
COMPONENTS_NUMBER = 1024
MFCC_NUMBER = 13
T_DIMENSION = 200

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


def save_file(name):
    if name not in request.files:
        return False
    else:
        file = request.files[name]
        if allowed_file(name):
            file.save(os.path.join("uploads", file.filename))
            return True
        else:
            return False


@app.route("/enrollment", methods=['GET', 'POST'])
def enroll():
    if request.method == "POST":
        username = request.form['username']
        if username == "":
            return render_template("enrollpage.html", error="Please insert "
                                                            "username!")
        print(username)
        if model_handler.check_username_exists(username):
            return render_template("enrollpage.html", error="Please insert "
                                                            "another "
                                                            "username! This "
                                                            "is already "
                                                            "taken.")
        else:
            for i in range(3):
                name = "file"
                name = name + str(i + 1)
                file = request.files[name]
                if file.filename == '':
                    paths = model_handler.get_records_paths()
                    model_handler.delete_files(paths)
                    return render_template("enrollpage.html",
                                           error="You didn't upload all files!")
                if allowed_file(file.filename):
                    file.save(os.path.join("uploads", file.filename))
                else:
                    paths = model_handler.get_records_paths()
                    model_handler.delete_files(paths)
                    return render_template("enrollpage.html",
                                           error="Please upload wav files!")
            model_handler.enroll_speaker(username, COMPONENTS_NUMBER,
                                         MFCC_NUMBER, T_DIMENSION)
            return render_template("enrollpage.html",
                                   message="Congratulations! You have successfully enrolled.")

    return render_template("enrollpage.html")


@app.route("/verification", methods=['GET', 'POST'])
def verify():
    if request.method == "POST":
        username = request.form['username']
        if username == "":
            return render_template("verificationpage.html", error="Please "
                                                                  "insert "
                                                                  "username!")
        if not model_handler.check_username_exists(username):
            return render_template("verificationpage.html", error="This "
                                                                  "username "
                                                                  "doesn't exist "
                                                                  "in the system!")
        else:
            name = "file1"
            file = request.files[name]
            if file.filename == '':
                return render_template("verificationpage.html",
                                       error="You didn't upload the wav file!")
            if allowed_file(file.filename):
                file.save(os.path.join("uploads", file.filename))
                # veeeerificaaare
                decision = model_handler.verify_speaker(username,
                                                        COMPONENTS_NUMBER,
                                                        MFCC_NUMBER,
                                                        T_DIMENSION)
                if decision == True:
                    return render_template("verificationpage.html",
                                           message="Congratulations! Successful verification.")
                else:
                    return render_template("verificationpage.html",
                                           error="We're sorry. Verification "
                                                 "failed!")
            else:
                paths = model_handler.get_records_paths()
                model_handler.delete_files(paths)
                return render_template("verificationpage.html",
                                       error="Please upload wav files!")

    return render_template("verificationpage.html")


@app.route("/home")
@app.route("/")
def home():
    return render_template("homepage.html")


if __name__ == "__main__":
    app.run(debug=True)
