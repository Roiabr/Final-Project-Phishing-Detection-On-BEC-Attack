from flask import Flask, render_template, request, get_template_attribute
from werkzeug.utils import secure_filename
import os
from Extract_features import getWebEmails
from WordsToVector import CountVectorWeb
import joblib

UPLOAD_FOLDER = 'Dataset/Web_Emails'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__, static_folder="Web", template_folder='Web')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('BEC_Web.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join('../Dataset/Web_Emails', secure_filename(f.filename)))
        email = getWebEmails()
        list_body = [email[1]]
        train_vectors = CountVectorWeb(list_body)
        model = joblib.load('../Saved_Model/Body/Random_Forest_Model_Body')
        predict = model.predict(train_vectors)
        print(predict)
        if predict == ['0']:
            return render_template('Ham.html')
        else:
            return render_template('Spam.html')


if __name__ == '__main__':
    app.run(debug=True)
