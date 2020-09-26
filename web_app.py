from flask import Flask, render_template, request, get_template_attribute
from werkzeug.utils import secure_filename
import os
from Extract_features import getWebEmails
from WordsToVector import CountVectorWeb
import joblib

UPLOAD_FOLDER = 'Dataset/Web_Emails'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__, static_folder="docs", template_folder='docs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('BEC_Web.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join('Dataset/Web_Emails', secure_filename(f.filename)))
        email = getWebEmails(os.path.join('Dataset/Web_Emails', secure_filename(f.filename)))
        list_header = [email[1]]
        model = joblib.load('Saved_Model/Body/Random_Forest_Model_Body')
        email_vectors = CountVectorWeb(list_header, model.n_features_)
        predict = model.predict(email_vectors)
        print(predict)
        if predict == ['0']:
            return render_template('BEC_Web.html', pred='This is Ham Email, Everything is good')
        else:
            return render_template('BEC_Web.html', pred='This is BEC Attack!!')


if __name__ == '__main__':
    app.run(debug=True)
