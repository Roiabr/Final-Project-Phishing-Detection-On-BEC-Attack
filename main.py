import sys

import joblib
import numpy

from sklearn.model_selection import train_test_split
from ML import Random_Forest, DecisionTreeClassifier, Knn, Svm
import Extract_features
import pandas as pd
import WordsToVector
from Email import Email

if __name__ == '__main__':
    numpy.set_printoptions(threshold=sys.maxsize)

    FILENAME = "emails_dataset.csv"
    fields = ['email_header', 'Body', 'label']
    Extract_features.save_output(FILENAME, fields)
    # read all emails,parser the email to a csv file
    Extract_features.get_email_ham(FILENAME)
    Extract_features.get_email_spam(FILENAME)

    # # make a dataframe of the header features and the label for the machine learning on header
    data = pd.read_csv(FILENAME, encoding="ISO-8859-1")
    data.dropna(axis=0, how='any')
    y = data['label']

    # We get the important word from the headers
    # WordsToVector.getTheWordHeader(data)

    # We create a list with email object
    # each email contains the header, body and the label
    list_emails = Email.List_of_emails(data)

    # Split the dataset for 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(list_emails, y, train_size=0.8)

    # make a list of all the header of the emails
    email_headers_train = [x.header for x in X_train if not pd.isnull(x.header)]
    email_headers_test = [x.header for x in X_test if not pd.isnull(x.header)]

    # Two way to work on text: 1) CountVector 2)TfIdf
    CountTrain, CountTest, vector = WordsToVector.CountVector(email_headers_train, email_headers_test)
    CountTrainTf, CountTestTf, vectorTf = WordsToVector.TfIdf(email_headers_train, email_headers_test)

    y_train = y_train.fillna("0")
    y_test = y_test.fillna("0")

    maxModel = 0
    Model = ""

    RFScore, RF = Random_Forest.random_forest(CountTrainTf, CountTestTf, y_train, y_test, 0)
    if RFScore > maxModel:
        maxModel = RFScore
        Model = RF

    DTScore, DT = DecisionTreeClassifier.decisionTreeClassifier(CountTrainTf, CountTestTf, y_train, y_test, 0)
    if DTScore > maxModel:
        maxModel = DTScore
        Model = DT

    KnnScore, Knn = Knn.knn(CountTrainTf, CountTestTf, y_train, y_test, 0)
    if KnnScore > maxModel:
        maxModel = KnnScore
        Model = Knn

    SvmScore, Svm = Svm.svm(CountTrainTf, CountTestTf, y_train, y_test, 0)
    if SvmScore > maxModel:
        maxModel = SvmScore
        Model = Svm

    print("The best Model is: ", Model, "and the best Score is: ", maxModel)

    # Save the model for future predictions
    joblib.dump(RF, 'Saved_Model/Random_Forest_Model_Header')
    joblib.dump(DT, 'Saved_Model/Decision_Tree_Model_Header')
    joblib.dump(Knn, 'Saved_Model/Knn_Model_Header')
    joblib.dump(Svm, 'Saved_Model/Svm_Model_Header')

    suspect_fraud = []
    for x in X_test:
        CountT = vector.transform([x.header])
        suspect = int(Model.predict(CountT))
        if suspect == 1:
            suspect_fraud.append(x)

    # Move to the Body Classification
    clean_suspect_fraud = []

    for x in suspect_fraud:
        if not pd.isnull(x.body):
            clean_suspect_fraud.append(x)

    X_train, X_test, y_train, y_test = train_test_split(clean_suspect_fraud, [x.label for x in clean_suspect_fraud],
                                                        train_size=0.8)
    X_train_body = (x.body for x in X_train if not pd.isnull(x.body))
    X_test_body = (x.body for x in X_test if not pd.isnull(x.body))

    CountTrainBody, CountTestBody, vector = WordsToVector.CountVector(X_train_body, X_test_body)

    maxModelBody = 0
    ModelBody = ""
    RFScoreBody, RFBody = Random_Forest.random_forest(CountTrainBody, CountTestBody, y_train, y_test, 1)
    if RFScoreBody > maxModelBody:
        maxModelBody = RFScoreBody
        ModelBody = RFBody

    DTScoreBody, DTBody = DecisionTreeClassifier.decisionTreeClassifier(CountTrainBody, CountTestBody, y_train, y_test,
                                                                        1)
    if DTScoreBody > maxModelBody:
        maxModelBody = DTScoreBody
        ModelBody = DTBody

    # KnnScoreBody, KnnBody = Knn.knn(CountTrainBody, CountTestBody, y_train, y_test)
    # if KnnScoreBody > maxModelBody:
    #     maxModelBody = KnnScoreBody
    #     ModelBody = KnnBody

    # SvmScoreBody, SvmBody = Svm.svm(CountTrainBody, CountTestBody, y_train, y_test)
    # if SvmScoreBody > maxModelBody:
    #     maxModelBody = SvmScoreBody
    #     ModelBody = SvmBody

    print("The best Model is: ", ModelBody, "and the best Score is: ", maxModelBody)
