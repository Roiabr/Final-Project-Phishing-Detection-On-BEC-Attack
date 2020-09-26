import sys
import joblib
import numpy
from sklearn.model_selection import train_test_split
from ML import Random_Forest, DecisionTreeClassifier, Knn, Svm
import Extract_features
import pandas as pd
import WordsToVector
from Email import Email


def SaveTheModel(TypeModel):
    if TypeModel is 'header':
        joblib.dump(RF, 'Saved_Model/Header/Random_Forest_Model_Header')
        joblib.dump(DT, 'Saved_Model/Header/Decision_Tree_Model_Header')
        joblib.dump(Knn, 'Saved_Model/Header/Knn_Model_Header')
        joblib.dump(Svm, 'Saved_Model/Header/Svm_Model_Header')
    else:
        joblib.dump(RFBody, 'Saved_Model/Body/Random_Forest_Model_Body')
        joblib.dump(DTBody, 'Saved_Model/Body/Decision_Tree_Model_Body')


if __name__ == '__main__':
    numpy.set_printoptions(threshold=sys.maxsize)
    # The first  part is to create the dataset
    data = Extract_features.Create_the_dataSet()

    y = data['label']

    # We get the important word from the headers
    WordsToVector.getTheWordHeader(data)

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
    SaveTheModel('header')

    # Move to the Body Classification
    suspect_fraud = []
    for x in X_test:
        CountT = vector.transform([x.header])
        suspect = Model.predict(CountT)
        if suspect == ['1']:
            suspect_fraud.append(x)

    clean_suspect_fraud = []

    for x in suspect_fraud:
        if not pd.isnull(x.body):
            clean_suspect_fraud.append(x)

    # Split the dataset for 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(clean_suspect_fraud, [x.label for x in clean_suspect_fraud],
                                                        train_size=0.8)
    X_train_body = (x.body for x in X_train if not pd.isnull(x.body))
    X_test_body = (x.body for x in X_test if not pd.isnull(x.body))

    CountTrainBody, CountTestBody, vector = WordsToVector.TfIdfBody(X_train_body, X_test_body)

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

    SaveTheModel('body')

    print("The best Model is: ", ModelBody, "and the best Score is: ", maxModelBody)

