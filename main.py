from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from wordcloud import WordCloud
from ML import Random_Forest, Svm, Adaboost, DecisionTreeClassifier, Knn
import Extract_features
import pandas as pd
import WordsToVector
from Email import Email

if __name__ == '__main__':
    FILENAME = "emails_dataset.csv"
    # fields = ['email_header', 'Body', 'label']
    # Extract_features.save_output(FILENAME, fields)
    # # read all emails,parser the email to a csv file
    # Extract_features.get_email_ham(FILENAME)
    # Extract_features.get_email_spam(FILENAME)

    # # make a dataframe of the header features and the label for the machine learning on header
    data = pd.read_csv(FILENAME, encoding="ISO-8859-1")

    y = data['label']

    # We get the important word from the headers
    # WordsToVector.getTheWordHeader(data)

    # We create a list with email object
    # each email contains the header, body and the label
    list_emails = Email.List_of_emails(data)

    X_train, X_test, y_train, y_test = train_test_split(list_emails, y, train_size=0.5)

    max_acc = 0

    email_headers_train = [x.header for x in X_train if not pd.isnull(x.header)]
    email_headers_test = [x.header for x in X_test if not pd.isnull(x.header)]
    CountTrain, CountTest, vector = WordsToVector.CountVector(email_headers_train, email_headers_test)

    # print(Svm.svmlinear(CountTrain, y_train, CountTest, y_test))
    # print(Svm.svmpoly(CountTrain, y_train, CountTest, y_test))
    # print(Svm.svmrbf(CountTrain, y_train, CountTest, y_test))
    # print(Svm.svmsigmoid(CountTrain, y_train, CountTest, y_test))
    # print(Random_Forest.random_forest(CountTrain, y_train, CountTest, y_test))
    # print(DecisionTreeClassifier.decisionTreeClassifier(CountTrain, y_train, CountTest, y_test))
    # print(Adaboost.adaBoost(CountTrain, y_train, CountTest, y_test))
    dc = DecisionTreeClassifier.decisionTreeClassifier()
    ada = Adaboost.AdaBoostClassifier()

    model = Random_Forest.RandomForestClassifier()
    model.fit(CountTrain, y_train)
    suspect_fraud = []
    for x in X_test:
        CountT = vector.transform([x.header])
        suspect = model.predict(CountT)
        if suspect == ['1']:
            suspect_fraud.append(x)


    # Move to the Body Classification
    clean_suspect_fraud = []

    for x in suspect_fraud:
        if not pd.isnull(x.body):
            clean_suspect_fraud.append(x)

    X_train, X_test, y_train, y_test = train_test_split(clean_suspect_fraud, [x.label for x in clean_suspect_fraud],
                                                        train_size=0.5)
    X_train_body = (x.body for x in X_train if not pd.isnull(x.body))
    X_test_body = (x.body for x in X_test if not pd.isnull(x.body))
    CountTrainBody, CountTestBody, vector = WordsToVector.CountVector(X_train_body, X_test_body)

    model2 = Random_Forest.RandomForestClassifier()
    model2.fit(CountTrainBody, y_train)

    print(model2.score(CountTestBody, y_test))
