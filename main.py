from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import Extract_features
import pandas as pd

import WordsToVector



if __name__ == '__main__':
    FILENAME = "emails_dataset.csv"
    # fields = ['email_header', 'Body', 'label']
    # Extract_features.save_output(FILENAME, fields)
    # #read all emails,parser the email to a csv file
    # Extract_features.get_email_ham(FILENAME)
    # # Extract_features.get_email_spam(FILENAME)

    # make a dataframe of the header features and the label for the machine learning on header
    df = pd.read_csv(FILENAME)

    X = df['email_header']
    y = df['label']

    X_train = df.loc[:, 'email_header'].values
    y_train = df.loc[:y, 'label'].values
    X_test = df.loc[X.shape:, 'email_header'].values
    y_test = df.loc[y.shape:, 'label'].values

    # Convert the header text to vector of number for machine learning

    train_vectors, test_vectors = WordsToVector.CountVector(X_train, X_test)

    train_vectors2, test_vectors2 = WordsToVector.TfIdf(X_train, X_test)






