from sklearn.model_selection import train_test_split

import Extract_features
import pandas as pd

from WordsToVector import email_pipeline

if __name__ == '__main__':
    FILENAME = "emails_dataset.csv"
    #fields = ['From', 'To', 'Cc', 'Subject', 'Body', 'label']
    #Extract_features.save_output(FILENAME, fields)
    # read all emails,parser the email to a csv file
    #Extract_features.get_email_ham(FILENAME)
    # Extract_features.get_email_spam(FILENAME)

    # make a dataframe of the header features and the label for the machine learning on header
    df = pd.read_csv(FILENAME)

    X = df[['From', 'To', 'Cc', 'Subject']]
    y = df['label']

    # Convert the header text to vector of number for machine learning

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_augmented_train = email_pipeline.fit_transform(X_train)