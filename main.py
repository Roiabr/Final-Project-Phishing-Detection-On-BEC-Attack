import Extract_features
import pandas as pd

if __name__ == '__main__':
    FILENAME = "emails_dataset.csv"
    fields = ['From', 'To', 'Cc', 'Subject', 'Body', 'label']
    Extract_features.save_output(FILENAME, fields)
    # read all emails,parser the email to a csv file
    Extract_features.get_email_ham(FILENAME)
    Extract_features.get_email_spam(FILENAME)

    # make a dataframe of the header features and the label for the machine learning on header
    df = pd.read_csv(FILENAME)

    X = df[['From', 'To', 'Cc', 'Subject']]
    y = df['label']
