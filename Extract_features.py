import csv
import io
import os
import mailparser
import pandas as pd


def Create_the_dataSet():
    FILENAME = "Dataset/emails_dataset.csv"
    fields = ['email_header', 'Body', 'label']
    save_output(FILENAME, fields)
    # read all emails,parser the email to a csv file
    # get_email_ham(FILENAME)
    # get_email_spam(FILENAME)

    # # make a dataframe of the header features and the label for the machine learning on header
    data = pd.read_csv(FILENAME, encoding="ISO-8859-1")
    data.dropna(axis=0, how='any')
    return data


def save_output(filename, csv_content):
    with io.open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if csv_content[0] == '':
            return
        writer.writerow(csv_content)


def get_email_ham(FILENAME):
    ham = 0
    for subdir, dirs, files in os.walk('Dataset/Ham/beck-s'):
        for filename in files:
            filepath = subdir + os.sep + filename
            csv_content = parser_email(filepath, ham)
            save_output(FILENAME, csv_content)


def get_email_spam(FILENAME):
    spam = 1
    for subdir, dirs, files in os.walk('Dataset/Spam'):
        for filename in files:
            filepath = subdir + os.sep + filename
            csv_content = parser_email(filepath, spam)

            save_output(FILENAME, csv_content)


def getWebEmails(filepath):
    csv_content = parser_email(filepath)
    return csv_content


def parser_email(filepath, flag=1):
    mail = mailparser.parse_from_file(filepath)

    email_header = mail.headers

    email_from = email_header.get('From')

    if email_from is None:
        email_from = ''
    email_to = email_header.get('To')

    if email_to is None:
        email_to = ''

    email_body = mail.body

    email_Cc = email_header.get('Reply-To')

    if email_Cc is None:
        email_Cc = ''
    if flag == 1:
        csv_content = [email_from + " " + email_to + " " + email_Cc, email_body, 1]
    else:
        csv_content = [email_from + " " + email_to + " " + email_Cc, email_body, 0]
    return csv_content
