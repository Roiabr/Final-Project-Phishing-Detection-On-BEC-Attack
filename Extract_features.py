import csv
import io
import os
import mailparser


def save_output(filename, csv_content):
    with io.open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if csv_content[0] == '':
            return
        writer.writerow(csv_content)


def get_email_ham(FILENAME):
    ham = 0
    for subdir, dirs, files in os.walk('Dataset/Enron - dataset/Enron-ham/beck-s'):
        for filename in files:
            filepath = subdir + os.sep + filename
            csv_content = parser_email(filepath, ham)
            print(csv_content)
            save_output(FILENAME, csv_content)


def get_email_spam(FILENAME):
    spam = 1
    for subdir, dirs, files in os.walk('Spam'):
        for filename in files:
            filepath = subdir + os.sep + filename
            csv_content = parser_email(filepath, spam)
            print(csv_content)
            save_output(FILENAME, csv_content)


def parser_email(filepath,flag):

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
