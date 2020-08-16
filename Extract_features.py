import csv
import os
import mailparser


def save_output(filename, csv_content):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_content)


def get_email_ham(FILENAME):
    for subdir, dirs, files in os.walk(r'Enron-ham'):
        for filename in files:
            filepath = subdir + os.sep + filename
            csv_content = parser_email(filepath)
            print(csv_content)
            save_output(FILENAME, csv_content)


def get_email_spam(FILENAME):
    for subdir, dirs, files in os.walk(r'Enron-spam'):
        for filename in files:
            filepath = subdir + os.sep + filename
            csv_content = parser_email(filepath)
            print(csv_content)
            save_output(FILENAME, csv_content)


def parser_email(filepath):
    mail = mailparser.parse_from_file(filepath)

    email_header = mail.headers
    email_from = email_header.get('From')
    email_to = email_header.get('To')
    email_subject = email_header.get('Subject')
    email_body = mail.body

    email_Cc = email_header.get('Cc')
    if email_Cc is None:
        email_Cc = ''
    csv_content = [email_from, email_to, email_Cc, email_subject, email_body, 0]
    return csv_content
