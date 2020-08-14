import csv
import os

import mailparser
import pandas

FILENAME = "emailCsv.csv"


def get_email():
    for subdir, dirs, files in os.walk(r'hopwood__tom'):
        for filename in files:
            filepath = subdir + os.sep + filename

            mail = mailparser.parse_from_file(filepath)

            email_from = mail.from_
            email_to = mail.to
            email_subject = mail.subject
            email_body = mail.body
            email_header = mail.headers

            csv_content = [email_from, email_to, email_subject, email_body, "spam"]
            print(csv_content)
            save_output(FILENAME, csv_content)


def save_output(filename, csv_content):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(csv_content)


get_email()
