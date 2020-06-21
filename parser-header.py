import csv
import numpy as np
import re
import pandas as pd
import csv

file_Name = "1.txt"

with open(file_Name, 'r') as f:
    FRom = re.findall('From', f.read())
    emails = re.findall(r"^\d+", f.read(), re.DOTALL)
    for email in emails:
         print(email)

    number=0;
    matches = []

rx = re.compile(r'^(?P<email>[^|\n]+)', re.MULTILINE)
with open("1.txt") as f:
    raw_data = f.read()
    emails = [match.group('email') for match in rx.finditer(raw_data)]


def copy_contents(from_path, to_path):
    from_file = open(from_path, 'r')
    to_file = open(to_path, 'w')

    for no, line, in enumerate(from_file.readlines()):
        if "To:" in line:
            insert(to_file,line)
            # print(line)
        if "From:" in line:
            insert(to_file,line)



def insert(file,line):
    file.write(line)

copy_contents(file_Name, "t.txt")


regEX = r"[\w\.]+@[\w\.]+"
pattern = re.compile(regEX)
findall = pattern.findall(open(file_Name, 'r').read())
for email in findall:
    print(email)



f