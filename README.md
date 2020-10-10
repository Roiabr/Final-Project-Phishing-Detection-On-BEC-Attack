# Phishing Detection On BEC Attack - Final Project

## Introduction:
In this project, we used machine learning algorithms to detect a BEC Attack as part of a final project at Ariel University.


## Motivation
Business email compromise (BEC) and employee impersonation have become one of the most costly cyber-security threats, causing over $12 billion in reported losses.
In this project, we suggest a solution to BEC attacks based on the work of Barracuda Networks, Columbia University on the BEC-Guard.
we decided to implement a machine learning that detects BEC attacks. Also, we decided to improve the BEC-Guard, we found that the BEC-Guard does not detect Impersonation emails
with an image as the body. 
Our key insight is to split the classification problem into two parts, one analyzing the header of the email, and the second applying natural language processing 
to detect phrases associated with BEC or suspicious links in the email body. 
We found out that there is a lack of codes for BEC attack prevention, so this work will be an open-source for the open-source community.

### Requirements

* Python 3.6+
* NumPy (pip install numpy)
* Pandas (pip install pandas)
* Scikit-learn (pip install scikit-learn)
* joblib (pip install joblib)
* glob (pip install glob)
* MatplotLib (pip install matplotlib)
* mailparser (pip install mailparser)
* nltk (pip install nltk)
* wordcloud (pip install wordcloud)

## Implementation:
For a detailed explanation of the work process, Data-set, and types of machines, you can read the article [here](https://docs.google.com/document/d/e/2PACX-1vQO96bRk1d3Y2A7rHh7AKjCzwUQxaUoeLJSfELVw61fwL9FLQ2j-mSQkhg2PMNe78PYNAkZ26mYl3xo/pub)

## Project Poster 
![Poster Ariel Projects 2020 ](https://user-images.githubusercontent.com/44756354/95667145-b1a8ca00-0b6a-11eb-8aa3-54924b9ad320.jpg)


## Contributor

* [Roi Abramovitch](https://www.linkedin.com/in/roi-abramovitch/)

* [Gal Hadida](https://www.linkedin.com/in/gal-hadida-648269190/)

* [Shira Baron](https://www.linkedin.com/in/shira-baron-013bba186/)

* [Hadar Baron](https://www.linkedin.com/in/hadar-baron-34527618b/)
 
 ### Supervisor: [Dr. Amit Dvir](https://www.ariel.ac.il/wp/amitd/)
