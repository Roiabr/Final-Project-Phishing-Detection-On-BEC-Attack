import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords

from wordcloud import WordCloud
import matplotlib.pyplot as plt


def getTheWordHeader(data):
    headers = data.email_header.str.cat(sep=' ')
    # function to split text into word
    tokens = word_tokenize(headers)
    vocabulary = set(tokens)
    frequency_dist = nltk.FreqDist(tokens)
    sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:50]
    stop_words = set(stopwords.words('english'))
    stop_words.update(("<", ">", "@", ".", ".com", ",", ":", ";", "''"))
    tokens = [w for w in tokens if not w in stop_words]
    print(tokens)
    wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def CountVector(X_train, X_test):
    stop = set(stopwords.words('english'))
    stop.update(("<", ">", "@", ".", "com"))
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stop)
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)
    return train_vectors, test_vectors, vectorizer


def TfIdf(X_train, X_test):
    stop = set(stopwords.words('english'))
    stop.update(("<", ">", "@", ".", "com"))
    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stop)
    train_vectors2 = tfidfconverter.fit_transform(X_train)
    test_vectors2 = tfidfconverter.transform(X_test)
    return train_vectors2, test_vectors2, tfidfconverter
